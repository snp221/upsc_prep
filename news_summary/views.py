from django.shortcuts import render
from django.conf import settings

from .services import get_google_sheet_data, organize_data_by_date
from .models import CaMainsDigest, Todo

import os
import base64
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from datetime import date

# Load environment variables for OpenAI, etc.
load_dotenv()


def home(request):
    return render(request, 'news_summary/home.html')


def current_affairs(request):
    data = get_google_sheet_data()
    date_dict = organize_data_by_date(data)
    # show only the latest 10 dates
    dates = sorted(date_dict.keys(), reverse=True)[:10]
    return render(request, 'news_summary/current_affairs.html', {'dates': dates})


def affairs_by_date(request, date):
    data = get_google_sheet_data()
    date_dict = organize_data_by_date(data)
    summaries = date_dict.get(date, [])
    # Pre-fetch any existing mains digests for this date keyed by topic
    digests = CaMainsDigest.objects.filter(date=date).values(
        "topic", "mains_points", "mains_question", "subject", "subtopic"
    )
    digest_map = {d["topic"]: d for d in digests}

    # Attach digest info to each summary row if available
    enriched = []
    for item in summaries:
        topic = item.get("Topic") or item.get("topic") or ""
        key = topic
        digest = digest_map.get(key)
        item_copy = item.copy()
        if digest:
            item_copy["mains_points"] = digest["mains_points"]
            item_copy["mains_question"] = digest["mains_question"]
            item_copy["subject"] = digest["subject"]
            item_copy["subtopic"] = digest["subtopic"]
        enriched.append(item_copy)

    return render(
        request,
        'news_summary/affairs_by_date.html',
        {
            'date': date,
            'summaries': enriched,
        },
    )


def ca_generate_digest(request, date):
    """Generate or refresh editorial-to-mains digest for all CA items on a given date."""

    from datetime import datetime as _dt

    # Normalise date string to date object
    try:
        date_obj = _dt.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return render(
            request,
            "news_summary/ca_digest_status.html",
            {"error": "Invalid date format. Use YYYY-MM-DD."},
        )

    data = get_google_sheet_data()
    date_dict = organize_data_by_date(data)
    summaries = date_dict.get(date, [])

    if not summaries:
        return render(
            request,
            "news_summary/ca_digest_status.html",
            {"error": "No current affairs found for this date."},
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return render(
            request,
            "news_summary/ca_digest_status.html",
            {"error": "OPENAI_API_KEY not set. Please configure it first."},
        )

    openai_client = OpenAI(api_key=api_key)

    created_or_updated = []

    for row in summaries:
        topic = row.get("Topic") or row.get("topic") or ""
        summary_text = row.get("Summary") or row.get("summary") or ""

        if not topic or not summary_text:
            continue

        # Prompt model to create mains points and one mains question
        system_prompt = (
            "You are an expert UPSC mains mentor. "
            "Convert editorials/current affairs summaries into exam-ready GS mains material."
        )

        user_prompt = (
            "Current affairs issue title: "
            + topic
            + "\n\nSummary:\n"
            + summary_text
            + "\n\nTasks (plain text only, no Markdown):\n"
            "1) In 3-5 short bullet-style lines, write mains-oriented points (causes, implications, GS linkages).\n"
            "2) Then give ONE GS mains-style question (mention Paper and topic in brackets, e.g. [GS2 – Polity – Federalism])."
        )

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=400,
            temperature=0.6,
        )

        text = (response.choices[0].message.content or "").replace("**", "").strip()

        # Simple split: mains points vs question (assume last line is the question)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue

        mains_question = lines[-1]
        mains_points = "\n".join(lines[:-1]) if len(lines) > 1 else ""

        digest, _ = CaMainsDigest.objects.update_or_create(
            date=date_obj,
            topic=topic,
            defaults={
                "summary": summary_text,
                "mains_points": mains_points,
                "mains_question": mains_question,
            },
        )

        created_or_updated.append(digest)

    return render(
        request,
        "news_summary/ca_digest_status.html",
        {"date": date_obj, "count": len(created_or_updated)},
    )


def todo_list(request):
    """Simple todo list for users to track UPSC study tasks."""

    if request.method == "POST":
        title = (request.POST.get("title") or "").strip()
        description = (request.POST.get("description") or "").strip()
        subject = (request.POST.get("subject") or "").strip()
        if title:
            Todo.objects.create(title=title, description=description, subject=subject)

    todos = Todo.objects.all()
    return render(
        request,
        "news_summary/todo_list.html",
        {"todos": todos},
    )


def todo_toggle(request, pk):
    """Toggle completion state of a todo item."""

    from django.shortcuts import get_object_or_404, redirect

    todo = get_object_or_404(Todo, pk=pk)
    todo.completed = not todo.completed
    todo.save(update_fields=["completed"])
    return redirect("todo_list")


def upsc_chat(request):
    """Web view that uses ChromaDB + OpenAI to answer UPSC questions."""
    # Retrieve chat history from the session and reset it when the day changes
    today_str = date.today().isoformat()
    last_date = request.session.get("upsc_chat_last_date")
    chat_history = request.session.get("upsc_chat_history", [])

    if last_date != today_str:
        chat_history = []
        request.session["upsc_chat_history"] = chat_history
        request.session["upsc_chat_last_date"] = today_str
        request.session.modified = True

    answer = None
    sources = []
    error = None

    if request.method == "POST":
        question = (request.POST.get("question") or "").strip()

        if not question:
            error = "Please enter a question."
        else:
            try:
                # Initialize ChromaDB client
                client = chromadb.PersistentClient(
                    path=os.path.join(settings.BASE_DIR, "chroma_db"),
                    settings=Settings(anonymized_telemetry=False),
                )

                # Initialize embedding model
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

                # Build query embedding for the latest user question
                query_embedding = embedding_model.encode([question]).tolist()[0]

                # Initialize OpenAI client
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "OPENAI_API_KEY not set. Please add it to your .env file."
                    )

                openai_client = OpenAI(api_key=api_key)

                # Get collections
                collections = {
                    "books": client.get_collection(name="books"),
                    "essays": client.get_collection(name="essays"),
                    "ncert": client.get_collection(name="ncert"),
                    "pyq": client.get_collection(name="pyq"),
                }

                # Collect relevant content (do not over-filter by distance)
                all_content = []
                for collection_name, collection in collections.items():
                    try:
                        results = collection.query(
                            query_embeddings=[query_embedding],
                            n_results=4,
                            include=["documents", "metadatas", "distances"],
                        )

                        if results["documents"] and results["documents"][0]:
                            for doc, metadata, distance in zip(
                                results["documents"][0],
                                results["metadatas"][0],
                                results["distances"][0],
                            ):
                                source = metadata.get("source", "") if metadata else ""
                                all_content.append(
                                    {
                                        "content": doc,
                                        "source": f"{collection_name.upper()}: {source}",
                                        "relevance": 1 - distance,
                                    }
                                )
                    except Exception as e:  # pragma: no cover - logging only
                        error = f"Error searching {collection_name}: {e}"

                if not all_content and not error:
                    error = "No relevant content found in knowledge base."

                if all_content:
                    # Sort by relevance and take top 5
                    all_content.sort(key=lambda x: x["relevance"], reverse=True)
                    top_content = all_content[:5]
                    sources = top_content

                    # Prepare context for LLM
                    context_text = "=== RELEVANT UPSC STUDY MATERIAL ===\n\n"
                    for i, item in enumerate(top_content, 1):
                        context_text += f"Source {i}: {item['source']}\n"
                        context_text += f"Content: {item['content'][:800]}...\n\n"

                    # Build conversation messages with history
                    system_prompt = (
                        "You are an expert UPSC preparation tutor. Use the provided "
                        "study materials to give concise, structured answers.\n\n"
                        "Guidelines (very important):\n"
                        "- Keep answers short and focused (around 120–180 words)\n"
                        "- Use simple structure that is easy to read in plain text\n"
                        "- Prefer numbered or hyphenated points, but DO NOT use Markdown bold/italics (no **, no *)\n"
                        "- Include 2–4 key points or dimensions of the answer, not long essays\n"
                        "- Cite which sources you used when relevant\n"
                        "- If information is insufficient, briefly mention what's missing\n"
                        "- Output must be plain text only, without any Markdown formatting"
                    )

                    # Temporary history including the new user question
                    temp_history = chat_history + [
                        {"role": "user", "content": question}
                    ]

                    # Limit history to last 10 turns to keep prompts manageable
                    recent_history = temp_history[-10:]

                    messages = [{"role": "system", "content": system_prompt}]
                    for m in recent_history[:-1]:
                        messages.append({"role": m["role"], "content": m["content"]})

                    # Latest user message augmented with RAG context
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Question: {question}\n\n{context_text}\n\n"
                                "Using the above study material and our previous conversation, "
                                "provide a comprehensive UPSC-focused answer."
                            ),
                        }
                    )

                    response = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7,
                    )

                    answer = response.choices[0].message.content or ""
                    # Strip any accidental Markdown bold markers
                    answer = answer.replace("**", "").strip()

                    # Persist updated chat history in the session
                    chat_history = temp_history + [
                        {"role": "assistant", "content": answer}
                    ]
                    request.session["upsc_chat_history"] = chat_history
                    request.session["upsc_chat_last_date"] = today_str
                    request.session.modified = True

            except Exception as exc:
                if error is None:
                    error = str(exc)

    context = {
        "question": "",
        "answer": answer,
        "sources": sources,
        "error": error,
        "chat_history": chat_history,
    }

    return render(request, "news_summary/upsc_chat.html", context)


def answer_eval(request):
    """Evaluate a 150-word UPSC answer using RAG + general knowledge."""

    question = ""
    user_answer = ""
    eval_result = None
    error = None

    if request.method == "POST":
        question = (request.POST.get("question") or "").strip()
        user_answer = (request.POST.get("answer") or "").strip()

        try:
            # Initialize embedding model
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            # If a handwritten image is uploaded, use OpenAI vision to transcribe it
            uploaded_image = request.FILES.get("handwritten_image")
            if uploaded_image:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "OPENAI_API_KEY not set. Please add it to your .env file."
                    )

                openai_client = OpenAI(api_key=api_key)
                image_bytes = uploaded_image.read()
                b64_image = base64.b64encode(image_bytes).decode("utf-8")

                vision_response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You transcribe handwritten UPSC mains answers into clean plain text, preserving meaning.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Transcribe this handwritten UPSC answer into plain text.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_image}",
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=400,
                    temperature=0.2,
                )

                ocr_text = vision_response.choices[0].message.content or ""
                user_answer = ocr_text.strip()

            if not question or not user_answer:
                error = "Please provide a question and either typed answer or a handwritten image."
            else:
                # Check semantic relevance between question and answer
                q_vec = np.array(embedding_model.encode([question])[0])
                a_vec = np.array(embedding_model.encode([user_answer])[0])
                sim = float(
                    q_vec @ a_vec
                    / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec) + 1e-8)
                )

                # If answer is largely off-topic, hard-cap marks at 0/15
                if sim < 0.5:
                    eval_result = (
                        "Marks: 0/15\n"
                        "Answer is not relevant to the question. Focus on the exact demand, "
                        "key concepts, and UPSC-style structure for this question."
                    )
                else:
                    # Initialize ChromaDB client
                    client = chromadb.PersistentClient(
                        path=os.path.join(settings.BASE_DIR, "chroma_db"),
                        settings=Settings(anonymized_telemetry=False),
                    )

                    # Initialize OpenAI client (if not already created for OCR)
                    if "openai_client" not in locals():
                        api_key = os.getenv("OPENAI_API_KEY")
                        if not api_key:
                            raise RuntimeError(
                                "OPENAI_API_KEY not set. Please add it to your .env file."
                            )
                        openai_client = OpenAI(api_key=api_key)

                    # Get collections
                    collections = {
                        "books": client.get_collection(name="books"),
                        "essays": client.get_collection(name="essays"),
                        "ncert": client.get_collection(name="ncert"),
                        "pyq": client.get_collection(name="pyq"),
                    }

                    # Build query embedding from the question
                    query_embedding = embedding_model.encode([question]).tolist()[0]

                    # Collect relevant content
                    all_content = []
                    for collection_name, collection in collections.items():
                        try:
                            results = collection.query(
                                query_embeddings=[query_embedding],
                                n_results=4,
                                include=["documents", "metadatas", "distances"],
                            )

                            if results["documents"] and results["documents"][0]:
                                for doc, metadata, distance in zip(
                                    results["documents"][0],
                                    results["metadatas"][0],
                                    results["distances"][0],
                                ):
                                    source = (
                                        metadata.get("source", "") if metadata else ""
                                    )
                                    all_content.append(
                                        {
                                            "content": doc,
                                            "source": f"{collection_name.upper()}: {source}",
                                            "relevance": 1 - distance,
                                        }
                                    )
                        except Exception:
                            # Ignore per-collection errors, still evaluate using what we have
                            continue

                    # Sort by relevance and take top few for context (if any)
                    context_text = ""
                    if all_content:
                        all_content.sort(key=lambda x: x["relevance"], reverse=True)
                        top_content = all_content[:5]
                        context_text = "=== RELEVANT UPSC STUDY MATERIAL ===\n\n"
                        for i, item in enumerate(top_content, 1):
                            context_text += f"Source {i}: {item['source']}\n"
                            context_text += (
                                f"Content: {item['content'][:800]}...\n\n"
                            )

                    # System prompt: short, structured, no Markdown, with marks
                    system_prompt = (
                        "You are a strict but fair UPSC mains examiner. "
                        "Use the provided study material (if any) plus your own UPSC-level "
                        "knowledge to evaluate the candidate's answer.\n\n"
                        "Answer style requirements (very important):\n"
                        "- Give marks out of 15 in the first line, in the form: Marks: X/15\n"
                        "- Then give 3–6 short points grouped as strengths and weaknesses\n"
                        "- Keep the language concise and exam-oriented\n"
                        "- DO NOT use Markdown, asterisks, or bold/italics (no **, no *)\n"
                        "- Prefer plain text only\n"
                        "- If context is missing, still evaluate based on general knowledge"
                    )

                    user_message = (
                        "Question: "
                        + question
                        + "\n\nCandidate answer (approx. 150 words):\n"
                        + user_answer
                    )

                    if context_text:
                        user_message += "\n\n" + context_text

                    response = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        max_tokens=600,
                        temperature=0.6,
                    )

                    raw_eval = response.choices[0].message.content or ""
                    eval_result = raw_eval.replace("**", "").strip()
        except Exception as exc:
            error = str(exc)

    context = {
        "question": question,
        "answer_text": user_answer,
        "eval_result": eval_result,
        "error": error,
    }

    return render(request, "news_summary/answer_eval.html", context)