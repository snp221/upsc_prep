FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir django==6.0 gunicorn

COPY upsc_prep /app

# Collect static files into /app/staticfiles during build (optional)
RUN python manage.py collectstatic --noinput || true

EXPOSE 8000

ENV DJANGO_SETTINGS_MODULE=upsc_prep.settings

CMD ["gunicorn", "upsc_prep.wsgi:application", "--bind", "0.0.0.0:8000"]
