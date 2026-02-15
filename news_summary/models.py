from django.db import models


class CaMainsDigest(models.Model):
	"""Cached mains-oriented summary for a single current affairs item."""

	date = models.DateField()
	topic = models.CharField(max_length=300)
	summary = models.TextField()

	# Classification based on your UPSC topic list
	subject = models.CharField(max_length=100, blank=True)
	subtopic = models.CharField(max_length=150, blank=True)

	mains_points = models.TextField(help_text="Bullet-style mains points", blank=True)
	mains_question = models.TextField(help_text="One mains-style question", blank=True)

	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		verbose_name = "CA Mains Digest"
		verbose_name_plural = "CA Mains Digests"
		indexes = [
			models.Index(fields=["date"]),
			models.Index(fields=["subject", "subtopic"]),
		]

	def __str__(self) -> str:  # pragma: no cover - trivial
		return f"{self.date} - {self.topic[:60]}"  # type: ignore[index]


class Todo(models.Model):
	"""Simple UPSC-oriented todo item for users."""

	title = models.CharField(max_length=200)
	description = models.TextField(blank=True)
	subject = models.CharField(max_length=100, blank=True)
	completed = models.BooleanField(default=False)
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		ordering = ["-created_at"]

	def __str__(self) -> str:  # pragma: no cover - trivial
		return self.title

