import uuid
from django.db import models
from django.conf import settings

class ChatSession(models.Model):
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.CASCADE)
    crop_plan_context = models.JSONField(null=True, blank=True, help_text="Stored crop plan context for this session")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Session {self.session_id}"

class ChatTurn(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="turns")
    role = models.CharField(max_length=15, choices=[('user', 'User'), ('assistant', 'Assistant')])
    content = models.TextField()
    rewritten_query = models.TextField(null=True, blank=True, help_text="The standalone query the model rewrote to search with")
    retrieved_chunk_ids = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role.capitalize()} at {self.created_at}"
