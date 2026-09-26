from django.contrib import admin

from .models import ChatSession, ChatTurn

@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'user', 'created_at')
    search_fields = ('session_id', 'user__email')
    ordering = ('-created_at',)

@admin.register(ChatTurn)
class ChatTurnAdmin(admin.ModelAdmin):
    list_display = ('session', 'role', 'created_at', 'short_content')
    list_filter = ('role', 'created_at')
    search_fields = ('content', 'session__session_id')
    ordering = ('-created_at',)

    def short_content(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    short_content.short_description = 'Content'
