"""
RAG API Views for KrishiPlan
──────────────────────────────
Endpoints:
    POST /api/rag/query/     → Auth-only knowledge base chat
    GET  /api/rag/status/    → Collection stats / health check (public)
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, AllowAny
import logging

logger = logging.getLogger(__name__)


class RAGQueryView(APIView):
    """
    POST /api/rag/query/
    Body: { "query": "...", "topic": "Soil" (optional), "plan_context": {...} (optional) }

    Authenticated only — the chatbot is a premium feature for logged-in farmers.
    If `plan_context` is provided, the chatbot can answer questions about the user's
    specific crop plan (weather, soil, farm details).
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        query_text = request.data.get("query", "").strip()
        topic = request.data.get("topic")            # optional metadata filter
        plan_context = request.data.get("plan_context")  # optional: user's plan data

        if not query_text:
            return Response(
                {"error": "Query text is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            from .services.rag_engine import RAGEngine

            engine = RAGEngine()

            # If user sent plan context, do a contextual query
            if plan_context and isinstance(plan_context, dict):
                result = engine.contextual_query(
                    user_query=query_text,
                    plan_context=plan_context,
                    topic_filter=topic,
                )
            else:
                result = engine.query(query_text, topic_filter=topic)

            return Response({
                "status": "success",
                "answer": result["answer"],
                "sources": result["sources"],
                "chunks_used": result["chunks_used"],
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("RAG query failed")
            return Response(
                {"error": f"RAG query failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class RAGStatusView(APIView):
    """
    GET /api/rag/status/
    Returns vector store stats. Public — used for dashboard indicators.
    """
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            from .services.rag_engine import RAGEngine
            stats = RAGEngine.health_check()
            return Response({"status": "ok", **stats})
        except Exception as e:
            logger.exception("RAG status check failed")
            return Response(
                {"status": "error", "error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
