from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
import logging
from .services.v2.rag_engine_v2 import RAGEngineV2
import uuid

logger = logging.getLogger(__name__)

class RAGQueryV2View(APIView):
    """
    POST /api/rag/query/v2/
    Body: { "query": "...", "session_id": "..." (optional), "plan_context": {...} (optional) }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        query_text = request.data.get("query", "").strip()
        session_id = request.data.get("session_id")
        plan_context = request.data.get("plan_context")

        if not query_text:
            return Response(
                {"error": "Query text is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )
            
        if not session_id:
            session_id = str(uuid.uuid4())

        try:
            engine = RAGEngineV2()
            result = engine.query(
                session_id=session_id,
                query=query_text,
                user=request.user,
                crop_plan_context=plan_context
            )

            if not result:
                return Response(
                    {"error": "Failed to generate response. Please try again."},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

            return Response({
                "session_id": session_id,
                "answer": result["answer"],
                "rewritten_query": result.get("rewritten_query"),
                "sources": result.get("sources", []),
                "chunks_used": result.get("chunks_used", 0)
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"RAG Engine V2 error: {e}", exc_info=True)
            return Response(
                {"error": "An internal error occurred while processing your query."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
