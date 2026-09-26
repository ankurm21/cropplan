import logging
from rag.models import ChatSession, ChatTurn
from .utility_v2 import call_llm_v2

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are **Krishi AI**, an expert precision-agriculture advisor exclusively for Indian farmers (Kisans).

RULES:
• You MUST call the `rag_search` tool if the user asks about agriculture, farming, crops, weather, or pests.
• DO NOT answer from your internal knowledge. Use the `rag_search` tool to fetch context first.
• Use ONLY the tools provided via the API tool-calling mechanism. NEVER generate tool calls as text, XML, JSON, or any other format in your response content.
• Call each tool AT MOST ONCE per user query. After receiving tool results, synthesize a final answer — do NOT call tools again.
• If the user says something conversational like "hello", reply normally without any tool.
• If numeric data (yield in quintals/ha, cost in INR, pH) is available in context, cite it strictly.
• Provide DETAILED and COMPREHENSIVE answers. Use formatting (bullet points, bold text) to make it easy to read.
• End with a one-line practical 'Krishi Tip'."""

class RAGEngineV2:
    def __init__(self):
        logger.info("RAGEngineV2 initialised")

    def query(self, session_id: str, query: str, user=None, crop_plan_context=None):
        # 1. Fetch or create session
        session, created = ChatSession.objects.get_or_create(session_id=session_id)
        if user and not session.user:
            session.user = user
            session.save()
        if crop_plan_context and not session.crop_plan_context:
            session.crop_plan_context = crop_plan_context
            session.save()

        # 2. Fetch history (last 6 turns, i.e., 3 exchanges)
        turns = session.turns.order_by('-created_at')[:6]
        turns = list(reversed(turns)) # chronological

        # 3. Build messages array
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for t in turns:
            messages.append({"role": t.role, "content": t.content})
        
        # Append current query
        messages.append({"role": "user", "content": query})

        # 4. Call LLM Router
        result = call_llm_v2(messages, crop_plan_context=session.crop_plan_context)

        if not result:
            return {
                "answer": "I apologise — I could not generate a response at this time. Please try again later.",
                "sources": [],
                "chunks_used": 0
            }

        answer = result["answer"]
        rewritten_query = result["rewritten_query"]
        chunks = result["chunks"]

        # 5. Persist the turn
        # Save user message
        ChatTurn.objects.create(
            session=session,
            role='user',
            content=query,
            rewritten_query=rewritten_query
        )
        
        # Extract chunk metadata for storage
        chunk_ids = [c.get("source", "unknown") for c in chunks] if chunks else []

        # Save assistant message
        ChatTurn.objects.create(
            session=session,
            role='assistant',
            content=answer,
            retrieved_chunk_ids=chunk_ids
        )

        # Map chunk metadata for the frontend (which expects 'document', 'topic', 'score')
        sources_for_frontend = []
        for c in chunks:
            sources_for_frontend.append({
                "document": c.get("document_name", ""),
                "topic": c.get("primary_topic", ""),
                "score": c.get("score", 0),
                "pages": c.get("page_numbers", "")
            })

        return {
            "answer": answer,
            "sources": sources_for_frontend,
            "chunks_used": len(chunks)
        }
