from rag.services.hybrid_search import get_hybrid_engine
from google.genai import types

OLLAMA_RAG_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": "Search the KrishiPlan agronomy knowledge base. Call this with a fully self-contained, standalone query (resolve all pronouns and prior context into the query text itself).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Standalone, context-resolved search query"
                }
            },
            "required": ["query"]
        }
    }
}

# Wrap the schema in the SDK's types.Tool and types.FunctionDeclaration
RAG_SEARCH_TOOL_SCHEMA = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="rag_search",
            description="Search the KrishiPlan agronomy knowledge base. Call this with a fully self-contained, standalone query (resolve all pronouns and prior context into the query text itself).",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING", 
                        "description": "Standalone, context-resolved search query"
                    }
                },
                "required": ["query"]
            }
        )
    ]
)

def execute_rag_search(query: str, top_k: int = 10):
    """
    Executes a hybrid search for the given standalone query.
    Returns the chunks metadata.
    """
    engine = get_hybrid_engine()
    results = engine.hybrid_search(query, final_k=top_k)
    return results
