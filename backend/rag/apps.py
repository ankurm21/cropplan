from django.apps import AppConfig
import threading

class RagConfig(AppConfig):
    name = 'rag'

    def ready(self):
        # Pre-load the sentence transformer model in a background thread
        # so it doesn't block Django startup, but is ready by the time
        # the user makes their first RAG query.
        def preload_model():
            try:
                from .services.vector_store import _get_embed_fn, get_collection
                _get_embed_fn() # Downloads/loads the model into memory
                get_collection() # Initializes ChromaDB connections
                print("✅ RAG AI Models Pre-loaded successfully in background.")
            except Exception as e:
                print(f"⚠️ Failed to preload RAG models: {e}")

        # Start background thread
        thread = threading.Thread(target=preload_model, daemon=True)
        thread.start()
