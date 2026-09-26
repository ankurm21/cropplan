from django.urls import path
from .views import RAGQueryView, RAGStatusView
from .views_v2 import RAGQueryV2View

urlpatterns = [
    path('query/', RAGQueryView.as_view(), name='rag-query'),
    path('query', RAGQueryView.as_view(), name='rag-query-noslash'),
    path('query/v2/', RAGQueryV2View.as_view(), name='rag-query-v2'),
    path('query/v2', RAGQueryV2View.as_view(), name='rag-query-v2-noslash'),
    path('status/', RAGStatusView.as_view(), name='rag-status'),
    path('status', RAGStatusView.as_view(), name='rag-status-noslash'),
]
