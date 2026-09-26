from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework import generics
from .models import Land
from .serializers import LandSerializer


class HandleUser(APIView):
    """
    Standard API view for processing basic user-centric land inquiries.
    """
    permission_classes = [AllowAny]

    def get(self, request):
        data = {
            "status": "online",
            "message": "Land planning engine is operational. Please initiate location selection."
        }
        return Response(data, status=status.HTTP_200_OK)


class LandListCreateView(generics.ListCreateAPIView):
    """
    GET: List all fields owned by the authenticated farmer.
    POST: Create a new field for the authenticated farmer.
    """
    serializer_class = LandSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Land.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class LandDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    GET, PUT, PATCH, DELETE a specific field owned by the farmer.
    """
    serializer_class = LandSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Land.objects.filter(user=self.request.user)
