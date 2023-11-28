from django.urls import re_path

from classifier import consumers

websocket_urlpatterns = [
    re_path(r'ws/(?P<client_name>\w+)/$', consumers.ChatConsumer.as_asgi()),
]
