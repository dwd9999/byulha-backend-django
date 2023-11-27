import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application

import classifier.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'byulha_backend_django.settings')

application = ProtocolTypeRouter(
    {
        'http': get_asgi_application(),
        'websocket': AllowedHostsOriginValidator(
            AuthMiddlewareStack(
                URLRouter(
                    classifier.routing.websocket_urlpatterns
                )
            )
        ),
    }
)
