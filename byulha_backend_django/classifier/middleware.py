import logging

from django.http import HttpResponseForbidden


class IpRestrictorMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        allowed_ips = ['127.0.0.1']  # 허용할 IP 주소 목록
        ip = request.META.get('REMOTE_ADDR')

        if ip not in allowed_ips:
            return HttpResponseForbidden("Access Denied")

        response = self.get_response(request)
        return response
