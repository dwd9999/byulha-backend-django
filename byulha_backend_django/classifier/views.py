import json

import boto3
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from byulha_backend_django.settings import S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_NAME
from classifier.image_model import model_setting


def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


@csrf_exempt
def post(request):
    try:
        request_json = json.loads(request.body)
        print(get_client_ip(request))

        # S3 접근
        s3_client = boto3.client('s3', aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY)
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key='/' + request_json['fileId'])

        # 모델
        category_percent = model_setting(response['Body'])

        # 데이터 json 형식 변환
        data_to_send = {
            'nickname': request_json['nickname'],
            'fileId': request_json['fileId'],
            'category_percent': category_percent,
        }
        print(data_to_send)

        # 데이터 반환
        return JsonResponse(data_to_send)

    # 오류 발생 시 Response 500
    except Exception as e:
        print(f'error: {e}')
        return JsonResponse({"error": str(e)}, status=500)
