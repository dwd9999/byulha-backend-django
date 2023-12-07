import json

import boto3
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from byulha_backend_django.settings import S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_NAME
from classifier.image_model import img_model


@csrf_exempt
def post(request):
    try:
        request_json = json.loads(request.body)

        # S3 접근
        s3_client = boto3.client('s3', aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY)
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key='/' + request_json['fileId'])
        img_model(response['Body'])

        # 모델
        category_name = 'SPORTY:50,asdf:49.7,werscv:30'

        # 데이터 json 형식 변환
        data_to_send = {
            'nickname': request_json['nickname'],
            'fileId': request_json['fileId'],
            'category_percent': category_name,
        }
        print(data_to_send)

        # 데이터 반환
        return JsonResponse(data_to_send)

    except Exception as e:
        print(f'error: {e}')
        return JsonResponse({"error": str(e)}, status=500)
