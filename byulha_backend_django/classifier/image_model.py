import os

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import tensorflow as tf
import numpy as np


def img_model(image):
    np.set_printoptions(suppress=True)

    # Tensorflow CPU 설정
    tf.config.threading.set_inter_op_parallelism_threads(2)

    # 모델 가져오기
    model = load_model(f"{os.getcwd()}/model/keras_model.h5", compile=False)

    # 클래스 별 라벨 추출
    class_names = open(f"{os.getcwd()}/model/labels.txt", "r", encoding='UTF8').readlines()
    class_names = [x.strip() for x in class_names]

    # 모델
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)

    # 내림차 정렬 후 상위 3개 항목 추출
    top_3_indices = np.argsort(prediction.ravel())[-3:][::-1]
    top_3_categories = [class_names[x] for x in top_3_indices]

    # 상위 3개 항목 모델 결과
    top_3_values = prediction.ravel()[top_3_indices]

    return "-".join([x + ":" + str(round(y * 100, 3)) for x, y in zip(top_3_categories, top_3_values)])
