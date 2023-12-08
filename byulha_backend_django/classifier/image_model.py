import os

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np


def img_model(image):
    np.set_printoptions(suppress=True)

    model = load_model(f"{os.getcwd()}/model/keras_model.h5", compile=False)

    class_names = open(f"{os.getcwd()}/model/labels.txt", "r", encoding='UTF8').readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(image).convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)


    # argsort를 사용해 값을 내림차순으로 정렬하고 상위 3개의 인덱스 추출
    top_3_indices = np.argsort(prediction.ravel())[-3:][::-1]

    # 상위 3개의 값과 해당 인덱스
    top_3_values = prediction.ravel()[top_3_indices]
    print(list(zip(top_3_indices, top_3_values)))

    index = np.argmax(prediction)

    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score
