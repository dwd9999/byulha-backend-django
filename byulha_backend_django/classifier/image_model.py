import os

import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model


def img_model(image):
    np.set_printoptions(suppress=True)
    class_names = open(f"{os.getcwd()}/model/labels.txt", "r", encoding='UTF8').readlines()
    model = load_model(f"{os.getcwd()}/model/keras_model.h5", compile=False)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score
