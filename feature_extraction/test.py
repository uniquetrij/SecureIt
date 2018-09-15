import cv2
from keras.applications import resnet50
import numpy as np

model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
cap = cv2.VideoCapture(-1)
image_shape = (224, 224, 3)
if __name__ == '__main__':
    while True:
        ret, image = cap.read()
        if not ret:
            continue

        image = cv2.resize(image, tuple(image_shape[:2][::-1]))
        print(image.shape)
        image = np.expand_dims(image, axis=0)
        print(image.shape)
        f_vec = model.predict(image)
        print(f_vec)
