import cv2
import numpy as np

DEBUG = True

face_cascade = cv2.CascadeClassifier('./cascade/haarcascade_frontalface_default.xml')


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def face_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (161, 252, 3), 4)

    return image