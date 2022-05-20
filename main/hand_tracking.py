import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


def hand_tracking(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    hand_landmarks = results.multi_hand_landmarks

    if hand_landmarks:
        for hand in hand_landmarks: # each hand; can be more than one.
            mp_draw.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
    
    return image