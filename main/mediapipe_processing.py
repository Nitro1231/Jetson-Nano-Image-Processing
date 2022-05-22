import cv2
import mediapipe as mp


# https://google.github.io/mediapipe/


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

mp_objectron = mp.solutions.objectron
objectron = mp_objectron.Objectron()


def face_detection_tracking(image):
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    detection = results.detections

    if detection:
        for face in detection:
            mp_drawing.draw_detection(image, face)
    
    return image


def face_mesh_tracking(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    face_landmarks = results.multi_face_landmarks

    if face_landmarks:
        for face in face_landmarks:
            mp_drawing.draw_landmarks(image, face, mp_face_mesh.FACEMESH_TESSELATION)
            # mp_drawing.draw_landmarks(image, face, mp_face_mesh.FACEMESH_CONTOURS)
            # mp_drawing.draw_landmarks(image, face, mp_face_mesh.FACEMESH_IRISES)
    
    return image


def hands_tracking(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    hand_landmarks = results.multi_hand_landmarks

    if hand_landmarks:
        for hand in hand_landmarks: # each hand; can be more than one.
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
    
    return image


def holistic_tracking(image):
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    return image


def object_detection(image):
    results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    detected_objects = results.detected_objects

    if detected_objects:
        for detected_object in detected_objects:
            mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)
    
    return image


def pose_tracking(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pose_landmarks = results.pose_landmarks

    if pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    return image
