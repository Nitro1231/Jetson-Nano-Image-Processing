import cv2
import time
import mediapipe as mp


cap = cv2.VideoCapture(1)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def pose_tracking(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pose_landmarks = results.pose_landmarks

    if pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS, #POSE_CONNECTIONS, POSE_WORLD_LANDMARKS
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        mp_drawing.plot_landmarks(
            results.pose_world_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    
    return image, pose_landmarks


def main():
    while True:
        start_time = time.time()
        ret, image_org = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        image, data = pose_tracking(image_org)

        if False:
            if data != None:
                print(type(data.landmark))
                print(len(data.landmark))
                break

        # FPS Counter
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(image, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (161, 252, 3), 3)

        # Display the resulting frame
        cv2.imshow('image', image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    main() # main loop.

    cap.release()
    cv2.destroyAllWindows()
