import cv2
import time
import image_processing as ip
import mediapipe_processing as mp
from flask import Response, Flask


app = Flask( __name__ )
cap = cv2.VideoCapture(1)


def get_image():
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, image_org = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None
    return image_org


def encode_video():
    while True:
        current_time = time.time()
        image_org = get_image()

        # image = image_org

        # image = ip.to_gray(image_org)
        # image = ip.bgr2rgb(image_org)
        # image = ip.face_detection(image_org)

        # image = mp.face_detection_tracking(image_org)
        image = mp.face_mesh_tracking(image_org)
        image = mp.hands_tracking(image_org)
        # image = mp.holistic_tracking(image_org)
        # image = mp.object_detection(image_org)


        # FPS Counter
        next_time = time.time()
        fps = 1 / (next_time - current_time)

        cv2.putText(image, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (161, 252, 3), 3)

        return_key, encoded = cv2.imencode(".jpg", image)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +  encoded.tobytes()  + b'\r\n\r\n' )


@app.route( "/" )
def streamFrames():
    return Response( encode_video(), mimetype="multipart/x-mixed-replace; boundary=frame" )


if __name__ == "__main__":
    app.run( "0.0.0.0", port="8000" )