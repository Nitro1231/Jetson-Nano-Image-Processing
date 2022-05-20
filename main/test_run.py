import cv2
import numpy as np
import image_processing as ip

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()


while True:
    ret, image_org = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    # Our operations on the frame come here
    image = ip.face_detection(image_org)


    # Display the resulting frame
    cv2.imshow('image', image)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()