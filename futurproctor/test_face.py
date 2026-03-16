import cv2
import face_recognition

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    faces = face_recognition.face_locations(frame)

    for top, right, bottom, left in faces:

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()