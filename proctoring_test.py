import cv2
from facial_detections import detectFace
from gaze_tracking import gaze_tracking
from object_detection import detectObject

print("Starting Proctoring System Test...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Resize for performance
    frame = cv2.resize(frame, (640, 480))

    # -----------------------------
    # FACE DETECTION
    # -----------------------------
    face_data = detectFace(frame)
    frame = face_data["annotated_frame"]

    # -----------------------------
    # GAZE TRACKING
    # -----------------------------
    gaze_data = gaze_tracking(frame)
    gaze = gaze_data["gaze"]

    cv2.putText(frame, f"Gaze: {gaze}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2)

    # -----------------------------
    # OBJECT DETECTION
    # -----------------------------
    object_data = detectObject(frame)
    frame = object_data["annotated_frame"]

    if object_data["violation"]:
        cv2.putText(frame,
                    f"Object Violation: {object_data['violation']}",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)

    # -----------------------------
    # DISPLAY
    # -----------------------------
    cv2.imshow("AI Proctoring System Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()