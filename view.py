import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 for close-range, 1 for full-range
    min_detection_confidence=0.5
)

# Initialize camera
cam = cv2.VideoCapture(1)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            # Calculate bounding box coordinates
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            box_width = int(bbox.width * w)
            box_height = int(bbox.height * h)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_min + box_width, y_min + box_height),
                (0, 255, 0),
                2
            )

    # Display the frame
    cv2.imshow("Face Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
face_detection.close()