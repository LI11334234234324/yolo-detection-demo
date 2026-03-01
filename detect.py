"""
Real-time Object Detection Demo
Author: 李佳睿
Description: A simple real-time object detection demo using YOLOv8 and OpenCV.
             Related to AI-driven perception research in robotics.
"""

from ultralytics import YOLO
import cv2
import time


def run_detection(source=0, model_name="yolov8n.pt", conf=0.5):
    """
    Run real-time object detection.

    Args:
        source: Camera index (0 = default webcam) or video file path
        model_name: YOLOv8 model variant (n=nano, s=small, m=medium)
        conf: Confidence threshold (0~1)
    """
    # Load model (auto-downloads on first run)
    model = YOLO(model_name)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Cannot open camera/video source.")
        return

    print(f"[INFO] Running detection with {model_name}, confidence threshold={conf}")
    print("[INFO] Press 'q' to quit, 's' to save a screenshot.")

    fps_list = []

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame, conf=conf, verbose=False)

        # Draw bounding boxes
        annotated = results[0].plot()

        # Calculate and display FPS
        fps = 1.0 / (time.time() - t_start)
        fps_list.append(fps)
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Display detection count
        n_objects = len(results[0].boxes)
        cv2.putText(
            annotated,
            f"Objects: {n_objects}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        cv2.imshow("YOLOv8 Real-time Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"[INFO] Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"[INFO] Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    # Change source=0 to a video file path if no webcam available
    # e.g., source="test_video.mp4"
    run_detection(source=0, model_name="yolov8n.pt", conf=0.5)