import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO('yolov8n.pt')

# Path to the video file
video_path = r"C:\Users\dell\OneDrive\Desktop\test2.mp4"
# Open video file
cap = cv2.VideoCapture(video_path)

# Check if the video has been opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop to process the video frames
while True:
    ret, frame = cap.read()

    # Break the loop if no frame is returned
    if not ret:
        print("End of video.")
        break

    # Perform object tracking
    results = model.track(frame, persist=True)

    # Plot the results on the frame
    frame_ = results[0].plot()

    # Display the frame
    cv2.imshow('frame', frame_)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
