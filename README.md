# Speed Estimation and Tracking of Vehicle using YOLOv8 and webapp for Realtime Estimation

This project is part of Task 2 of the ITSOLERA internship. It involves creating a Streamlit app that allows users to upload a video, draw a line for speed estimation, and process the video using the YOLO model for object detection and a custom speed estimator. The processed video can be downloaded by the user.

## Features

- Upload a video file in MP4 or AVI format.
- Draw a line on the first frame of the video to set the region for speed estimation.
- Process the video with YOLO for object detection and estimate speeds of detected objects.
- Download the processed video.

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/video-processing-yolo.git
   cd video-processing-yolo
2. Run the Script:
    ```sh
   streamlit run main.py
## Usage
1. Upload a video file in MP4 or AVI format.
2. Click the "Draw Line" button. A new window will open displaying the first frame of the video.
3. Draw a line by clicking two points on the frame. Close the window after you make a line. (Just click on the screen where to start and where to end the line will automatically be displayed.
4. The video processing will start automatically.
5. #### Wait for the processing to complete.

### Once the video is processed, a download button will appear. Click the button to download the processed video.

## Dependencies
- streamlit: Streamlit library for creating the web app.
- opencv-python-headless: OpenCV library for video processing.
- ultralytics: YOLO model for object detection.

## Notes
- Ensure that the YOLO model file yolov8n.pt is available in the same directory as speed_estimator.py.
- The video processing uses the YOLO model for object detection and a custom speed estimator. Adjust the line points and model settings as needed for different use cases.
-  #### Note: after you make line just closed the window and the points will be on the screen to start processing

## Links
- Check my linkedin for more details:
  Connect with me on LinkedIn: [My LinkedIn Profile](www.linkedin.com/in/mammarali)
