import streamlit as st
import cv2
from ultralytics import YOLO, solutions
import tempfile
import os

# Streamlit app
st.title("Computer Vision Team(Beta): Speed Estimationa and Tracking of Vehicles")

# Video upload
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])

if uploaded_video is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_video_file:
        tmp_video_file.write(uploaded_video.read())
        tmp_video_path = tmp_video_file.name

    # Function to let the user draw a line on the video frame
    def draw_line_on_frame(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("Failed to read video")
            return None

        #frame = cv2.resize(frame, (640, 480))

        line_pts = []

        def draw_line(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                line_pts.append((x, y))
                if len(line_pts) == 2:
                    cv2.line(frame, line_pts[0], line_pts[1], (0, 255, 0), 2)
                    cv2.imshow("Draw Line", frame)

        cv2.imshow("Draw Line", frame)
        cv2.setMouseCallback("Draw Line", draw_line)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(line_pts) == 2:
            return line_pts
        else:
            st.error("Please draw a complete line")
            return None

    # Function to process the video
    def process_video(video_path, line_pts):
        model = YOLO("yolov8n.pt")
        names = model.model.names

        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        output_path = os.path.join(tempfile.gettempdir(), "speed_estimation.avi")
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init speed-estimation obj
        speed_obj = solutions.SpeedEstimator(
            reg_pts=line_pts,
            names=names,
            view_img=False,
        )

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            tracks = model.track(im0, persist=True, show=False)

            im0 = speed_obj.estimate_speed(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        return output_path

    # Let the user draw the line
    st.write("Draw a line for speed estimation")
    if st.button("Draw Line"):
        line_pts = draw_line_on_frame(tmp_video_path)
        if line_pts is not None:
            st.write(f"Line points: {line_pts}")

            # Process the uploaded video
            with st.spinner("Processing video..."):
                processed_video_path = process_video(tmp_video_path, line_pts)

            st.success("Video processed successfully!")

            # Provide download link for the processed video
            with open(processed_video_path, "rb") as processed_video_file:
                processed_video_bytes = processed_video_file.read()
                st.download_button(label="Download Processed Video", data=processed_video_bytes, file_name="processed_video.avi", mime="video/avi")

# Run the app using `streamlit run app.py`
