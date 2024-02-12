import streamlit as st
import cv2 as cv
from scripts.img_road_lane_detection import detect_lanes
import moviepy.editor as moviepy
from tempfile import NamedTemporaryFile
import os


def main():
    st.header("Road Lane Detector")
    st.markdown(
        """:dart: Road Lane Detector is an application to detect road lanes present in a video feed coming from the front camera of any autonomous vehicle."""
    )

    st.info("Upload the video to detect road lanes")
    uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi'])
    
    if uploaded_file:
        vid = uploaded_file.name

        # Save video to disk
        with open(vid, 'wb') as f:
            f.write(uploaded_file.read())

        # Read the video
        st_video = open(vid, 'rb')
        video_bytes = st_video.read()
        
        # Show uploaded video
        st.write("Uploaded Video")
        st.video(video_bytes)

        # Generate detections on the video
        detection_video = generate_detection_video(vid)

        try:
            clip = moviepy.VideoFileClip(detection_video)
            out_file = "out.mp4"
            clip.write_videofile(out_file)
            st_video = open(out_file, "rb")
            video_bytes = st_video.read()
            st.write("Detected Video")
            st.video(video_bytes)
            clip.close()

            # Delete the videos
            st_video.close()
            os.unlink(vid)
            os.unlink(detection_video)
            os.unlink(out_file)

        except OSError:
            ''
    

def generate_detection_video(video_path, fsize=(640, 480)):
    # Access the video
    cap =  cv.VideoCapture(video_path)

    if not cap.isOpened():
        return None
    
    # Frame per second
    fps = cap.get(cv.CAP_PROP_FPS)

    # fourcc parameter
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    out = cv.VideoWriter("detection.mp4", fourcc, float(fps), fsize)

    while True:
        ret, video_frame = cap.read()
        
        if not ret:
            break
        
        detection = detect_lanes(video_frame)

        detection = cv.cvtColor(detection, cv.COLOR_BGR2RGB)
        detection = cv.resize(detection, fsize)
        # Save the frame in out
        out.write(detection)
        

    cap.release()

    return "detection.mp4"


if __name__=="__main__":
    main()