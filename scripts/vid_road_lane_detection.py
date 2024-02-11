import cv2
from img_road_lane_detection import detect_lanes

def video_road_lane_detector(video_path, out_name, frame_size=(640, 480)):
    # Access the video
    cap =  cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Can't open video")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../.data/outputs/'+out_name, fourcc, 20.0, frame_size)
    
    while True:
        ret, video_frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        detection = detect_lanes(video_frame)

        detection = cv2.resize(detection, frame_size)

        # Write the new video with detections
        out.write(detection)

        cv2.imshow("Detection", detection)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()