import cv2
import numpy as np

def detect_lanes(img):
    # Convert it to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply canny edge detector
    edges = cv2.Canny(blur, 50, 150)

    # Masking
    mask = np.zeros_like(edges)
    height, width = edges.shape
    triangle = np.array([
        [(0, height), (width/2, height/2), (width, height)]
    ])
    mask = cv2.fillPoly(mask, np.array(triangle, dtype=np.int32), 255)
    mask = cv2.bitwise_and(edges, mask)

    # Default detection
    detection = img
    
    # Apply Hough Transform
    lines = cv2.HoughLinesP(mask, rho=6, theta=np.pi/180, threshold=140, minLineLength=40, maxLineGap=25)

    if lines is not None:
        line_img = np.zeros_like(img)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)

        detection = cv2.addWeighted(img, 0.8, line_img, 1, 0)

    return detection