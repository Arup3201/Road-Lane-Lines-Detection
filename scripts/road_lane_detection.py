import cv2
import numpy as np

DECREASE_BRIGHTNESS = 80 # AMOUNT OF BRIGHTNESS TO DECREASE
CLIPLIMIT = 5.0  # Clip limit for CLEHA
LOW_CANNY = 100 # Lower bound value of Canny Edge Detector
HIGH_CANNY = 200 # Higher bound value of Canny Edge Detector
RHO = 6  # Distance resolution of the accumulator in pixels.
THETA = np.pi/180  # Angle resolution of the accumulator in radians.
THRESHOLD = 20  # Only lines that are greater than threshold will be returned.
MINLINELENGTH = 20  # Line segments shorter than that are rejected.
MAXLINEGAP = 300  # Maximum allowed gap between points on the same line to link them
DETECTION_LINE_COLOR = [0, 255, 0]
DETECTION_LINE_THICKNESS = 7

def detect_lanes_v1(img):
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

        # Weighted addition by giving more weight to the line
        detection = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    return detection

def detect_lanes_v2(img):
    '''version 2 of lane detection.
    steps:
        1. Decrease brightness and increase contrast
        2. Extract white and yellow parts from the image
        3. Convert to gray scale and apply gaussian blur
        4. ROI selection in the image and cutting that part
        5. Apply Canny Edge detector
        6. Find out the lines using Hough Transformation
    '''

    detection = np.copy(img)
    
    transformed = decrease_brightness(detection, DECREASE_BRIGHTNESS)
    transformed = increase_contrast(transformed, CLIPLIMIT)
    transformed = extract_white_yellow(transformed)
    transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    transformed = cv2.GaussianBlur(transformed, (5, 5), 0)
    transformed = ROI_selection(transformed)
    transformed = cv2.Canny(transformed, LOW_CANNY, HIGH_CANNY)
    
    lines = cv2.HoughLinesP(transformed, rho=RHO, theta=THETA, threshold=THRESHOLD, minLineLength=MINLINELENGTH, maxLineGap=MAXLINEGAP)

    if lines is not None:
        line_img = np.zeros_like(img)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), color=DETECTION_LINE_COLOR, thickness=DETECTION_LINE_THICKNESS)
        detection = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    return detection


def decrease_brightness(img, value):
    '''method is used to decrease the brightness of the image. Image should be RGB channeled and brightness will be decreased by `value`.
        img: numpy array, RGB format image
        value: int, amount of brightness to decrease
    '''
    transformed = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(transformed)

    limit = value
    v[v < limit] = 0
    v[v >= limit] -= value

    transformed = cv2.merge((h, s, v))
    transformed = cv2.cvtColor(transformed, cv2.COLOR_HSV2BGR)
    return transformed

def increase_contrast(img, climit):
    '''method to increase the contrast of the image
    img: numpy array, RGB format image
    climit: int, clip limit parameter value of cv2.createCLAHE method
    '''
    transformed = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(transformed)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=climit, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    transformed = cv2.merge((cl, a, b))

    transformed = cv2.cvtColor(transformed, cv2.COLOR_LAB2BGR)
    
    return transformed

def extract_white_yellow(img):
    '''method to extract the white and yellow parts from the image.
    img: numpy array, RGB format image
    '''
    transformed = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Define lower and upper bounds for white color in HSV
    lower_white = np.uint8([0, 200, 0])
    upper_white = np.uint8([255, 255, 255])
    
    # Define lower and upper bounds for yellow color in HSV
    lower_yellow = np.uint8([10, 0, 100])
    upper_yellow = np.uint8([40, 255, 255])
    
    # Threshold the HSV image to extract white and yellow regions
    mask_white = cv2.inRange(transformed, lower_white, upper_white)
    mask_yellow = cv2.inRange(transformed, lower_yellow, upper_yellow)
    
    # Combine the white and yellow masks
    mask_combined = cv2.bitwise_or(mask_white, mask_yellow)
    
    # Apply the combined mask to the original image
    transformed = cv2.bitwise_and(transformed, transformed, mask=mask_combined)

    # Convert back to RGB
    transformed = cv2.cvtColor(transformed, cv2.COLOR_HLS2RGB)
    
    return transformed

def ROI_selection(img):
    # Get height & width
    height, width = img.shape
    
    # Image height & width
    mask = np.zeros_like(img)
    
    # ROI Selection
    bottom_left = (0, height)
    top_left = (0, height*0.4)
    bottom_right = (width, height)
    top_right = (width, height*0.4)
    
    roi = np.array([
        [bottom_left, top_left, bottom_right, top_right]
    ], dtype=np.int32)

    # ROI cut
    roi_cut = cv2.fillPoly(mask, roi, 255)
    mask = cv2.bitwise_and(img, roi_cut)

    return mask
