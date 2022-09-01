import cv2
import mediapipe as mp
from flask import Flask, request, jsonify


def mediaPipe():
    IMAGE_FILES = [
        "C:/Users/mohamed.elhalwagui/Downloads/yolov5-master/yolov5-master/runs/detect/exp/crops/person/mess0-ronaldo.jpg",
        "C:/Users/mohamed.elhalwagui/Downloads/yolov5-master/yolov5-master/runs/detect/exp/crops/person/mess0-ronaldo2.jpg",
        "C:/Users/mohamed.elhalwagui/Downloads/yolov5-master/yolov5-master/runs/detect/exp/crops/person/mess0-ronaldo3.jpg"]
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # global variables
    pose_estimator = []
    pose_estimator_dim = []
    # For each object detected
    # WHICH POSE ESTIMATOR TO USE.
    with open(
            "C:/Users/mohamed.elhalwagui/Downloads/yolov5-master/yolov5-master/runs/detect/exp/labels/mess0-ronaldo.txt") as detected_objects_file:
        lines = detected_objects_file.readlines()
    annotated_image = cv2.imread(
        "C:/Users/mohamed.elhalwagui/Downloads/yolov5-master/yolov5-master/runs/detect/exp/mess0-ronaldo.jpg")
    annotated_image_height, annotated_image_width, _ = annotated_image.shape
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            yoloFormat = lines[idx][2:-1]
            yoloPointsString = yoloFormat.split(' ')
            yoloPointsFloat = []
            for point in yoloPointsString:
                yoloPointsFloat.append(float(point))
            x, y, w, h = convert(yoloPointsFloat[0], yoloPointsFloat[1], yoloPointsFloat[2], yoloPointsFloat[3])
            print(x)
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Draw pose landmarks on the image.

            # Use mp_pose.UPPER_BODY_POSE_CONNECTIONS for drawing below when
            # upper_body_only is set to True.
            for landmark in results.pose_landmarks.landmark:
                landmark.x = (landmark.x * (image_width / annotated_image_width)) + x
                landmark.y = (landmark.y * (image_height / annotated_image_height)) + y

            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("output image", annotated_image)
        cv2.waitKey(0)


# mediapipe


def convert(x, y, w, h):
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1, y1, x2, y2

# For
# static
# images:
