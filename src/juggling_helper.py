from time import process_time, strftime

import cv2
import numpy as np

from config_manager import ConfigManager
from centroid_tracker import CentroidTracker
from hud_controller import HUDController

print("Loading... This could take a minute.")

# Retrieve user configurable settings
conf = ConfigManager()
settings = conf.get_settings()

# Constants
RESIZE_SCALAR = settings.scale
FRAME_WIDTH = settings.frame_width
FRAME_HEIGHT = settings.frame_height
RESIZED_WIDTH = int(FRAME_WIDTH * RESIZE_SCALAR)
RESIZED_HEIGHT = int(FRAME_HEIGHT * RESIZE_SCALAR)
TRACKER_REACQUISITION_RANGE = int(settings.trackrange * RESIZE_SCALAR)
TRACKER_REACQUISITION_TIME = settings.tracktime
DEBUG_MODE = settings.debug

# Start the stopwatch / counter  
t1_start = process_time() 

# Initialize the heads up display controller which draws the UI on top of the frame
hud = HUDController(
    success_area_y=settings.success_area_y,
    success_area_length=settings.success_area_length,
    frame_width=RESIZED_WIDTH,
    reacquisition_range=TRACKER_REACQUISITION_RANGE
)

# Initialize the centroid tracker which keeps track of the same balls between frames
ct = CentroidTracker(TRACKER_REACQUISITION_RANGE, TRACKER_REACQUISITION_TIME)

# Initialize the height checker and desired starting height boundary
starting_y = FRAME_HEIGHT / 4 * RESIZE_SCALAR
starting_height = FRAME_HEIGHT / 10 * RESIZE_SCALAR

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Enable GPU processing
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Name custom object
classes = ["juggling ball"]

# Set up image categories (just 1 in our case)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Capture live camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    # Capture first webcam

# Record a video of the capture
ret, frame = cap.read() # If there is a video feed, ret is true
frame = cv2.resize(frame, None, fx=RESIZE_SCALAR, fy=RESIZE_SCALAR)
height, width, channels = frame.shape # cv2 resizes to closest compatible resolution
fourcc = cv2.VideoWriter_fourcc(*'XVID')    # Video encoder
pathname = "../captures/capture_" + strftime("%Y%m%d-%H%M%S") + ".avi"
capture_out = cv2.VideoWriter(pathname, fourcc, 24, (width, height)) # Encoded video properties

while True:
    frame_start_time = process_time()
    ret, frame = cap.read() # If there is a video feed, ret is true

    # Resize Camera
    frame = cv2.resize(frame, None, fx=RESIZE_SCALAR, fy=RESIZE_SCALAR)
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (height, width), (0, 0, 0), True, crop=False)

    # Set input/output layers
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    
    # Detect objects
    for out in outs:
        for detection in out:
            scores = detection[5:]  # Element at index 5 contains confidence
            class_id = np.argmax(scores) # Pick most confident label
            confidence = scores[class_id]

            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates (corner coordinates)
                startX = int(center_x - w / 2)
                startY = int(center_y - h / 2)
                endX = int(center_x + w / 2)
                endY = int(center_y + h / 2)

                boxes.append([startX, startY, endX, endY])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Select objects with high probability
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if DEBUG_MODE:
        # Draw bounding boxes
        for i in range(len(boxes)):
            if i in indexes:
                startX, startY, endX, endY = boxes[i]
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(boxes)

    if DEBUG_MODE:
        # Loop over the tracked objects and draw their centroids
        for (objectID, ball) in objects.items():
            # Draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (ball.centroid[0] - 10, ball.centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Detect any user input
    pressed_key = cv2.waitKey(1)

    # Raise height checker's height boundary
    if pressed_key & 0xFF == ord('w'):
        settings.success_area_y -= 2
        hud.change_boundary_y_pos(-2)

    # Lower height checker's height boundary
    if pressed_key & 0xFF == ord('s'):
        settings.success_area_y += 2
        hud.change_boundary_y_pos(2)

    # Increase height checker's boundary length
    if pressed_key & 0xFF == ord('d'):
        settings.success_area_length += 2
        hud.change_boundary_length(2)

    # Decrease height checker's boundary length
    if pressed_key & 0xFF == ord('a'):
        settings.success_area_length -= 2
        hud.change_boundary_length(-2)

    # Update UI
    hud.draw_boundary(frame)
    hud.draw_ball_points(frame, objects)
    hud.draw_success_counters(frame)

    # Show frame with boxes drawn
    capture_out.write(frame)
    cv2.imshow('frame', frame)

    # Quit if q is pressed
    if pressed_key & 0xFF == ord('q'):
        break

    if DEBUG_MODE:
        # Calculate FPS and print it out
        frame_time = process_time() - frame_start_time
        frames_per_second = int(1 / frame_time) if frame_time else 1 # div by 0 check
        print("FPS = ", frames_per_second)

# Clean up
cap.release()
capture_out.release()
cv2.destroyAllWindows()

# Write user settings to a file
conf.set_settings(settings)

# Append use statistics to a file in a Success,Failure format
with open('statistics.txt', 'a') as stat_file:
    csv_stats = str(hud.get_successes()) + "," + str(hud.get_failures()) + "\n"
    stat_file.write(csv_stats)
