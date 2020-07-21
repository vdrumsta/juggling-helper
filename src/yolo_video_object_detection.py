import cv2
import numpy as np
import height_checker
from time import process_time
from centroid_tracker import CentroidTracker

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Start the stopwatch / counter  
t1_start = process_time() 

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Name custom object
classes = ["juggling ball"]

# Set up image categories (just 1 in our case)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Capture live camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    # Capture first webcam
fourcc = cv2.VideoWriter_fourcc(*'XVID')    # Video encoder
capture_out = cv2.VideoWriter('output.avi', fourcc, 20.0, 
    (FRAME_WIDTH, FRAME_HEIGHT))    # Encoded video properties

print("Time to load", process_time() - t1_start)
while True:
    frame_start_time = process_time()
    ret, frame = cap.read() # If there is a video feed, ret is true

    # Resize Camera
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
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
                #print(class_id)
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

    # Draw bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            startX, startY, endX, endY = boxes[i]
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            
            #font = cv2.FONT_HERSHEY_PLAIN
            #label = str(classes[class_ids[i]])
            #cv2.putText(frame, label, (x, y + 30), font, 3, color, 2)
    
    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(boxes)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    

    # Show frame with boxes drawn
    capture_out.write(frame)
    cv2.imshow('frame', frame)

    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Calculate FPS and print it out
    frame_time = process_time() - frame_start_time
    frames_per_second = int(1 / frame_time) if frame_time else 1 # div by 0 check
    print("FPS = ", frames_per_second)

    

# Clean up
cap.release()
capture_out.release()
cv2.destroyAllWindows()