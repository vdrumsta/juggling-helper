import cv2
import numpy as np
from time import process_time

# Start the stopwatch / counter  
t1_start = process_time() 

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Name custom object
classes = ["juggling ball"]

# Set up image categories (just 1 in our case)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Capture live camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    # Capture first webcam
fourcc = cv2.VideoWriter_fourcc(*'XVID')    # Video encoder
capture_out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))    # Encoded video properties

print("Time to load", process_time() - t1_start)
while True:
    frame_start_time = process_time()
    ret, frame = cap.read() # If there is a video feed, ret is true

    # Resize Camera
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set input/output layers
    net.setInput(blob)
    outs = net.forward(output_layers)
    #print("t3", process_time() - frame_start_time)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []

    
    # Detect objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                #print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Select objects with high probability
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes)

    # Draw boxes and label them
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 2)
    
    

    # Show frame with boxes drawn
    capture_out.write(frame)
    cv2.imshow('frame', frame)

    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Calculate FPS and print it out
    frames_per_second = process_time() - frame_start_time if frame_start_time else 0
    print("FPS = ", frames_per_second)

# Clean up
cap.release()
capture_out.release()
cv2.destroyAllWindows()