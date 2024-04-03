##importing Libraries

import cv2
import numpy as np
import time
import math

##loading the pre-trained model 
net = cv2.dnn.readNet("C:\\Users\\vidya Peddinti\\Downloads\\yolov3.weights", "C:\\Users\\vidya Peddinti\\Downloads\\yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

##loading the labels file for 'person' detection
with open("C:\\Users\\vidya Peddinti\\Downloads\\coco.names", "r") as f:
    classes = [line.strip() for line in f]

##giving the path to the input video
video_path = "D:\\Users\\vidya Peddinti\\Downloads\\video1.mp4"


cap = cv2.VideoCapture(video_path)
desired_width = 1000
desired_height = 700

def plot_coordinates(frame, coordinates, direction, distance_after_t, dot_color):
    for coord, dist_after_t in zip(coordinates, distance_after_t):
        x, y = coord
        if direction == "Left":
            cv2.circle(frame, (int(x), int(y)), 5, dot_color, -1)
        elif direction == "Right":
            cv2.circle(frame, (int(x), int(y)), 5, dot_color, -1)

# Optical flow initialization
ret_flow, frame1 = cap.read()
frame1 = cv2.resize(frame1, (desired_width, desired_height))

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
car_speed = 0.0
frames = 0
avg_car_speed = 0.0  

# Pedestrian detection initialization
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize a dictionary to store person identities and their bounding boxes
person_dict = {}
prev_boxes = {}

prev_time = time.time()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('C:\\Users\\vidya Peddinti\\Desktop\\FSP_SEM_2\\FSP_Project\\Final_Output\\video7\\video7_output.mp4', fourcc, 20.0, (desired_width, desired_height))

while True:
    # Optical flow processing
    ret_flow, frame2 = cap.read()
    if not ret_flow:
        print("Error: Could not read frame for optical flow.")
        break

    frame2 = cv2.resize(frame2, (desired_width, desired_height))

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 5, 3, 5, 1.2, 0)
    mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_car_speed = np.average(mag)
    avg_angle = np.average(angle)
    car_speed += avg_car_speed
    frames += 1
    prvs = next

    # Pedestrian detection processing
    blob = cv2.dnn.blobFromImage(frame2, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.2 and class_id == 0: 
                center_x, center_y = int(obj[0] * frame2.shape[1]), int(obj[1] * frame2.shape[0])
                width, height = int(obj[2] * frame2.shape[1]), int(obj[3] * frame2.shape[0])
                x, y = int(center_x - width/2), int(center_y - height/2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, width, height = boxes[i]

            # Logic to calculate distance, direction, and plot dots is similar to the HOG version
            distance = 100 / width

            
            # Calculate time to reach pedestrian using s = ut + 1/2 a t^2
            # Assuming initial velocity (u) of the pedestrian is 0
            time_to_reach_pedestrian = math.sqrt(2 * distance / 0.5)

            # Assume initial velocity (u) of the pedestrian is 0
            distance_traveled = 0.5 * (avg_car_speed / frames) * time_to_reach_pedestrian**2

            direction = "Right" if x > frame2.shape[1] / 2 else "Left"
            pixel_coordinates = (x + width, y + height) if direction == "Right" else (x, y + height)

            if distance_traveled < 1.7:
                dot_color = (0, 0, 255)
                box_color = (0, 0, 255)
            else:
                dot_color = (255, 0, 0)
                box_color = (0, 255,0)

            plot_coordinates(frame2, [pixel_coordinates], direction, [distance_traveled], dot_color)

            # Display bounding box
            cv2.rectangle(frame2, (x, y), (x + width, y + height), box_color, 2)

            # Update previous boxes
            if i in prev_boxes:
                prev_x, _, _, _ = prev_boxes[i]
                direction = "Right" if x > prev_x else "Left"
            else:
                direction = "Unknown"

            prev_boxes[i] = (x, y, width, height)

            # Display distance and coordinates information
           # print(f"Pedestrian {i + 1} - Distance: {distance:.2f} pixels, Direction: {direction}")
           # print(f"Original Coordinates: ({x}, {y}), Final Coordinates: ({x + width}, {y + height})")
           # print(f"Distance Traveled: {distance_traveled:.2f} pixels\n")


            print(f"Pedestrian {i + 1}:")
            print(f"  Top-left: ({x}, {y}) ,"f"  Bottom-right: ({x + width}, {y + height})")
            print(f"  Distance: {distance:.2f} pixels")
            print(f"  Direction: {direction}")
            print(f"  Average Angle: {math.degrees(avg_angle):.2f} degrees")
            print(f"  Pedestrian speed: {distance_traveled / time_to_reach_pedestrian:.2f} pixels/frame")
            print(f"  Car speed: {avg_car_speed:.3f} pixels/frame")
            print(f"  Pedestrian distance after time {time_to_reach_pedestrian:.2f} seconds is: {distance_traveled:.2f} pixels {direction}\n")

       

    # Display car_speed information on the frame
    car_speed_text = f'car_speed: {avg_car_speed:.3f} pixels/frame'
    avg_car_speed_text = f'Avg car_speed: {car_speed / frames:.3f} pixels/frame'
    cv2.putText(frame2, car_speed_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame2, avg_car_speed_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Combined Processing', frame2)

    # Write the frame to the output video file
    output_video.write(frame2)

    if cv2.waitKey(1) == 27:
        break

# Release the VideoWriter object and capture
output_video.release()
cap.release()
cv2.destroyAllWindows()
