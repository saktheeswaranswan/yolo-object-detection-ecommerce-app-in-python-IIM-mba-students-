import cv2
import numpy as np
import requests

# Load the YOLOv3-tiny model
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

# Load the COCO object classes
classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "orange", "sandwich", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hairbrush", "toothbrush"]

# Initialize the video stream
cap = cv2.VideoCapture(0)

# Create a CSV file to store the results
csv_file = open("results.csv", "w")

# Write the header row to the CSV file
csv_file.write("object,url,image\n")

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to a NumPy array
    frame_np = np.array(frame)

    # Detect objects in the frame
    detections = net.detectObjects(frame_np, confThreshold=0.5)

    # Iterate over the detected objects
    for detection in detections:
        # Get the object name
        object_name = classes[detection[0].index]

        # Get the object confidence
        confidence = detection[2]

        # If the confidence is greater than the threshold,
        # then search for the object on Amazon and Flipkart
        if confidence > 0.5:
            # Search for the object on Amazon
            amazon_url = "https://www.amazon.com/s?k=" + object_name

            # Search for the object on Flipkart
            flipkart_url = "https://www.flipkart.com/search?q=" + object_name

            # Crop the image to the detected object
            crop_img = frame_np[detection[1][1]:detection[1][1] + detection[2][1], detection[1][0]:detection[1][0] + detection[2][0]]

            # Write the object name, URL, and image to the CSV file
            csv_file.write(object_name + "," + amazon_url + "," + flipkart_url + "\n")

            # Display the image
            cv2.imshow("Cropped Image", crop_img)

    # Wait for a key press
    key = cv2.waitKey(1)

    # If the key pressed is ESC, then stop the loop
    if key == 27:
        break

# Close the CSV file
csv_file.close()

# Release the video capture object
cap.release()

# Close the window
cv2.destroyAllWindows()
