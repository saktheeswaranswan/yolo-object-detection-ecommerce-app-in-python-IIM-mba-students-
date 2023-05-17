import cv2
import numpy as np

# Load the YOLOv3-tiny model
net = cv2.dnn.readNet("face-yolov3-tiny_41000.weights", "face-yolov3-tiny.cfg")

# Load the COCO object classes
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Initialize the video stream
cap = cv2.VideoCapture(0)

# Create a CSV file to store the results
csv_file = open("results.csv", "w")
csv_file.write("object,url\n")

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to a NumPy array
    frame_np = np.array(frame)

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame_np, 1 / 255, (416, 416), swapRB=True, crop=False)

    # Set the input blob for the neural network
    net.setInput(blob)

    # Forward pass through the network
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    layer_outputs = net.forward(output_layers)

    # Gather predictions
    detections = []
    class_ids = []
    confidences = []
    boxes = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame_np.shape[1])
                center_y = int(detection[1] * frame_np.shape[0])
                width = int(detection[2] * frame_np.shape[1])
                height = int(detection[3] * frame_np.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    for i in indices:
        i = i
        x, y, width, height = boxes[i]
        label = classes[class_ids[i]]

        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the object name and URL to the CSV file
        object_name = label
        amazon_url = "https://www.amazon.com/s?k=" + label
        flipkart_url = "https://www.flipkart.com/search?q=" + label
        csv_file.write(object_name + "," + amazon_url + "\n")
        csv_file.write(object_name + "," + flipkart_url + "\n")

    # Display the frame
    cv2.imshow("Frame", frame)

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


