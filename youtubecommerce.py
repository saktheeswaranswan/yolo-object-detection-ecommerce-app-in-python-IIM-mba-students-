#!pip install opencv-python-headless youtube_dl fpdf pandas
#!pip install --upgrade youtube_dl

import cv2
import numpy as np
import time
import os
import webbrowser
from fpdf import FPDF
import pandas as pd
import youtube_dl

# Paths to YOLO files
weights_path = 'face-yolov3-tiny_41000.weights'
config_path = 'face-yolov3-tiny.cfg'
labels_path = 'coco.names'

# Load the YOLO network
net = cv2.dnn.readNet(weights_path, config_path)

# Get the class labels
classes = []
with open(labels_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the video stream
youtube_url = 'https://www.youtube.com/watch?v=pbu4ecHhBAc'  # Replace with the actual YouTube video URL
video_id = youtube_url.split('=')[1]  # Extract the video ID from the URL
stream_url = 'https://www.youtube.com/get_video_info?&video_id=' + video_id
ydl_opts = {'quiet': True}

try:
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(stream_url, download=False)
        formats = info.get('formats', None)
        for f in formats:
            if f.get('vcodec') == 'none':
                stream_url = f['url']
                break
except youtube_dl.utils.DownloadError as e:
    print('Error: Unable to fetch video information. Make sure the video is available.')
    print(e)
    exit(1)

cap = cv2.VideoCapture(stream_url)

# Create a directory to save the cropped images
os.makedirs('detected_images', exist_ok=True)

# Create a PDF object
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Create a DataFrame to store the links
df = pd.DataFrame(columns=['Object', 'Shop', 'Link', 'Place', 'Time'])

start_time = time.time()
update_interval = 300  # 5 minutes

while True:
    # Read frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Process the detection results
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Perform non-maximum suppression to eliminate overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Crop the detected objects and generate links
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box

        # Crop the detected object
        crop_img = frame[y:y+h, x:x+w]

        # Generate a unique filename based on date, time, and place
        date_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = f'detected_images/{date_time}_place.jpg'
        cv2.imwrite(filename, crop_img)

        # Search the object on shopping platforms
        object_name = classes[class_ids[i]]
        search_query = f'Search {object_name} on'
        links = {
            'Amazon': f'https://www.amazon.com/s?k={object_name}',
            'Myntra': f'https://www.myntra.com/search/{object_name}',
            'Mesho': f'https://www.mesho.com/search?q={object_name}',
            'Flipkart': f'https://www.flipkart.com/search?q={object_name}'
        }

        # Store the links in the DataFrame
        for shop, link in links.items():
            df = df.append({
                'Object': object_name,
                'Shop': shop,
                'Link': link,
                'Place': 'PLACE_NAME',  # Replace with the actual place name
                'Time': date_time
            }, ignore_index=True)

        # Add the cropped image and links to the PDF
        pdf.add_page()
        pdf.cell(60, 60, txt='', border=0, ln=1, align='C')
        pdf.image(filename, x=pdf.get_x() + 5, y=pdf.get_y(), w=50)
