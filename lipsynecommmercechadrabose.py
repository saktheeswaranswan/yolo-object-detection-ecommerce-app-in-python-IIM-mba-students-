import cv2
import dlib
import face_recognition
import csv
import speech_recognition as sr

# Load pre-trained facial landmarks detector from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize variables
people = {}  # Dictionary to store the detected people and their spoken words

# Function to extract lips region from face using facial landmarks
def extract_lips(face_landmarks):
    lips = face_landmarks[48:68]  # Indices for the lips region in the facial landmarks
    return lips

# Function to recognize and track people based on face encodings
def recognize_people(frame, face_locations, face_encodings):
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(list(people.values()), face_encoding)

        if True in matches:
            matched_person = list(people.keys())[matches.index(True)]
        else:
            matched_person = "Person" + str(len(people) + 1)
            people[matched_person] = face_encoding

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, matched_person, (left + 5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Function to save spoken words to CSV file
def save_to_csv(person, words):
    with open('spoken_words.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([person, words])

# Open the webcam for live streaming
video_capture = cv2.VideoCapture(0)

# Initialize speech recognition
r = sr.Recognizer()

while True:
    # Read a frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to gray scale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # Iterate over detected faces
    for face in faces:
        landmarks = predictor(gray, face)

        # Extract lips region from facial landmarks
        lips = extract_lips(landmarks.parts())

        # Draw lips region on the frame
        cv2.polylines(frame, [lips], True, (0, 0, 255), 2)

        # Perform lip-reading
        with sr.Microphone() as source:
            print("Please speak:")
            audio = r.record(source, duration=2)  # Record audio for 2 seconds

        try:
            spoken_words = r.recognize_google(audio)
            print("You said:", spoken_words)
        except sr.UnknownValueError:
            spoken_words = ""
            print("Unable to recognize speech")

        # Recognize and track people based on face encodings
        face_encodings = face_recognition.face_encodings(frame, [face])
        face_locations = face_recognition.face_locations(frame, model="hog")
        recognize_people(frame, face_locations, face_encodings)

        # Save spoken words to CSV file
        if spoken_words:
            matched_person = list(people.keys())[-1]  # Get the most recently detected person
            save_to_csv(matched_person, spoken_words)

    # Display theresulting frame
    cv2.imshow('Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()

