import face_recognition
import cv2
import sys, time
import os

# Get a reference to the  webcam
print("[INFO] sampling frames from webcam...")
video_capture = cv2.VideoCapture(0)

# Creating the dataset and learning how to recognise it

shesh_image = face_recognition.load_image_file("known/shesh_nath.jpeg")
shesh_face_encoding = face_recognition.face_encodings(shesh_image)[0]

geeta_image = face_recognition.load_image_file("known/geeta_rai.jpeg")
geeta_face_encoding = face_recognition.face_encodings(geeta_image)[0]

shambhavi_image = face_recognition.load_image_file("known/shambhavi_rai.jpeg")
shambhavi_face_encoding = face_recognition.face_encodings(shambhavi_image)[0]

shivang_image = face_recognition.load_image_file("known/shivang_rai.jpeg")
shivang_face_encoding = face_recognition.face_encodings(shivang_image)[0]

nidhi_image = face_recognition.load_image_file("known/nidhi_rai.jpeg")
nidhi_face_encoding = face_recognition.face_encodings(nidhi_image)[0]

pragati_image = face_recognition.load_image_file("known/pragati_rai.jpeg")
pragati_face_encoding = face_recognition.face_encodings(pragati_image)[0]

advika_image = face_recognition.load_image_file("known/advika_rai.jpeg")
advika_face_encoding = face_recognition.face_encodings(advika_image)[0]

pallavi_image = face_recognition.load_image_file("known/pallavi_rai.jpeg")
pallavi_face_encoding = face_recognition.face_encodings(pallavi_image)[0]

tanisha_image = face_recognition.load_image_file("known/tanisha_rai.jpeg")
tanisha_face_encoding = face_recognition.face_encodings(tanisha_image)[0]

anshita_image = face_recognition.load_image_file("known/anshita_rai.jpeg")
anshita_face_encoding = face_recognition.face_encodings(anshita_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    shesh_face_encoding,
    geeta_face_encoding,
    shambhavi_face_encoding,
    shivang_face_encoding,
    nidhi_face_encoding,
    pragati_face_encoding,
    advika_face_encoding,
    pallavi_face_encoding,
    tanisha_face_encoding,
    anshita_face_encoding,
    
]
known_face_names = [
    "Shesh Nath",
    "Geeta Rai",
    "Shambhavi Rai",
    "Shivang Rai",
    "Nidhi Rai",
    "Pragati Rai",
    "Advika Rai",
    "Pallavi Rai",
    "Tanisha Rai",
    "Anshita Rai",
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.55)
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
	    
            name = "Unknown"

            # If a match was found in known_face_encodings, use the one which had minimum face distance i.e. the closest match
            if True in matches:
                best_match_index = distances.argmin()
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 5)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
       
    # Display the resulting image
    
    cv2.imshow('Camera', frame)
    if 'Unknown' in face_names:
         # To prevent sending multiple emails when the face is in the frame for a long time       
	    if (time.time()-os.path.getctime('/home/pi/facerecog/test.jpg')) > 30:   
    		img = cv2.imwrite("test.jpg",frame)
    	os.system('sh mailme.sh')

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()













   

