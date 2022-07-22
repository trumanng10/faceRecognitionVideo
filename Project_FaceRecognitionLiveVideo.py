import cv2 as cv
import numpy as np
import os

faceCascade = cv.CascadeClassifier('haar_face.xml')

features = np.load('features.npy', allow_pickle = True)
labels = np.load('labels.npy')


DIR = r'C:\Users\TrumanNg\PycharmProjects\pythonProject2-openCV\img\train'

people = []
for i in os.listdir(DIR):
    people.append(i)


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

video_capture = cv.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:

        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        faces_roi = gray[y:y + h, x:x + w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with confidence level of {confidence}')

        #cv.putText(frame, str(people[label])+", Confidence: "+ str(confidence), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=1)
        cv.putText(frame, str(people[label]) + ", Confidence: " + str(confidence), (x+50, y-50),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=1)

        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)

        # Display the resulting frame

    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
video_capture.release()
cv.destroyAllWindows()
