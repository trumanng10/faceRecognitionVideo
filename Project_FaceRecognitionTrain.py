import os
import cv2 as cv
import numpy as np

#Step 1: Obtain the Dataset Names(Folder Names) from the training set folder
DIR = r'C:\Users\TrumanNg\PycharmProjects\pythonProject2-openCV\img\train'

people = []
for i in os.listdir(DIR):
    people.append(i)

print(f"The Labels of the DataSet: {people}")

#Step 2:
haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        #Open up all the sample data(photos) in each folder
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)

            #Convert to gray photo
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            #print(f'Number of faces found = {len(faces_rect)}')
            #print(faces_rect)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                # Convert the labels to numeric data to reduce the strain
                # For example: nicole: 0, tom: 1
                labels.append(label)


create_train()
print("----------------- Training Completed -----------------")
features = np.array(features, dtype='object')
labels = np.array(labels)

print(f'Features:{features}')
print(f'Labels:{labels}')
print(f'Length of the features: {len(features)} ')
print(f'Length of the labels: {len(labels)} ')

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the feature
face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')

print("----------------- Face Recognizer Training Completed -----------------")
np.save('features.npy', features)
np.save('labels.npy', labels)
