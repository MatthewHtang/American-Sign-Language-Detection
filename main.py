#ASL Detection
#Import Libraries
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

from TTS import speak

#Capture Video
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("/Users/matthewhtang/Desktop/American-Sign-Language-Detection/model/keras_model.h5", "/Users/matthewhtang/Desktop/American-Sign-Language-Detection/model/labels.txt")

#Set Parameters
offset = 20
imgSize = 300
counter = 0

labels = ["A","B","C","D","E","F","G","H",
          "I","K","L","M","N","O","P","Q",
          "R","S","T","U","V","W","X","Y", 
          "Period"]

#Initialize variables
previous_label = ""
start_time = time.time()
outPut = ""
letter_added = False
double_letter_added = False
space_added = False
elapsed = 0

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+60-50), (255,255,255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x,y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2)
        cv2.rectangle(imgOutput, (x-offset,y-offset), (x + w + offset, y + h + offset), (0,255,0), 4)

        #cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)


        #Check if the same label continues
        current_label = labels[index]

        if current_label == previous_label:
            elapsed = time.time() - start_time

            # Add first character after 0.8 seconds
            if elapsed > 0.8 and not letter_added:
                outPut += current_label
                letter_added = True
                space_added = False

            # Add second character after 1.5 seconds (for double letters like TT)
            elif elapsed > 1.5 and not double_letter_added:
                outPut += current_label
                double_letter_added = True      

        # Reset timer if label changes
        else:
            previous_label = current_label
            start_time = time.time()
            elapsed = 0
            letter_added = False
            double_letter_added = False
            space_added = False

    # Add space when hand disappears
    else:
        if not hands and not space_added and outPut != "":
            outPut += " "
            space_added = True
            letter_added = False
            double_letter_added = False

    # display sentence every frame
    cv2.putText(imgOutput, outPut, (50, 450),
                cv2.FONT_HERSHEY_COMPLEX, 2,
                (0,0,255), 3)

    cv2.imshow('Image', imgOutput)

    key = cv2.waitKey(1)

    #When label is Period it will output "."
    if labels == "Period":
        outPut += "." 
        
        #converting the final output from text to speech
        speak(outPut)
    
    if key == ord('q'):
        break

    if key == ord('c'):
        outPut = ""    

cap.release()
cv2.destroyAllWindows()

#Note
"""
Now i need to collect another data (img) and 
assign it with'.'

**data for numbers need to be collected as well
"""