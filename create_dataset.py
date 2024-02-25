# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:29:47 2024
https://www.youtube.com/watch?v=MJCSjXepaAM

https://github.com/computervisioneng/sign-language-detector-python/blob/master/create_dataset.py

@author: Hoysala
"""

import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

DATA_DIR = './data'

# objects to detect landmarks from mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#hands detector
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = [] #to stire all the data produced
labels = [] # for all the labels or categories

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): #[]:`]
        #print(img_path)
        
        data_aux = []

        x_ = []
        y_ = []
        
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        #convert to RGB to work with mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #plt.fugure()
        #plt.imshow(img_rgb)
        #plt.show()

        #take all the landmarks into the image
        #all the info is in hte landmarks
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks: #if there are any landmarks or a hand
            
            #then iterate thru landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    
                    #get the position of each landmark in the image
                    # store only the x & y cordinates in an array
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_) #the directory label is the category for now, that can be labelled into human readable name later

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

