# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:05:10 2024

Hand gesture reconition - https://www.youtube.com/watch?v=MJCSjXepaAM
https://github.com/computervisioneng/sign-language-detector-python/blob/master/collect_imgs.py


@author: Hoysala
"""

import os
import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of gestures to detect
number_of_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        #wait till Q is pressed to capture the next gesture
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('image capture frame', frame)
qq

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25) #wait time between image grab
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        print(counter)
        counter += 1

cap.release()
cv2.destroyAllWindows()