# tap detection using the y axis coordination.
#  1. detect the tips
#  2. ask for the tap height for each finger 
#  3. then after that detect the tap if it goes delow that height in the key board. 

import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import requests
import imutils
import threading
import pygame.mixer
from pygame import *
import os
import sys
import multiprocessing
from collections import deque


mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
connections=list(mp_hands.HAND_CONNECTIONS)

def make_connections(x, y,img,color = (0, 0, 255),thickness=2):
    for connection in connections:
        start_point = (x[connection[0]], y[connection[0]])
        end_point = (x[connection[1]], y[connection[1]])
        img = cv2.line(img, start_point, end_point, color, thickness)
    return img

def modify(x,y,z,shape=(800,600)):
    x=(np.array(x) * shape[0]).astype(int)
    y=(np.array(y) * shape[1]).astype(int)
    z=(np.array(z) * shape[0]).astype(int)
    x=[abs(i) for i in x]
    y=[abs(i) for i in y]
    # z=[abs(i) for i in z]
    return x,y,z

def modify2(x,y,shape=(800,600)):
    x=(np.array(x) * shape[0]).astype(int)
    y=(np.array(y) * shape[1]).astype(int)
    # x=[abs(i) for i in x]
    # y=[abs(i) for i in y]
    return x,y

tip_landmarks = [4,8,12,16,20] #index of tip position of all fingers
dist_threshold_param= {'thumb': 8.6, 'index': 6, 'middle': 6, 'ring': 6, 'little': 5} #customized dist threshold values for calibration of finger_detect_and_compute module
left_detect=np.zeros(5);right_detect=np.zeros(5) #arrays representing detected finger presses for each hand
left_coordinates=np.zeros((5,2));right_coordinates=np.zeros((5,2)) #arrays representing pixel coordinates of each detected finger press (tip landmark)
key_index_array=[]#stores indexes and colors for all detected key presses
play_music_status=1
visualizer_status=1


def check_threshold2(p1, p2, finger):
        if ((p1- p2)>40):
            print("key_pressed")
            print(finger)
        return (p1- p2)>10


def check_threshold(p1,p2,p3,finger):

    """ Function: Checks whether a key press is detected for a finger based on a mathematical condition
            Arguments: p1,p2,p3: positions of landmarks of a finger
                       finger: string name of the finger pressed (not required)
            returns: boolean value of whether key press is detected or not """
    # print(type(np.linalg.norm(p1)))
    # if(445< np.linalg.norm(p1)):
        #  print(finger)
    return(445< np.linalg.norm(p1))
    # global dist_threshold_param
    # p1=p1/10
    # p2=p2/10
    # p3=p3/10

    # dist = np.linalg.norm(p1 - p2) +  np.linalg.norm(p3 - p2) + np.linalg.norm(p1 - p3) #Calculating sum of absolute distances b/w three landmark points of a finger. This is a noobie algo. Can be improved!
    # return (dist<dist_threshold_param[finger]) #return True if this is smaller than a prespecified threshold during calibration
    
def finger_detect_and_compute(list, list_before):

    """ Function: Computes whether a key is actually pressed using fingers of a hand in an image frame. Also computes the coordinates of tip_landmarks corresponding to the pressed fingers
            Arguments: list: a list containing all position landmarks of a hand
            returns: detected_array: boolean array representing corresponding key presses
                     coordinates: pixel coordinates of the tip landmakrs of the pressed keys """
    
    # detect_array=np.array([(int)(check_threshold(list[2][1:3],list[3][1:3],list[4][1:3],'thumb')),(int)(check_threshold(list[6][1:3],list[7][1:3],list[8][1:3],'index')),(int)(check_threshold(list[10][1:3],list[11][1:3],list[12][1:3],'middle')),(int)(check_threshold(list[14][1:3],list[15][1:3],list[16][1:3],'ring')),(int)(check_threshold(list[18][1:3],list[19][1:3],list[20][1:3],'little'))])
    detect_array = np.array([(int)(check_threshold2(list[2][2],list_before[2][2], "thumb" )),(int)(check_threshold2(list[6][2],list_before[6][2],'index')),(int)(check_threshold2(list[10][2],list_before[10][2],'middle')),(int)(check_threshold2(list[14][2],list_before[14][2],'ring')),(int)(check_threshold2(list[18][2],list_before[18][2],'little'))])
    # print(list[2][2])
    coordinates=np.zeros((5,2))
    for i in range(5):
        if(detect_array[i]!=0):
            coordinates[i]=list[tip_landmarks][i,1:3]
    
    return detect_array,coordinates



pTime = 0; cTime = 0; right_hand=1; left_hand=0
lmList=[]; rmList=[]
    # global right_detect,right_coordinates,left_detect,left_coordinates,play_music_status,key_index_array
music_list_curr=[]
music_list_prev=[]

frame_lists_buffer = deque(maxlen=24)
connections=list(mp_hands.HAND_CONNECTIONS)
shape=(640, 480)
cap = cv2.VideoCapture(0)
desired_frame_rate = 12
cap.set(cv2.CAP_PROP_FPS, desired_frame_rate)
while True:
    ret,frame=cap.read()
    frame = cv2.resize(frame, shape)
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, shape)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    img1= np.zeros((shape[0],shape[1],3), dtype=np.uint8)
    img2= np.zeros((shape[0],shape[1],3), dtype=np.uint8)
    img3= np.zeros((shape[0],shape[1],3), dtype=np.uint8)

    if results.multi_hand_landmarks:
        # print(results.multi_hand_landmarks)
        for hand_landmark in results.multi_hand_landmarks:
            x,y,z=[],[],[]
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmark, connections)
            for id, lm in enumerate(hand_landmark.landmark):
                x.append(lm.x)
                y.append(lm.y)
                z.append(lm.z)
            x,y,z=modify(x,y,z)
            # print(x)
            # print(y)
            # print(z)
            # print("no")
            img1=make_connections(x,y,img1)
            img2=make_connections(x,x[0]+z,img2)
            img3=make_connections(y[0]+z,y,img3)
    

    List =[]
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            
 
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, c = img1.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                List.append([id, cx, cy])
                
    

    List=np.array(List)
    frame_lists_buffer.append(List)

    
    dim=np.shape(np.array(results.multi_hand_landmarks))
    if(dim):
            hands_count = dim[0]
    else:
            hands_count= 0
    print(List)
    if len(List) != 0:
            # if len(frame_lists_buffer) == 40:
        # Pass the current frame's list and the list from 40 frames ago to the processing function
                # process_frame(List, frame_lists_buffer[0])
                left_detect,left_coordinates = finger_detect_and_compute(List, frame_lists_buffer[0]) 
                # print("Left Hand Detection Array=", left_detect)
                # print("left coordinates are", left_coordinates)
                for i in range(5):
                    if(left_detect[i]!=0):
                        x,y=left_coordinates[i]
                        # x,y = modify2(x,y)
                        img=cv2.circle(img1, (int(x),int(y)), 10, (10,50,50), 5)
                        cv2.imshow('Detection', img)
        
    

    cv2.imshow('Hand Detection', frame)
    cv2.imshow('Front View', img1)
    # cv2.imshow('Top View', img2)
    # cv2.imshow('Side View', img3)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()