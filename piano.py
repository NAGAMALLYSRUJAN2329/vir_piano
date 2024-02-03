import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

from keyboard import make_keyboard,convert_coordinates_to_lines,check_key
from piano_sound import play_piano_sound

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
connections=list(mp_hands.HAND_CONNECTIONS)

def modify(x,y,z,shape=(800,600)):
    x=(np.array(x) * shape[0]).astype(int)
    y=(np.array(y) * shape[1]).astype(int)
    z=(np.array(z) * shape[0]).astype(int)
    x=[abs(i) for i in x]
    y=[abs(i) for i in y]
    # z=[abs(i) for i in z]
    return x,y,z

def point_distance(pts,x,y):
    for i,point in enumerate(pts):
        distance = cv2.norm(np.array((x,y)), point[0], cv2.NORM_L2)
        if(distance<20):
            return i,True
    return 4,False

def check_configure_piano(configure_piano,x1,y1,x2,y2,x,y,l,w):
    distance = cv2.norm(np.array((x1,y1)), np.array((x2,y2)), cv2.NORM_L2)
    t=False
    if x1>x and x1<x+l and y1>y and y1<y+w:
        t=True
    if distance<20 and t:
        time.sleep(0.1)
        return not configure_piano
    return configure_piano

def piano_config(frame,pts,configure_piano,x0,y0,x1,y1,button_x,button_y,button_l,button_w):
    finger_distance = cv2.norm(np.array((x0,y0)), np.array((x1,y1)), cv2.NORM_L2)
    configure_piano=check_configure_piano(configure_piano,x0,y0,x1,y1,button_x,button_y,button_l,button_w)
    if configure_piano:
        frame=cv2.rectangle(frame,(button_x,button_y),(button_x+button_l,button_y+button_w),(0,255,0),-1)
        cv2.putText(frame, "ON", (button_x,button_y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        idx,bool=point_distance(pts,x[8],y[8])
        if finger_distance<20 and bool:
            pts[idx][0]=[x[8],y[8]]
    else:
        frame=cv2.rectangle(frame,(button_x,button_y),(button_x+button_l,button_y+button_w),(0,0,255),-1)
        cv2.putText(frame, "OFF", (button_x,button_y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    # print(finger_distance)
    return frame,pts,configure_piano

def get_piano_notes(n):
    white_piano_notes=[]
    black_piano_notes=[]
    white_piano_notes=['A0','B0','A1','B1','C1','D1','E1','F1','G1','A2','B2','C2','D2','E2','F2','G2']
    black_piano_notes=['Bb0','Bb1','Db1','Eb1','Gb1','Ab1','Bb2','Db2','Eb2','Gb2','Ab2']
    return white_piano_notes,black_piano_notes


shape=(800, 600)
n=2
pts=np.array([[[100,350]],[[700,350]],[[700,550]],[[100,550]]])
button_x,button_y,button_l,button_w=[600,75,50,50]

last_pressed_key=None
white_piano_notes,black_piano_notes=get_piano_notes(2)
previousTime=0
configure_piano=False
connections=list(mp_hands.HAND_CONNECTIONS)
cap = cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    frame = cv2.resize(frame, shape)
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame)

    white,black=make_keyboard(frame,pts,n)
    white_lines,black_lines=convert_coordinates_to_lines(white,black)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            x,y,z=[],[],[]
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmark, connections)
            for landmark in hand_landmark.landmark:
                x.append(landmark.x),y.append(landmark.y),z.append(landmark.z)
            x,y,z=modify(x,y,z)

            frame,pts,configure_piano=piano_config(frame,pts,configure_piano,x[4],y[4],x[8],y[8],button_x,button_y,button_l,button_w)
            frame,pts,configure_piano=piano_config(frame,pts,configure_piano,x[4],y[4],x[8],y[8],button_x,button_y,button_l,button_w)
            if not configure_piano:
                pressed_key=check_key(x[8],y[8],white_lines,black_lines)
                if pressed_key and pressed_key!=last_pressed_key:
                    if pressed_key[0]=="Black":
                        play_piano_sound([black_piano_notes[pressed_key[1]]])
                    else:
                        play_piano_sound([white_piano_notes[pressed_key[1]]])
                last_pressed_key=pressed_key

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(frame, str(int(fps)) + "FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    
    cv2.namedWindow("Hand Detection", cv2.WINDOW_NORMAL)
    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
