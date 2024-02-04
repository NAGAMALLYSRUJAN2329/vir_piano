import numpy as np
import cv2
import time

from class_keyboard import Piano
from class_piano_config import piano_configuration
from class_hand_detection import Hands_detection
from class_check_key import check_keys
from class_piano_sound import play_piano_sound


SHAPE=(800,600,3)

class VirPiano():
    def __init__(self,model_path):
        self.model_path=model_path
        self.hand_detection=Hands_detection(self.model_path)
        self.shape=SHAPE
        self.image=np.zeros(self.shape, np.uint8)
        self.num_octaves=2
        self.list_of_octaves=[3,4]
        pts=np.array([[[100,350]],[[700,350]],[[700,550]],[[100,550]]])
        self.piano_keyboard=Piano(pts,self.num_octaves)
        self.piano_config=True
        self.x=[]
        self.y=[]
        self.z=[]
        self.white_piano_notes,self.black_piano_notes=self.get_piano_notes()

    def get_piano_notes(self):
        white_piano_notes=['A0','B0']
        black_piano_notes=['Bb0']
        # white_piano_notes=['A0','B0','A1','B1','C1','D1','E1','F1','G1','A2','B2','C2','D2','E2','F2','G2']
        # black_piano_notes=['Bb0','Bb1','Db1','Eb1','Gb1','Ab1','Bb2','Db2','Eb2','Gb2','Ab2']
        for i in range(self.list_of_octaves[0],self.list_of_octaves[1]+1):
            for note in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                white_piano_notes.append(f"{note}{i}")
            for note in ['Bb', 'Db', 'Eb', 'Gb', 'Ab']:
                black_piano_notes.append(f"{note}{i}")
        return white_piano_notes,black_piano_notes

    def circle_fingertips(self,img):
        radius=4
        color=(0,0,255)
        thickness=-1
        fingertips=[4,8,12,16,20]
        if len(self.x)>0:
            for x,y in zip(self.x,self.y):
                for tip in fingertips:
                    cv2.circle(img,(x[tip],y[tip]),radius,color,thickness)
        return img

    def start(self):
        cap=cv2.VideoCapture(0)
        previousTime=0
        while True:
            _,frame=cap.read()
            frame = cv2.resize(frame, (self.shape[0],self.shape[1]))
            frame = cv2.flip(frame, 1)
            while self.piano_config:
                pts=piano_configuration(self.model_path,self.shape)
                self.piano_keyboard=Piano(pts,self.num_octaves)
                print(pts)
                self.piano_config=False
                cap=cv2.VideoCapture(0)
            self.image=frame.copy()
            self.hand_detection.detect(frame)
            self.x=self.hand_detection.x
            self.y=self.hand_detection.y
            self.z=self.hand_detection.z
            self.image=self.piano_keyboard.make_keyboard(self.image)
            self.image=self.circle_fingertips(self.image)
            pressed_keys,pressed_notes=check_keys(self.x,self.y,self.piano_keyboard.white,self.piano_keyboard.black,self.white_piano_notes,self.black_piano_notes)
            self.image=self.piano_keyboard.change_color(self.image,pressed_keys)
            if len(pressed_notes)>0:
                play_piano_sound(pressed_notes)
            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime
            cv2.putText(self.image, str(int(fps)) + "FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (88, 205, 54), 3)
            cv2.imshow('Hand detection', self.image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = 'hand_landmarker.task'
    vp=VirPiano(model_path)
    vp.start()