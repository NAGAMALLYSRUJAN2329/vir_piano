import cv2

from hand_detection import Hands_detection

def circle_fingertips(img,x,y):
    radius=4
    color=(0,0,255)
    thickness=-1
    fingertips=[4,8,12,16,20]
    if len(x)>0:
        for x,y in zip(x,y):
            for tip in fingertips:
                cv2.circle(img,(x[tip],y[tip]),radius,color,thickness)
    return img

cap=cv2.VideoCapture(0)
hd=Hands_detection('hand_landmarker.task')
while True:
    _,frame=cap.read()
    frame=cv2.flip(frame,1)
    hd.detect(frame)
    frame=circle_fingertips(frame,hd.x,hd.y)
    cv2.imshow('g',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

