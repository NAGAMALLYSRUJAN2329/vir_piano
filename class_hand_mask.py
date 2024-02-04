import cv2
import numpy as np

def hand_mask(frame,frame_copy):
    im_ycrcb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2YCR_CB)
    mask = np.zeros_like(frame_copy)
    skin_ycrcb_mint = np.array((0, 133, 77))
    skin_ycrcb_maxt = np.array((255, 173, 127))
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    contours, _ = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(mask, contours, i, (255, 255, 255), -1)
            cv2.drawContours(frame, contours, i, (0, 0, 0), -1)
    result = cv2.bitwise_and(frame_copy,mask)
    result=cv2.add(frame,result)
    return result