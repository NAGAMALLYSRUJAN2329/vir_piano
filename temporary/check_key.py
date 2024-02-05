import cv2
import numpy as np


prev_points_left = [None] * 21  # One entry for each hand landmark
distance_traveled_left = [0] * 21  # One entry for each hand landmark
prev_points_right = [None] * 21  # One entry for each hand landmark
distance_traveled_right = [0] * 21  # One entry for each hand landmark
height_threshold = 20 
last_pressed_key=None
previousTime=0
configure_piano=False

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def point_and_line(m,b,x,y):
    y_=x*m+b
    if y<=y_:
        return True
    else:
        return False

def is_point_in_area(point, area):
    x, y = point
    # Create a polygon using the area coordinates
    polygon = np.array(area, np.int32)
    polygon = polygon.reshape((-1, 1, 2))
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

def check_key2(x1,y1,white_lines,black_lines,white_piano_notes,black_piano_notes):
    
        pressed_keys=[]
        another_pressed_keys={"White":[],"Black":[]}
   
        flag=False
        for i,key in enumerate(black_lines):
            [[[m0,b0]],[[m1,b1]],[[m2,b2]],[[m3,b3]]]=key
            
            if point_and_line(m0,b0,x1,y1) ^ point_and_line(m2,b2,x1,y1) and point_and_line(m1,b1,x1,y1) ^ point_and_line(m3,b3,x1,y1):
            
                another_pressed_keys['Black'].append(i)
                pressed_keys.append(black_piano_notes[i])
                flag=True
                break
        
        for i,key in enumerate(white_lines):
            [[[m0,b0]],[[m1,b1]],[[m2,b2]],[[m3,b3]]]=key
            
            if point_and_line(m0,b0,x1,y1) ^ point_and_line(m2,b2,x1,y1) and point_and_line(m1,b1,x1,y1) ^ point_and_line(m3,b3,x1,y1):
            
                
                another_pressed_keys['White'].append(i)
                pressed_keys.append(white_piano_notes[i])
                break
        pressed_keys=list(set(pressed_keys))
        another_pressed_keys['White']=list(set(another_pressed_keys['White']))
        another_pressed_keys['Black']=list(set(another_pressed_keys['Black']))
        return pressed_keys


def detect(frame, hand_landmarks, white_lines,black_lines,white_piano_notes,black_piano_notes):
            hand_landmarks_x = [int(point.x * frame.shape[1]) for point in hand_landmarks.landmark]
            hand_center_x = sum(hand_landmarks_x) / len(hand_landmarks_x)

            landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in hand_landmarks.landmark]

            for landmark in landmarks:
                cv2.circle(frame, landmark, 5, (0, 255, 0), -1)

            for i in [4,8,12,16,20]:  # Skip the wrist (landmark 0)
                if hand_center_x < frame.shape[1] / 2:  # Left hand
                    if prev_points_left[i] is not None:
                        distance_traveled_left[i] += calculate_distance(landmarks[i], prev_points_left[i])

                        if landmarks[i][1] > prev_points_left[i][1] + height_threshold:
                            print(f"Left Hand - Finger {i} Pressed!")
                            # Reset the distance traveled
                            cx, cy = landmarks[i][0],landmarks[i][1]
                            fingertip_coordinates = (cx,cy)
                            pressed_key= check_key2(cx,cy,white_lines,black_lines,white_piano_notes,black_piano_notes)
                            distance_traveled_left[i] = 0
                    prev_points_left[i] = landmarks[i]
                    return pressed_key

                else:  # Right hand
                    if prev_points_right[i] is not None:
                        distance_traveled_right[i] += calculate_distance(landmarks[i], prev_points_right[i])

                        # Check if the finger tip has moved below the height threshold
                        if landmarks[i][1] > prev_points_right[i][1] + height_threshold:
                            print(f"Right Hand - Finger {i} Pressed!")
                            # Reset the distance traveled
                            cx, cy = landmarks[i][0],landmarks[i][1]
                            fingertip_coordinates = (cx,cy)
                            
                            pressed_key= check_key2(cx,cy,white_lines,black_lines,white_piano_notes,black_piano_notes)
                            distance_traveled_right[i] = 0

                    # Update the previous point for the right hand
                    prev_points_right[i] = landmarks[i]
                    return pressed_key