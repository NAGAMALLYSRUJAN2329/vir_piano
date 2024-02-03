import cv2
import mediapipe as mp
import numpy as np
from keyboard import make_keyboard,convert_coordinates_to_lines,check_key,check_key2, get_keyboard_keys
from piano_sound import play_piano_sound


# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_point_in_area(point, area):
    x, y = point
    # Create a polygon using the area coordinates
    polygon = np.array(area, np.int32)
    polygon = polygon.reshape((-1, 1, 2))
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0


# def check_tip_location(image, point, keys_coordinates):
#     for index, keys in enumerate(keys_coordinates):
#         if is_point_in_area(point, keys):
#             # draw_polygon(image, keys)
#             return index
#     return None  # Or any default value you prefer when the condition is not satisfied

    


# def draw_polygon(image, vertices):
#     pts = np.array(vertices, np.int32)
#     pts = pts.reshape((-1, 1, 2))
#     cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

def get_piano_notes(n):
    white_piano_notes=[]
    black_piano_notes=[]
    white_piano_notes=['A0','B0','A1','B1','C1','D1','E1','F1','G1','A2','B2','C2','D2','E2','F2','G2']
    black_piano_notes=['Bb0','Bb1','Db1','Eb1','Gb1','Ab1','Bb2','Db2','Eb2','Gb2','Ab2']
    return white_piano_notes,black_piano_notes

def find_polygon_index(polygons, fingertip):
    for index, polygon in enumerate(polygons):
        result = is_point_in_area(fingertip, polygon)
        if result == 1:
            return index
    return -1  # Fingertip is outside all polygons

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize variables for both hands
prev_points_left = [None] * 21  # One entry for each hand landmark
distance_traveled_left = [0] * 21  # One entry for each hand landmark
prev_points_right = [None] * 21  # One entry for each hand landmark
distance_traveled_right = [0] * 21  # One entry for each hand landmark
height_threshold = 20  # Set your desired height threshold

white_piano_notes,black_piano_notes=get_piano_notes(2)
shape=(800, 600)
n=2
pts=np.array([[[100,350]],[[700,350]],[[700,550]],[[100,550]]])

last_pressed_key=None
# white_piano_notes,black_piano_notes=get_piano_notes(2)
previousTime=0
configure_piano=False

keys_cordinates=get_keyboard_keys(pts,n)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    frame = cv2.resize(frame, shape)


    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    white,black=make_keyboard(frame,pts,n)
    white_lines,black_lines=convert_coordinates_to_lines(white,black)
    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Identify which hand is detected (left or right)
            hand_landmarks_x = [int(point.x * frame.shape[1]) for point in hand_landmarks.landmark]
            hand_center_x = sum(hand_landmarks_x) / len(hand_landmarks_x)

            # Extract the coordinates of all hand landmarks
            landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in hand_landmarks.landmark]

            # Draw circles at all hand landmarks
            for landmark in landmarks:
                cv2.circle(frame, landmark, 5, (0, 255, 0), -1)

            # Calculate distance traveled for each finger based on the detected hand
            for i in [4,8,12,16,20]:  # Skip the wrist (landmark 0)
                if hand_center_x < frame.shape[1] / 2:  # Left hand
                    if prev_points_left[i] is not None:
                        distance_traveled_left[i] += calculate_distance(landmarks[i], prev_points_left[i])

                        # Check if the finger tip has moved below the height threshold
                        if landmarks[i][1] > prev_points_left[i][1] + height_threshold:
                            print(f"Left Hand - Finger {i} Pressed!")
                            # Reset the distance traveled
                            cx, cy = landmarks[i][0],landmarks[i][1]
                            fingertip_coordinates = (cx,cy)
                            # print(fingertip_coordinates)
                            # print(check_tip_location(frame,fingertip_coordinates,keys_cordinates))
                            # index = find_polygon_index(keys_cordinates, fingertip_coordinates)
                            pressed_key= check_key2(cx,cy,white_lines,black_lines,white_piano_notes,black_piano_notes)
                            if pressed_key and pressed_key!=last_pressed_key:
                                if pressed_key[0]=="Black":
                                    print(pressed_key[0])
                                    play_piano_sound(pressed_key[0])
                                else:
                                    print(pressed_key[0])
                                    play_piano_sound(pressed_key[0])

                            # if index != -1:
                            #     print(f"Fingertip is inside the polygon at index {index}.")
                            # else:
                            #     print("Fingertip is outside all polygons.")

                            distance_traveled_left[i] = 0

                    # Update the previous point for the left hand
                    prev_points_left[i] = landmarks[i]
                else:  # Right hand
                    if prev_points_right[i] is not None:
                        distance_traveled_right[i] += calculate_distance(landmarks[i], prev_points_right[i])

                        # Check if the finger tip has moved below the height threshold
                        if landmarks[i][1] > prev_points_right[i][1] + height_threshold:
                            print(f"Right Hand - Finger {i} Pressed!")
                            # Reset the distance traveled
                            cx, cy = landmarks[i][0],landmarks[i][1]
                            fingertip_coordinates = (cx,cy)
                            # print(fingertip_coordinates)
                            # print(check_tip_location(frame,fingertip_coordinates,keys_cordinates))
                            # index = find_polygon_index(keys_cordinates, fingertip_coordinates)

                            # if index != -1:
                            #     print(f"Fingertip is inside the polygon at index {index}.")
                            # else:
                            #     print("Fingertip is outside all polygons.")
                            pressed_key= check_key2(cx,cy,white_lines,black_lines,white_piano_notes,black_piano_notes)
                            if pressed_key and pressed_key!=last_pressed_key:
                                if pressed_key[0]=="Black":
                                    print(pressed_key[0])
                                    play_piano_sound(pressed_key[0])
                                else:
                                    print(pressed_key[0])
                                    play_piano_sound(pressed_key[0])


                            distance_traveled_right[i] = 0

                    # Update the previous point for the right hand
                    prev_points_right[i] = landmarks[i]


    # Display the frame
    cv2.imshow("Virtual Piano", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
