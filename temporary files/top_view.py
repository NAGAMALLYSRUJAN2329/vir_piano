import cv2
import mediapipe as mp
import numpy as np
from keyboard import make_keyboard,convert_coordinates_to_lines,check_key,get_keyboard_keys


# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open the webcam
cap = cv2.VideoCapture(0)

shape=(800, 600)
n=2
pts=np.array([[[100,350]],[[700,350]],[[700,550]],[[100,550]]])


pts1=np.float32([[[100,350]],[[700,350]],[[100,550]],[[700,550]]])

last_pressed_key=None
previousTime=0
configure_piano=False

keys_cordinates=get_keyboard_keys(pts,n)

pointIndex = 0
ASPECT_RATIO = (200,600)

pts2 = np.float32([[0,0],[ASPECT_RATIO[1],0],[0,ASPECT_RATIO[0]],[ASPECT_RATIO[1],ASPECT_RATIO[0]]])

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


    # Display the frame
    cv2.imshow("Virtual Piano", frame)

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(frame,M,(600,200))
    # dst = cv2.flip(dst, 0)
    cv2.imshow("output",dst)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
