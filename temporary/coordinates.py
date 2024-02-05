import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize variables
pinch_coordinates = []
drawn_circles = []

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get coordinates of thumb and forefinger
            hand_landmarks_x = [int(point.x * frame.shape[1]) for point in hand_landmarks.landmark]
            hand_center_x = sum(hand_landmarks_x) / len(hand_landmarks_x)

            # Extract the coordinates of all hand landmarks
            landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in hand_landmarks.landmark]

            # Draw circles at all hand landmarks
            for landmark in landmarks:
                cv2.circle(frame, landmark, 5, (0, 255, 0), -1)
            
            
            thumb_x = landmarks[4][0]
            thumb_y = landmarks[4][1]
            forefinger_x = landmarks[8][0]
            forefinger_y = landmarks[8][1]


            # Calculate distance between thumb and forefinger
            distance =  (thumb_y-forefinger_y)
            # print(distance)

            # Check for pinch gesture (adjust the threshold as needed)
            if distance < 30:

                if len(pinch_coordinates) < 3:
                # Append the coordinates to the list
                    if pinch_coordinates:
                        # Check if the new pinch has at least a 100-pixel difference in either x-axis or y-axis from the previous detected pinch
                        if abs(forefinger_x  - pinch_coordinates[-1][0]) >= 100 or abs(forefinger_y  - pinch_coordinates[-1][1]) >= 100:
                            # Append the coordinates to the list
                            pinch_coordinates.append((int(forefinger_x ), int(forefinger_y )))
                            print("Pinch gesture detected!")
                            print("Coordinates:", pinch_coordinates)
                            radius = 30  # You can adjust the radius as needed
                            center = (int(forefinger_x), int(forefinger_y ))
                            drawn_circles.append((center, radius))
                    else:
                        # Append the coordinates to the list if it's the first pinch detected
                        pinch_coordinates.append((int(forefinger_x ), int(forefinger_y )))
                        print("Pinch gesture detected!")
                        print("Coordinates:", pinch_coordinates)
                        radius = 30  # You can adjust the radius as needed
                        center = (int(forefinger_x), int(forefinger_y ))
                        drawn_circles.append((center, radius))
                
                else:
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            # Draw landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    for circle in drawn_circles:
        cv2.circle(frame, circle[0], circle[1], (0, 0, 255), thickness=3)




    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
