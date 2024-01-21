import cv2
import numpy as np

marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
im_src = cv2.imread("C:\\Users\\91706\\Downloads\\ppiano.jpg")

# detect the marker
param_markers = cv2.aruco.DetectorParameters()
# Assuming 'cap' is the video capture object
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = cv2.aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    # Assuming 'marker_IDs' and 'marker_corners' are the lists of detected marker IDs and corners
    
    top_rightcorners=[]
    if marker_IDs is not None and len(marker_IDs) == 4:
        # Extract the corners of the four markers
        count = 0
        for ids, corners in zip(marker_IDs, marker_corners):
            
            cv2.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            centre =  (top_right + top_left)/2 
            dist = ( bottom_right[0] - bottom_left[0])/2
            print(centre)
            top_rightcorners.append(centre)
            # if count==0:

            #     top_rightcorners.append(top_right)
            # elif ( count==1):
            #     top_rightcorners.append(bottom_right)
            # elif ( count==2):
            #     top_rightcorners.append(bottom_left)
            # else:
            #     top_rightcorners.append(top_left)
            cv2.putText(
                frame,
                f"id: {ids[0]}",
                top_right,
                cv2.FONT_HERSHEY_PLAIN,
                1.3,
                (200, 100, 0),
                2,
                cv2.LINE_AA,
            )
            # print(ids, "  ", corners)
            # count= count+1
        
        top_rightcorners = np.array(top_rightcorners).astype(int)
        print(top_rightcorners)
        scalingFac = 0.02;
        distance = np.linalg.norm(top_rightcorners[0][0]-top_rightcorners[1][0])
        pts_dst = [[top_rightcorners[0][0] - round(scalingFac*distance), top_rightcorners[0][1] - round(scalingFac*distance)]];
        pts_dst = pts_dst + [[top_rightcorners[1][0] + round(scalingFac*distance), top_rightcorners[1][1] - round(scalingFac*distance)]];

        pts_dst = pts_dst + [[top_rightcorners[2][0] + round(scalingFac*distance), top_rightcorners[2][1] + round(scalingFac*distance)]];

        pts_dst = pts_dst + [[top_rightcorners[3][0] - round(scalingFac*distance), top_rightcorners[3][1] + round(scalingFac*distance)]];
        cv2.polylines(frame, [np.array(top_rightcorners).astype(int)], True, (0, 255, 255), 2, cv2.LINE_AA)

        pts_src = [[0,0], [im_src.shape[1], 0], [im_src.shape[1], im_src.shape[0]], [0, im_src.shape[0]]];

        pts_src_m = np.asarray(pts_src)
        pts_dst_m = np.asarray(pts_dst)

        # Calculate Homography
        h, status = cv2.findHomography(pts_src_m, pts_dst_m)

        # Warp source image to destination based on homography
        warped_image = cv2.warpPerspective(im_src, h, (frame.shape[1],frame.shape[0]))

        # Prepare a mask representing region to copy from the warped image into the original frame.
        mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8);
        cv2.fillConvexPoly(mask, np.int32([pts_dst_m]), (255, 255, 255), cv2.LINE_AA);

        # Erode the mask to not copy the boundary effects from the warping
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3));
        mask = cv2.erode(mask, element, iterations=3);

        # Copy the mask into 3 channels.
        warped_image = warped_image.astype(float)
        mask3 = np.zeros_like(warped_image)
        for i in range(0, 3):
            mask3[:,:,i] = mask/255

        # Copy the warped image into the original frame in the mask region.
        warped_image_masked = cv2.multiply(warped_image, mask3)
        frame_masked = cv2.multiply(frame.astype(float), 1-mask3)
        im_out = cv2.add(warped_image_masked, frame_masked)

        # Showing the original image and the new output image side by side
        concatenatedOutput = cv2.hconcat([frame.astype(float), im_out]);
        cv2.imshow("AR using Aruco markers", concatenatedOutput.astype(np.uint8))


        # pts_src = [[0,0], [im_src.shape[1], 0], [im_src.shape[1], im_src.shape[0]], [0, im_src.shape[0]]];
        # Display the frame with markers and rectangle
    # cv2.imshow("Augmented Reality", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()









































































































































































































# import numpy as np
# import time
# import cv2

# ARUCO_DICT = {
#     "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
#     "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
#     "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
#     "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
#     "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
#     "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
#     "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
#     "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
#     "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
#     "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
#     "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
#     "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
#     "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
#     "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
#     "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
#     "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
#     "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
#     "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
#     "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
#     "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
#     "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11

# }

# def aruco_display(corners, ids, rejected, image):
#     if len(corners)>0:
#         ids = ids.flatten()

#         for (markersCorner, markerID) in zip(corners, ids):
#             corners = markersCorner.reshape((4,2))
#             (topLeft, topRight, BottonRight, bottomLeft) = corners

#             topRight = (int(topRight[0]), int(topRight[1]))
#             bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
#             bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
#             topLeft = (int(topLeft[0]), int(topLeft[1]))

#             cv2.line(image, topLeft, topRight, (0,225,0), 2)
#             cv2.line(image, topRight, bottomRight, (0,225,0), 2)
#             cv2.line(image, bottomRight,  bottomLeft, (0,225,0), 2)
#             cv2.line(image, bottomLeft, topLeft, (0,225,0), 2)

#             cX = int ((topLeft[0] + bottomRight[0]) /2.0)
#             cY = int ((topLeft[1] + bottomRight[1]) /2.0)
#             cv2.circle(image, (cX, cY), 4, (0,0,225), -1)

#             cv2.putText(image, str(markerID), (topLeft[0], topLeft[1]) -10)
#                 0.5, (0,225,0),2)
#             print("[Inference] ArUco marker ID: {}".format(markerID))

#     return image




