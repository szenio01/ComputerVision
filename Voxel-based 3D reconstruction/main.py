import cv2
import numpy as np
from helper_functions import *
import matplotlib as ptl
# # Now you can import your function from main.py
# from Camera Geometric Calibration.main1 import *
# Define the chess board size and square size
chessboard_size = (6, 8)
square_size = 115.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corner_points = []
# Prepare object points like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size
# chessboard = (6, 8)


def online_phase(test_img, objp, mtx, dist,rvec,tvec):
    """
    Performs the online phase of camera calibration by detecting corners in a test image and visualizing 3D objects.

    Args:
        test_img (numpy.ndarray): The test image.
        objp (list): 3D object points.
        mtx (numpy.ndarray): Camera matrix.
        dist (numpy.ndarray): Distortion coefficients.

    Returns:
        None
    """
    # ret, corners = automatic_corner_detection(test_img, criteria, chessboard)
    # if not ret:
    #     print('Cant detect the corners of the test picture')
    #
    # ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)

    # Define 3D coordinates for drawing axes
    axis = np.float32([[0, 0, 0], [300, 0, 0], [0, 300, 0], [0, 0, 300]]).reshape(-1, 3)

    # Define 3D coordinates for a cube
    cube = np.float32([[0, 0, 0], [0, 2, 0], [2, 2, 0], [2, 0, 0],
                       [0, 0, -2], [0, 2, -2], [2, 2, -2], [2, 0, -2]])

    # Project the 3D points onto the 2D plane
    imgpts_axis, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
    imgpts_cube, _ = cv2.projectPoints(cube, rvec, tvec, mtx, dist)
    # Draw the axes and cube
    dst_with_objects = draw(test_img, imgpts_axis, imgpts_cube)

    cv2.namedWindow('Output with 3D Objects', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Output with 3D Objects', 1000, 1200)
    cv2.imshow('Output with 3D Objects', dst_with_objects)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_circle(event, x, y, flags, param):
    """
       Draws a circle on the image when the left mouse button is clicked.

       Args:
           event (int): Type of mouse event.
           x (int): x-coordinate of the mouse position.
           y (int): y-coordinate of the mouse position.
           flags (int): Additional flags.
           param: Additional parameters.

       Returns:
           None
    """
    global corner_points  # Still need this global because it's being modified here
    if event == cv2.EVENT_LBUTTONDOWN:
        corner_points.append((x, y))
        cv2.circle(param, (x, y), 2, (0, 0, 255), -1)  # Draw on the image passed as 'param'
        cv2.imshow('Image', param)  # Display the updated image passed as 'param'


def calibrate_cameras():
    # Set the termination criteria for the corner sub-pixel algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # Dictionary to hold calibration parameters for each camera
    calibration_params = {}

    # Loop to calibrate for each camera
    for cam_id in range(1, 5):
        cap = cv2.VideoCapture(f'data/cam{cam_id}/intrinsics.avi')

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Calculate the number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(0, total_frames, 100):
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, img = cap.read()

            # If the frame was grabbed successfully
            if ret:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

                if ret:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
        accepted_indices = []
        # Calibrate the camera using the object points and image points
        print(len(objpoints))
        if len(objpoints) > 0 and len(imgpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)




            # Calculate and check the reprojection error for each image
            max_acceptable_error = 0.075  # Define the maximum acceptable reprojection error
            for i, (rvec, tvec, objp1, imgp) in enumerate(zip(rvecs, tvecs, objpoints, imgpoints)):
                imgpoints2, _ = cv2.projectPoints(objp1, rvec, tvec, mtx, dist)
                error = cv2.norm(imgp, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                if error <= max_acceptable_error:
                    accepted_indices.append(i)  # Accept the image
                #     print(f"Image {i + 1} accepted with reprojection error: {error}")
                # else:
                #     print(f"Image {i + 1} rejected due to high reprojection error: {error}")

            # Filter objpoints and imgpoints to include only those from accepted images
            filtered_objpoints = [objpoints[i] for i in accepted_indices]
            filtered_imgpoints = [imgpoints[i] for i in accepted_indices]
            print(len(filtered_imgpoints))
            # Calculate mean error using only the accepted images
            mean_error = np.mean(
                [cv2.norm(imgpoints[i], cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)[0],
                          cv2.NORM_L2) / len(imgpoints[i]) for i in accepted_indices])
            print(f"Total mean error after rejection: {mean_error}")
            # Recalibrate the camera using only the accepted images
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(filtered_objpoints, filtered_imgpoints, img.shape[1::-1],
                                                               None,
                                                               None)



            calibration_params[cam_id] = {'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
        else:
            print(f'Not enough points for calibration were found for Camera {cam_id}')

        cap.release()

    return calibration_params


def calculate_extrinsics(calibration_parameters):
    global corner_points, img

    # Dictionary to hold the extrinsic parameters for each camera
    extrinsic_parameters = {}

    for cam_id, params in calibration_parameters.items():
        mtx = params['mtx']
        dist = params['dist']

        # Load the image for corner selection
        img_path = f'data/cam{cam_id}/checkerboard.avi'
        cap = cv2.VideoCapture(img_path)

        # Check if the video is opened successfully
        if not cap.isOpened():
            print(f"Error opening video file {img_path}")
            continue

        # Read the first frame of the video
        ret, img = cap.read()
        cap.release()
        img_new = img.copy()

        if not ret or img is None:
            print(f"Error reading the first frame from {img_path}")
            continue
        # x = img.shape
        # target_size = (x[0]*2, x[1]*2)
        # img = cv2.resize(img,target_size)
        # Display the image and set the mouse callback function
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image', 1000, 1200)
        cv2.imshow('Image', img)
        cv2.setMouseCallback('Image', draw_circle,img)

        # Wait until the user has clicked four points
        while len(corner_points) < 4:
            cv2.waitKey(1)

        # Use the selected points to find the chessboard corners
        _, points = find_corners(img, corner_points, chessboard_size=chessboard_size)
        # Reset corner points for the next camera
        corner_points = []

        # Find the rotation and translation vectors
        ret, rvecs, tvecs = cv2.solvePnP(objp, points, mtx, dist)

        if ret:
            extrinsic_parameters[cam_id] = {'rvecs': rvecs, 'tvecs': tvecs}
        else:
            print(f"Failed to calculate extrinsics for Camera {cam_id}.")

        cv2.destroyAllWindows()


        online_phase(img_new, objp, mtx, dist,rvecs,tvecs)

    return extrinsic_parameters


def create_average_frames_background_model(video_path):
    cap = cv2.VideoCapture(video_path)
    accumulator = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        # Read the video until there are no more frames
        if not ret:
            break
        # Here the point is to find the average frames of the background
        frame_float = frame.astype(np.float32)
        if accumulator is None:
            accumulator = frame_float
        else:
            accumulator += frame_float

        frame_count += 1
    background_model = accumulator / frame_count
    b = background_model.astype(np.uint8)
    cv2.imshow('Background Model', b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    return background_model.astype(np.uint8)


def create_background_model_with_GMM(video_path):
    cap = cv2.VideoCapture(video_path)
    # Create the background subtractor object
    # Feel free to adjust the history, varThreshold, and detectShadows parameters as needed
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the background subtractor to get the foreground mask
        fgMask = backSub.apply(frame)

    # Once done processing all frames, retrieve the background model
    background_model = backSub.getBackgroundImage()

    cap.release()

    # # Display the background model for verification
    # cv2.imshow('Background Model', background_model)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return background_model


def create_background_models():
    background_models = {}
    for cam_id in range(1, 5):
        video_path = f'data/cam{cam_id}/background.avi'
        background_model = create_background_model_with_GMM(video_path)
        background_models[cam_id] = background_model
        # Optionally save the background model to a file
        cv2.imwrite(f'data/cam{cam_id}/background_model.jpg', background_model)
    return background_models


def evaluate_segmentation(mask, ground_truth):
    # Calculate the XOR to find differences
    xor_result = cv2.bitwise_xor(mask, ground_truth)
    # Count the non-zero pixels in the XOR result
    discrepancies = np.count_nonzero(xor_result)
    return discrepancies


def apply_morphological_ops(foreground_mask):
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
    params = cv2.SimpleBlobDetector_Params()
    # Adjust parameters according to your needs
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs
    keypoints = detector.detect(foreground_mask)

    # Draw detected blobs as red circles (adjust drawing parameters as needed)
    im_with_keypoints = cv2.drawKeypoints(foreground_mask, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints


def generate_mask(diff, th_hue, th_sat, th_val):
    # Apply thresholding
    _, thresh_hue = cv2.threshold(diff[:, :, 0], th_hue, 255, cv2.THRESH_BINARY)
    _, thresh_saturation = cv2.threshold(diff[:, :, 1], th_sat, 255, cv2.THRESH_BINARY)
    _, thresh_value = cv2.threshold(diff[:, :, 2], th_val, 255, cv2.THRESH_BINARY)

    # Combine the thresholds to get the final mask
    foreground_mask = cv2.bitwise_and(thresh_hue, thresh_saturation)
    foreground_mask = cv2.bitwise_and(foreground_mask, thresh_value)
    return foreground_mask


def optimize_thresholds(video_frame, ground_truth):
    optimal_thresholds = None
    minimal_discrepancies = float('inf')

    # Example ranges, adjust based on your experimentation
    for th_hue in range(0, 180, 10):
        for th_sat in range(50, 250, 10):
            for th_val in range(50, 250, 10):
                # Generate the mask based on current thresholds
                mask = generate_mask(video_frame, int(th_hue), int(th_sat), int(th_val))
                # Apply morphological operations to reduce noise
                mask_cleaned = apply_morphological_ops(mask)
                # Evaluate the mask against the ground truth
                discrepancies = evaluate_segmentation(mask_cleaned, ground_truth)

                if discrepancies < minimal_discrepancies:
                    minimal_discrepancies = discrepancies
                    optimal_thresholds = (th_hue, th_sat, th_val)

    return optimal_thresholds


def subtraction(video_path, background_model_hsv):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the absolute difference
        diff = cv2.absdiff(frame_hsv, background_model_hsv)

        # Thresholds - adjust these values based on your needs
        th_hue = 0
        th_sat = 5
        th_val = 5

        foreground_mask = generate_mask(diff, th_hue, th_sat, th_val)
        foreground_mask = apply_morphological_ops(foreground_mask)

        #Display the result
        cv2.imshow('Foreground Mask', foreground_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def background_subtraction():
    for cam_id in range(1, 5):
        background_model_path = f'data/cam{cam_id}/background_model.jpg'
        video_path = f'data/cam{cam_id}/video.avi'

        # Read the background model and convert it to HSV
        background_model = cv2.imread(background_model_path)
        background_model_hsv = cv2.cvtColor(background_model, cv2.COLOR_BGR2HSV)
        subtraction(video_path, background_model_hsv)


def main():
    global corner_points
    # extrinsic_parameters = {}
    # calibration_parameters = calibrate_cameras()
    # extrinsic_parameters = calculate_extrinsics(calibration_parameters)

    # write_camera_configs('data', calibration_parameters, extrinsic_parameters)
    all_camera_configs=read_all_camera_configs('data')

    # Task 2. Background subtraction
    # 1. Create a background image
    create_background_models()

    # 2. Create a background image
    background_subtraction()


    #Find Best treshholds
    # video_frame = f'data/cam{1}/frame1.jpg'
    # ground_truth = f'data/cam{1}/frame_manual.jpg'
    # ground_truth = cv2.imread(ground_truth, cv2.IMREAD_GRAYSCALE)
    #
    # # If your ground_truth image is not already binary, apply thresholding
    # _, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
    # print(optimize_thresholds(background_subtraction(), ground_truth))
if __name__ == "__main__":
    main()
