import cv2
import numpy as np
from helper_functions import *
# Define the chess board size and square size
chessboard_size = (8, 6)
square_size = 115.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corner_points = []
# Prepare object points like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

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
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)  # Draw on the image passed as 'param'
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

        for i in range(0, total_frames, 300):
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

        # Calibrate the camera using the object points and image points
        if len(objpoints) > 0 and len(imgpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
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

        if not ret or img is None:
            print(f"Error reading the first frame from {img_path}")
            continue

        # Display the image and set the mouse callback function
        cv2.namedWindow('Image')
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

    return extrinsic_parameters



def main():
    global corner_points
    # extrinsic_parameters = {}
    # calibration_parameters = calibrate_cameras()
    # extrinsic_parameters = calculate_extrinsics( calibration_parameters)
    # write_camera_configs('data', calibration_parameters, extrinsic_parameters)
    all_camera_configs=read_all_camera_configs('data')
if __name__ == "__main__":
    main()
