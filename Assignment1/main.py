import cv2
import numpy as np
import os

from matplotlib import pyplot as plt

from Assignment1.manual_corners import *

chessboard = (9, 6)
image_dir = "images"
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
DEBUG = False


def automatic_corner_detection(img, criteria, chessboard_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        # This function refines corner locations to subpixel accuracy

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("output", int(img.shape[1] / 3), int(img.shape[0] / 3))
        # cv2.imshow('output', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return ret, corners


def automatic_corner_detection_live(img, criteria, chessboard_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)

    return ret, corners


def undistort(gray_test_img, mtx, dist):
    h, w = gray_test_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(gray_test_img, mtx, dist, None, newcameramtx)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return newcameramtx, dst


def mean_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))


def online_phase(test_img, objp, mtx, dst, dist, newcameramtx):
    ret, corners = automatic_corner_detection(test_img, criteria, chessboard)
    if not ret:
        print('Cant detect the corners of the test picture')

    ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
    # Define 3D coordinates for drawing axes
    # axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    axis = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    # Define 3D coordinates for a cube
    cube = np.float32([[0, 0, 0], [0, 2, 0], [2, 2, 0], [2, 0, 0],
                       [0, 0, -2], [0, 2, -2], [2, 2, -2], [2, 0, -2]])
    # Project the 3D points onto the 2D plane
    imgpts_axis, _ = cv2.projectPoints(axis, rvec, tvec, newcameramtx, dist)
    imgpts_cube, _ = cv2.projectPoints(cube, rvec, tvec, newcameramtx, dist)

    # Draw the axes and cube
    # Correct the function call
    dst_with_objects = draw(dst, imgpts_axis, imgpts_cube)
    cv2.imshow('Output with 3D Objects', dst_with_objects)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def online_phase_live(frame, objp, mtx, dist):
    ret, corners = automatic_corner_detection_live(frame, criteria, chessboard)
    axis = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    # Define 3D coordinates for a cube
    cube = np.float32([[0, 0, 0], [0, 2, 0], [2, 2, 0], [2, 0, 0],
                       [0, 0, -2], [0, 2, -2], [2, 2, -2], [2, 0, -2]])
    if ret:
        ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
        imgpts_axis, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
        imgpts_cube, _ = cv2.projectPoints(cube, rvec, tvec, mtx, dist)
        frame_with_objects = draw(frame, imgpts_axis, imgpts_cube)
    # Display the frame with objects drawn
        cv2.imshow('Webcam - Press Q to Quit', frame_with_objects)
    else:
        # Display the original frame if corner detection fails
        cv2.imshow('Webcam - Press Q to Quit', frame)

def draw(img, imgpts_axis, imgpts_cube):
    # Draw axes lines
    origin = tuple(imgpts_axis[0].ravel().astype(int))
    for pt in imgpts_axis[1:]:
        img = cv2.line(img, origin, tuple(pt.ravel().astype(int)), (255, 0, 0), 5)
    # Draw cube
    imgpts = np.int32(imgpts_cube).reshape(-1, 2)
    # Draw bottom
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 3)
    # Draw pillars
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
    # Draw top
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img
def draw_circle(event, x, y, flags, param):
    global corner_points
    # If the left mouse button was clicked, record the position and draw a circle there
    if event == cv2.EVENT_LBUTTONDOWN:
        corner_points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', img)

def Run1():
    objpoints = []
    imgpoints = []
    global img  # Ensure img is accessible globally
    global corner_points  # Declare this if it's also used globally
    corner_points = []  # Initialize corner_points list if not already initialized
    # Defining the world coordinates for 3D points
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    for filename in os.listdir(image_dir):
        if "test" not in filename:
            path = os.path.join(image_dir, filename)
            img = cv2.imread(path)
            ret, corners = automatic_corner_detection(img, criteria, chessboard)
            if ret:
                if DEBUG:
                    print("Automatic corner detection was succesfull for image " + path)
                objpoints.append(objp)
                imgpoints.append(corners)
            else:
                print("FAIL to detect corners for image " + path)
                cv2.namedWindow('Image')  # Create the window named 'Image'
                cv2.imshow('Image', img)  # Show the image in 'Image' window
                cv2.setMouseCallback('Image', draw_circle)  # Now set the mouse callback

                # Wait until the user has clicked four points
                while len(corner_points) < 4:
                    cv2.waitKey(1)
                img_with_chessboard = draw_chessboard_lines(img, corner_points,chessboard_size=(7,10))
                corner_points = []
                # Show the result
                cv2.imshow('Chessboard', img_with_chessboard)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    if DEBUG:
        print("Camera matrix : \n")
        print(mtx)
        print("dist : \n")
        print(dist)
        print("rvecs : \n")
        print(rvecs)
        print("tvecs : \n")
        print(tvecs)

    # # Undistort test image
    # newcameramtx, dst = undistort(gray_test_img, mtx, dist)

    # Calculate mean error
    mean_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)
    # Online_phase
    # online_phase(test_img, objp, mtx, test_img, dist, mtx)
    # online_phase(test_img, objp, mtx, dst, dist, newcameramtx)
    return mtx, dist, objp


def Run2():
    objpoints = []
    imgpoints = []
    global img  # Ensure img is accessible globally
    global corner_points  # Declare this if it's also used globally
    corner_points = []  # Initialize corner_points list if not already initialized
    # Defining the world coordinates for 3D points
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    for i in range(10,20):
        path = os.path.join(image_dir, f'IMG-20240212-WA00{i}.jpg')
        img = cv2.imread(path)
        ret, corners = automatic_corner_detection(img, criteria, chessboard)
        if ret:
            if DEBUG:
                print("Automatic corner detection was succesfull for image " + path)
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("FAIL to detect corners for image " + path)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

    # Calculate mean error
    mean_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)
    # Online_phase
    # online_phase(test_img, objp, mtx, test_img, dist, mtx)
    # online_phase(test_img, objp, mtx, dst, dist, newcameramtx)
    return mtx, dist, objp


def Run3():
    objpoints = []
    imgpoints = []
    global img  # Ensure img is accessible globally
    global corner_points  # Declare this if it's also used globally
    corner_points = []  # Initialize corner_points list if not already initialized
    # Defining the world coordinates for 3D points
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    for i in range(15,20):
        path = os.path.join(image_dir, f'IMG-20240212-WA00{i}.jpg')
        img = cv2.imread(path)
        ret, corners = automatic_corner_detection(img, criteria, chessboard)
        if ret:
            if DEBUG:
                print("Automatic corner detection was succesfull for image " + path)
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("FAIL to detect corners for image " + path)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

    # Calculate mean error
    mean_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)
    # Online_phase
    # online_phase(test_img, objp, mtx, test_img, dist, mtx)
    # online_phase(test_img, objp, mtx, dst, dist, newcameramtx)
    return mtx, dist, objp


def live_camera(mtx, dist, objp):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open video capture.")
        return
    while True:
        ret, frame = cap.read()
        processed_frame = frame.copy()
        if not ret:
            print("Failed to grab frame")
            break


        online_phase_live(processed_frame, objp, mtx, dist)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # After the loop release the cap object
    cap.release()
    cv2.destroyAllWindows()


def main():
    test_img = cv2.imread('images/IMG-20240212-WA0009.jpg')
    print("Run 1: Offline Phase")
    mtx, dist, objp = Run1()
    print("Run 1: Online Phase")
    online_phase(test_img, objp, mtx, test_img, dist, mtx)
    test_img = cv2.imread('images/IMG-20240212-WA0009.jpg')
    print("\nRun 2: Offline Phase")
    mtx, dist, objp = Run2()
    print("Run 2: Online Phase")
    online_phase(test_img, objp, mtx, test_img, dist, mtx)
    test_img = cv2.imread('images/IMG-20240212-WA0009.jpg')
    print("\nRun 3: Offline Phase")
    mtx, dist, objp = Run3()
    print("Run 3: Online Phase")
    online_phase(test_img, objp, mtx, test_img, dist, mtx)

    # CHOICES
    live_camera(mtx, dist, objp)

if __name__ == "__main__":
    main()
