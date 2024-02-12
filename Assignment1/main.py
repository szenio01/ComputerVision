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
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", int(img.shape[1] / 3), int(img.shape[0] / 3))
        # cv2.imshow('output', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return ret, corners


def draw(img, corner, imgpts):
    # Ensure the corner is in the correct format
    corner = tuple(int(i) for i in corner.ravel())

    # Draw X axis in red
    x_axis_end_point = tuple(int(i) for i in imgpts[1].ravel())
    img = cv2.line(img, corner, x_axis_end_point, (255, 0, 0), 5)

    # Draw Y axis in green
    y_axis_end_point = tuple(int(i) for i in imgpts[2].ravel())
    img = cv2.line(img, corner, y_axis_end_point, (0, 255, 0), 5)

    # Draw Z axis in blue
    z_axis_end_point = tuple(int(i) for i in imgpts[3].ravel())
    img = cv2.line(img, corner, z_axis_end_point, (0, 0, 255), 5)

    return img


def drawAxisAndCube(ret, mtx, dist, objpoints, imgpoints, img_path):
    # Load the image
    img = cv2.imread(img_path)
    objpoints = np.array(objpoints, dtype=np.float32)
    imgpoints = np.array(imgpoints, dtype=np.float32)
    objpoints = np.reshape(objpoints, (-1, 3))  # Reshape to ensure shape is (-1, 3)
    imgpoints = np.reshape(imgpoints, (-1, 2))  # Reshape to ensure shape is (-1, 2)

    # Solve for camera pose
    ret, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, mtx, dist)

    # Define 3D points for axes and cube
    axis = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    cube = np.float32([[0, 0, 0], [0, 2, 0], [2, 2, 0], [2, 0, 0],
                       [0, 0, -2], [0, 2, -2], [2, 2, -2], [2, 0, -2]])

    # Project 3D points to 2D
    imgpts, jac = cv2.projectPoints(np.concatenate((axis, cube)), rvec, tvec, mtx, dist)

    # Draw axes and cube
    img_with_axes_and_cube = draw(img, imgpts[0].reshape(-1, 1, 2), imgpts[1:])

    # Display the result
    cv2.imshow('Image with 3D Axes and Cube', img_with_axes_and_cube)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Run1():
    objpoints = []
    imgpoints = []
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, chessboard[0] * chessboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

    print("Run 1: Using all training samples")
    for filename in os.listdir(image_dir):
        if "test" not in filename:
            path = os.path.join(image_dir, filename)
            img = cv2.imread(path)
            ret, corners = automatic_corner_detection(img, criteria, chessboard_size=(9, 6))
            if ret:
                if DEBUG:
                    print("Automatic corner detection was succesfull for image " + path)
                objpoints.append(objp)
                imgpoints.append(corners)
            else:
                print("FAIL to detect corners for image " + path)
                cv2.imshow('Select 4 corners manually', img)
                cv2.setMouseCallback('Image', draw_circle)

                # Wait until the user has clicked four points
                while len(corner_points) < 4:
                    cv2.waitKey(1)
                img_with_chessboard = draw_chessboard_lines(img, corner_points, chessboard)

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

    drawAxisAndCube(ret, mtx=mtx, dist=dist, objpoints=objpoints, imgpoints=imgpoints,
                    img_path="images/IMG-20240212-WA0012.jpg")


def main():
    Run1()


if __name__ == "__main__":
    main()
