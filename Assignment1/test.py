import cv2
import numpy as np


def automatic_corner_detection(img, criteria, chessboard_size=(3, 3)):
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("output", int(img.shape[1] / 4), int(img.shape[0] / 4))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    # gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow('output', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    print(ret, corners)
    # If found, add object points, image points (after refining them)
    if ret:
        # This function refines corner locations to subpixel accuracy

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", int(img.shape[1] / 4), int(img.shape[0] / 4))
        cv2.imshow('output', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return corners
    else:
        raise ValueError("Chessboard corners not found")


def main():
    # Define the criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Load the image
    img_path = 'images/testing_image.jpg'
    img = cv2.imread(img_path)
    # img = cv2.rotate(img, cv2.ROTATE_180)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", int(img.shape[1] / 4), int(img.shape[0] / 4))
    cv2.imshow('output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    rows, cols, ch = img.shape

    # Automatically find corners
    try:
        corners = automatic_corner_detection(img, criteria, chessboard_size=(9, 6))
        # The first and last corner points are assumed to be the top-left and bottom-right points of the chessboard
        ordered_points = np.array([corners[0, 0, :], corners[6, 0, :], corners[-1, 0, :], corners[-7, 0, :]],
                                  dtype="float32")
        print(ordered_points)

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
