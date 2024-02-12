import cv2
import numpy as np

# Initialize a list to store the coordinates of the corner points
corner_points = []


# Define a function to order the points clockwise starting from the top left
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# Define a function to draw the chessboard grid
def draw_chessboard_lines(img, points, chessboard_size):
    # Order the points
    # ordered_points = order_points(np.array(points))
    tl, tr, br, bl = np.array(points)
    print(tl, tr, br, bl)
    # Interpolate points between the corners
    # Interpolate points between the corners
    for i in range(chessboard_size[0] + 1):
        # Interpolate horizontal line points
        start_horiz = tl + (br - tl) * (i / chessboard_size[0])
        end_horiz = tr + (bl - tr) * (i / chessboard_size[0])
        cv2.line(img, tuple(start_horiz.astype(int)), tuple(end_horiz.astype(int)), (0, 255, 0), 2)
        for p in np.linspace(start_horiz, end_horiz, chessboard_size[0] - 2):
            cv2.circle(img, tuple(p.astype(int)), 2, (0, 0, 255), 3)

    for j in range(chessboard_size[1] + 1):
        # Interpolate vertical line points
        start_vert = tl + (tr - tl) * (j / chessboard_size[1])
        end_vert = bl + (br - bl) * ((chessboard_size[1] - j) / chessboard_size[1])
        cv2.line(img, tuple(start_vert.astype(int)), tuple(end_vert.astype(int)), (0, 255, 0), 2)

        # print(start_vert,end_vert)
        # cv2.imshow('Image', img)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
    return img


def draw_circle(event, x, y, flags, param):
    global corner_points, img

    # If the left mouse button was clicked, record the position and draw a circle there
    if event == cv2.EVENT_LBUTTONDOWN:
        corner_points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', img)


def main():
    # Load the image
    img = cv2.imread('images/img.png')
    img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
    cv2.imshow('Image', img)
    # Set the mouse callback function to `draw_circle`
    cv2.setMouseCallback('Image', draw_circle)
    # Wait until the user has clicked four points
    while len(corner_points) < 4:
        cv2.waitKey(1)
    # Chessboard size (number of inner corners per a chessboard row and column)
    chessboard_size = (10, 7)  # 8x8 grid
    # Draw the chessboard grid
    img_with_chessboard = draw_chessboard_lines(img, corner_points, chessboard_size)
    # Show the result
    cv2.imshow('Chessboard', img_with_chessboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
