import cv2
import numpy as np
from matplotlib import pyplot as plt

from helper_functions import *
import matplotlib as ptl
from scipy.ndimage import binary_opening, binary_closing
from skimage.measure import marching_cubes
# Define the chess board size and square size
chessboard_size = (6, 8)
square_size = 115.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corner_points = []
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size
directory = "data"

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
    # kernel = np.ones((2,2), np.uint8)
    # foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
    # params = cv2.SimpleBlobDetector_Params()
    # # Adjust parameters according to your needs
    # detector = cv2.SimpleBlobDetector_create(params)
    #
    # # Detect blobs
    # keypoints = detector.detect(foreground_mask)
    #
    # # Initialize an empty mask for filled blobs
    # mask_with_filled_blobs = np.zeros_like(foreground_mask)
    #
    # # Fill the detected blobs in the mask
    # for k in keypoints:
    #     # Drawing filled circles on the mask for each blob
    #     radius = int(k.size / 2)  # Adjust radius as necessary
    #     center = (int(k.pt[0]), int(k.pt[1]))  # Blob center
    #     cv2.circle(mask_with_filled_blobs, center, radius, (255), thickness=5)  # Fill the circle
    # combined_mask = cv2.bitwise_or(foreground_mask, mask_with_filled_blobs)

    # combined_mask = optimize_noise(foreground_mask, max_iterations=10)
    return foreground_mask


def optimize_noise(foreground_mask, max_iterations=10):
    best_mask = foreground_mask.copy()
    lowest_noise = np.inf  # Initialize with infinity; aim to minimize this

    for iteration in range(1, max_iterations + 1):
        # Dynamically adjust the kernel size based on iteration
        kernel_size = iteration
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply erosion to remove noise
        eroded_mask = cv2.erode(foreground_mask, kernel, iterations=1)

        # Apply dilation to restore object size
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

        # Evaluate the segmentation quality (example metric: count isolated pixels)
        noise_level = count_isolated_pixels(dilated_mask)

        # Update the best mask if current iteration has lower noise
        if noise_level < lowest_noise:
            lowest_noise = noise_level
            best_mask = dilated_mask.copy()

    return best_mask


def count_isolated_pixels(mask):
    # Define a 3x3 kernel with all ones. This kernel will be used to count the number of white neighbors for each pixel.
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],  # Notice the center is 0, so we don't count the pixel itself, only its neighbors
                       [1, 1, 1]], dtype=np.uint8)

    # Use convolution to count the number of white neighbors for each white pixel in the mask
    neighbor_count = cv2.filter2D((mask / 255).astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)

    # A pixel is considered isolated if it is white and has 2 or fewer white neighbors
    # Adjust this threshold as needed based on your definition of "isolated"
    isolated_pixels = np.where((mask == 255) & (neighbor_count <= 2), 1, 0)

    # Count the number of isolated pixels
    count = np.sum(isolated_pixels)

    return count


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


def dynamic_threshold_estimation(diff):
    # Example: Use the 95th percentile of the difference values as the threshold
    th_hue = np.percentile(diff[:, :, 0], 80)
    th_sat = np.percentile(diff[:, :, 1], 80)
    th_val = np.percentile(diff[:, :, 2], 80)
    print(th_hue,th_sat,th_val)
    return th_hue, th_sat, th_val


# def subtraction(video_path, background_model_hsv):
#     cap = cv2.VideoCapture(video_path)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Convert the frame to HSV
#         frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#         # Calculate the absolute difference
#         diff = cv2.absdiff(frame_hsv, background_model_hsv)
#
#         # Thresholds - adjust these values based on your needs
#         th_hue = 0
#         th_sat = 5
#         th_val = 5
#
#         foreground_mask = generate_mask(diff, th_hue, th_sat, th_val)
#         foreground_mask = apply_morphological_ops(foreground_mask)
#
#         #Display the result
#         cv2.imshow('Foreground Mask', foreground_mask)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     return foreground_mask

def subtraction(video_path, background_model_hsv):
    cap = cv2.VideoCapture(video_path)
    frame_diffs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV and calculate the absolute difference
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        diff = cv2.absdiff(frame_hsv, background_model_hsv)
        frame_diffs.append(diff)

    # Assuming the use of multiple frames to estimate dynamic thresholds
    # Convert the list of diffs into a single numpy array for easier processing
    diffs = np.stack(frame_diffs, axis=0)
    th_hue, th_sat, th_val = dynamic_threshold_estimation(diffs)

    # Reinitialize the capture and process again with determined thresholds
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    foreground_mask = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        diff = cv2.absdiff(frame_hsv, background_model_hsv)
        foreground_mask = generate_mask(diff, th_hue, th_sat, th_val)
        foreground_mask = apply_morphological_ops(foreground_mask)

        cv2.imshow('Foreground Mask', foreground_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    return foreground_mask

def background_subtraction():
    forground_masks = []
    for cam_id in range(1, 5):
        background_model_path = f'data/cam{cam_id}/background_model.jpg'
        video_path = f'data/cam{cam_id}/video.avi'

        # Read the background model and convert it to HSV
        background_model = cv2.imread(background_model_path)
        background_model_hsv = cv2.cvtColor(background_model, cv2.COLOR_BGR2HSV)
        forground_masks.append(subtraction(video_path, background_model_hsv))
    return forground_masks

# def create_lookup_table(voxel_grid, all_camera_configs):
#     lookup_table = {}
#
#     for voxel in voxel_grid:
#         voxel_3D = np.array([[voxel.x, voxel.y, voxel.z]], dtype=np.float32)  # Voxel's 3D coordinates
#
#         for cam_id, config in all_camera_configs.items():
#             rvec = config['rvecs']
#             tvec = config['tvecs']
#             mtx = config['mtx']
#             dist = config['dist']
#
#             # Project the 3D voxel coordinates to 2D image coordinates
#             projected_2D, _ = cv2.projectPoints(voxel_3D, rvec, tvec, mtx, dist)
#
#             # Store the projected 2D coordinates in the lookup table
#             if voxel not in lookup_table:
#                 lookup_table[voxel] = {}
#             lookup_table[voxel][cam_id] = projected_2D[0][0]  # First point, first coordinate pair
#
#     return lookup_table


def parse_camera_config_from_file(config_path):
    # Load and parse the XML file
    tree = ET.parse(config_path)
    root = tree.getroot()

    # Extract and parse the camera matrix
    camera_matrix_rows = root.find('Intrinsics/CameraMatrix').findall('Row')
    camera_matrix = np.array([[float(num) for num in row.text.split()] for row in camera_matrix_rows])

    # Extract and parse the distortion coefficients
    distortion_coeffs_row = root.find('Intrinsics/DistortionCoefficients/Row').text
    distortion_coeffs = np.array([float(num) for num in distortion_coeffs_row.split()])

    # Extract and parse the rotation vector
    rotation_vector_row = root.find('Extrinsics/RotationVectors/Vector').text
    rotation_vector = np.array([float(num) for num in rotation_vector_row.split()])

    # Extract and parse the translation vector
    translation_vector_row = root.find('Extrinsics/TranslationVectors/Vector').text
    translation_vector = np.array([float(num) for num in translation_vector_row.split()])

    return camera_matrix, distortion_coeffs, rotation_vector, translation_vector


def project_to_2d(points_3d, camera_matrix, dist_coeffs, rvec, tvec):
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)


def create_lut(voxels):
    lut = {}
    for cam_id in range(1, 5):  # Assuming 4 cameras
        config_path = f"{directory}/cam{cam_id}/camera_properties.xml"
        camera_matrix, dist_coeffs, rvec, tvec = parse_camera_config_from_file(config_path)

        # Project all voxels to 2D for this camera
        points_2d = project_to_2d(voxels, camera_matrix, dist_coeffs, rvec, tvec)
        lut[cam_id] = points_2d  # Store the 2D points in the LUT

    return lut


# def check_voxel_visibility(voxel_index, lut, silhouette_masks):
#     visible_in_any_camera = False
#     for cam_id, points_2d in lut.items():
#         point = points_2d[voxel_index]
#         x, y = int(point[0]), int(point[1])
#         # Check if the point is within the image bounds and corresponds to the foreground
#         if 0 <= x < silhouette_masks[cam_id].shape[1] and 0 <= y < silhouette_masks[cam_id].shape[0]:
#             if silhouette_masks[cam_id][y, x] == 255:
#                 visible_in_any_camera = True
#                 break  # No need to check other cameras if visible in one
#         if visible_in_any_camera:
#             break
#
#     return visible_in_any_camera


def check_voxel_visibility(voxel_index, lut, silhouette_masks):
    visible_in_every_camera = True
    for cam_id, points_2d in lut.items():
        # Adjust cam_id for 0-based indexing when accessing the silhouette_masks array
        adjusted_cam_id = cam_id - 1  # Adjusting cam_id to 0-based index
        point = points_2d[voxel_index]
        x, y = int(point[0]), int(point[1])

        # Ensure the adjusted_cam_id is used for indexing silhouette_masks
        if 0 <= x < silhouette_masks[adjusted_cam_id].shape[1] and 0 <= y < silhouette_masks[adjusted_cam_id].shape[0]:
            if silhouette_masks[adjusted_cam_id][y, x] == 255:
                pass
                # break  # No need to check other cameras if visible in one
            else:
                return False
        #
        # if visible_in_any_camera:
        #     break

    return visible_in_every_camera


def check_visibility_and_reconstruct(silhouette_masks):
    # Define the 3D grid (example)

    x_range = np.linspace(-1000, 1000, num=100)
    y_range = np.linspace(-1000, 1000, num=100)
    z_range = np.linspace(0, 2000, num=100)
    voxels = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)
    # print(x_range)
    # print(voxels.shape)
    lookup_table = create_lut(voxels)
    visible_points=[]
    # Initialization of the 3D reconstruction space
    reconstruction_space = np.zeros((100, 100, 100), dtype=bool)
    # Assuming silhouette_masks is defined
    for voxel_index in range(len(voxels)):
        x, y, z = voxels[voxel_index]  # Get voxel coordinates (you may need to adjust the mapping based on your grid definition)

        if check_voxel_visibility(voxel_index, lookup_table, silhouette_masks):
            # This voxel is part of the reconstruction
            # print("hello")

            # reconstruction_space[voxel_index // (100 * 100), (voxel_index // 100) % 100, voxel_index % 100] = True
            # Corrected the indexing to reflect the voxel grid setup
            ix = min(int((x + 1000) / 20), 99)
            iy = min(int((y + 1000) / 20), 99)
            iz = min(int(z / 20 ), 99)

            reconstruction_space[ix, iy, iz] = True
            visible_points.append([ix, iy, iz])  # Add visible voxel center to the list
            # visible_points.append([x, y, z])

    with open("Computer-Vision-3D-Reconstruction/voxels.txt", "w") as file:
        for point in visible_points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

    # Convert the list of visible points to a NumPy array for easier plotting
    visible_points = np.array(visible_points)

    # Create a 3D plot of the visible points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(visible_points[:, 0], visible_points[:, 1], visible_points[:, 2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('3D Visualization of Visible Voxels')
    plt.show()



def main():
    global corner_points
    # extrinsic_parameters = {}
    # calibration_parameters = calibrate_cameras()
    # extrinsic_parameters = calculate_extrinsics(calibration_parameters)
    #
    # write_camera_configs('data', calibration_parameters, extrinsic_parameters)
    # all_camera_configs=read_all_camera_configs('data')

    # Task 2. Background subtraction
    # 1. Create a background image
    create_background_models()

    # 2. Create a background image
    foreground_masks = background_subtraction()
    foreground_masks = np.array(foreground_masks)
    # cv2.imshow('Background Model', forground_masks[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 3. Create a background image
    check_visibility_and_reconstruct(foreground_masks)


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
