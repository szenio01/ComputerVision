import cv2
import numpy as np
import xml.etree.ElementTree as ET
chessboard_size = (6, 8)
square_size = 115.0
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size


def mean_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    """
      Computes the mean reprojection error of a camera calibration.

      Args:
          objpoints (list): List of 3D object points.
          imgpoints (list): List of 2D image points.
          mtx (numpy.ndarray): Camera matrix.
          dist (numpy.ndarray): Distortion coefficients.
          rvecs (list): List of rotation vectors.
          tvecs (list): List of translation vectors.

      Returns:
          None
    """
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))


def write_camera_configs(directory, calibration_parameters, extrinsic_parameters):
    """
    Writes camera configuration parameters to XML files.

    Parameters:
        directory: The base directory where the XML files will be saved.
        calibration_parameters: A dictionary containing the intrinsic parameters for each camera.
        extrinsic_parameters: A dictionary containing the extrinsic parameters for each camera.
    """
    for cam_id, intrinsic_params in calibration_parameters.items():
        extrinsic_params = extrinsic_parameters.get(cam_id, {})

        # Create the root element
        root = ET.Element("CameraProperties")

        # Add intrinsic parameters
        intrinsics = ET.SubElement(root, "Intrinsics")

        # Camera matrix
        camera_matrix = ET.SubElement(intrinsics, "CameraMatrix")
        for row in intrinsic_params['mtx']:
            ET.SubElement(camera_matrix, "Row").text = ' '.join(map(str, row))

        # Distortion coefficients
        distortion = ET.SubElement(intrinsics, "DistortionCoefficients")
        ET.SubElement(distortion, "Row").text = ' '.join(map(str, intrinsic_params['dist'].ravel()))

        # Add extrinsic parameters
        extrinsics = ET.SubElement(root, "Extrinsics")

        # Rotation vectors
        rotation = ET.SubElement(extrinsics, "RotationVectors")
        ET.SubElement(rotation, "Vector").text = ' '.join(map(str, extrinsic_params['rvecs'].ravel()))

        # Translation vectors
        translation = ET.SubElement(extrinsics, "TranslationVectors")
        ET.SubElement(translation, "Vector").text = ' '.join(map(str, extrinsic_params['tvecs'].ravel()))

        # Write to an XML file
        tree = ET.ElementTree(root)
        tree.write(f"{directory}/cam{cam_id}/camera_properties.xml")

def read_all_camera_configs(directory):
    """
    Reads camera configuration parameters from XML files for all cameras.

    Parameters:
        directory: The base directory from which the XML files will be read.

    Returns:
        A dictionary containing the camera configurations, indexed by camera ID. Each entry
        contains nested dictionaries for 'intrinsics' and 'extrinsics' parameters.
    """
    all_camera_configs = {}
    for cam_id in range(1, 5):  # Adjust the range based on the number of cameras
        config_path = f"{directory}/cam{cam_id}/camera_properties.xml"
        print(config_path)
        try:
            tree = ET.parse(config_path)
            root = tree.getroot()

            # Initialize storage for intrinsic and extrinsic parameters
            intrinsics = {'mtx': [], 'dist': []}
            extrinsics = {'rvecs': [], 'tvecs': []}

            # Read intrinsic parameters
            mtx_elems = root.find('Intrinsics/CameraMatrix')
            if mtx_elems is not None:
                for row in mtx_elems.findall('Row'):
                    intrinsics['mtx'].append([float(val) for val in row.text.split()])

            dist_elems = root.find('Intrinsics/DistortionCoefficients')
            if dist_elems is not None:
                for row in dist_elems.findall('Row'):
                    intrinsics['dist'].extend([float(val) for val in row.text.split()])

            # Read extrinsic parameters
            rvecs_elems = root.find('Extrinsics/RotationVectors/Vector')
            if rvecs_elems is not None:
                extrinsics['rvecs'].append([float(val) for val in rvecs_elems.text.split()])

            tvecs_elems = root.find('Extrinsics/TranslationVectors/Vector')
            if tvecs_elems is not None:
                extrinsics['tvecs'].append([float(val) for val in tvecs_elems.text.split()])

            all_camera_configs[cam_id] = {'intrinsics': intrinsics, 'extrinsics': extrinsics}

        except FileNotFoundError:
            print(f"Configuration file for camera {cam_id} not found.")

    return all_camera_configs


def automatic_corner_detection(img, criteria, chessboard_size, showboard=False):
    """
    Automatically detects corners of a chessboard pattern on an image.

    Args:
        img (numpy.ndarray): The input image.
        criteria: Termination criteria for the iterative corner refinement process.
        chessboard_size (tuple): Tuple containing the number of inner corners per row and column.
        showboard (bool, optional): Whether to display the detected chessboard corners on the image.

    Returns:
        tuple: A tuple containing a boolean indicating successful detection and an array of detected corner points.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if showboard:
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("output", int(img.shape[1] / 3), int(img.shape[0] / 3))
            cv2.imshow('output', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return ret, corners

def draw(img, imgpts_axis, imgpts_cube):
    """
       Draws 3D axes and a cube on the input image.

       Args:
           img (numpy.ndarray): Input image.
           imgpts_axis (numpy.ndarray): Image points of the 3D axes.
           imgpts_cube (numpy.ndarray): Image points of the cube.

       Returns:
           numpy.ndarray: Image with 3D axes and cube drawn.
    """
    # Draw axes lines
    origin = tuple(imgpts_axis[0].ravel().astype(int))
    print(origin)
    for pt in imgpts_axis[1:]:
        print(pt)
        img = cv2.line(img, origin, tuple(pt.ravel().astype(int)), (255, 0, 0), 2)
    # # Draw cube
    # imgpts = np.int32(imgpts_cube).reshape(-1, 2)
    # # Draw bottom
    # img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 3)
    # # Draw pillars
    # for i, j in zip(range(4), range(4, 8)):
    #     img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
    # # Draw top
    # img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img

def find_corners(img, points, chessboard_size):
    """
       Finds the corners of a chessboard on an image.

       Args:
           img (numpy.ndarray): The input image.
           points (list): List of corner points of the chessboard.
           chessboard_size (tuple): Tuple containing the number of inner corners per row and column.

       Returns:
           tuple: A tuple containing the image with lines and circles drawn on the corners and an array of all detected
           points.
       """
    tl, tr, br, bl = np.array(points)
    # print(tl, tr, br, bl)
    all_points = []

    for j in range(chessboard_size[1]):
        # Interpolate vertical line points
        start_vert = tl + (tr - tl) * (j / (chessboard_size[1]-1))
        end_vert = bl + (br - bl) * ((chessboard_size[1]-1 - j) / (chessboard_size[1]-1))
        cv2.line(img, tuple(start_vert.astype(int)), tuple(end_vert.astype(int)), (0, 255, 0), 2)
        for p in np.linspace(start_vert, end_vert, chessboard_size[0]):
            all_points.append([p])
            cv2.circle(img, tuple(p.astype(int)), 2, (0, 0, 255), 3)

    all_points_np = np.array(all_points, dtype=np.float32).reshape(-1, 1, 2)

    return img, all_points_np


def match_color_distribution(source_img, reference_img):
    """
    Matches the color distribution of the source image to that of the reference image.

    Parameters:
        source_img: The source image whose color distribution is to be adjusted.
        reference_img: The reference image with the desired color distribution.

    Returns:
        An image (numpy array) with the color distribution of the source image
        matched to that of the reference image.
    """
    matched_img = np.zeros_like(source_img)
    for channel in range(3):  # For each color channel
        source_mean, source_std = cv2.meanStdDev(source_img[:, :, channel])
        reference_mean, reference_std = cv2.meanStdDev(reference_img[:, :, channel])

        scaled_img = ((source_img[:, :, channel] - source_mean) * (reference_std / source_std)) + reference_mean
        matched_img[:, :, channel] = np.clip(scaled_img, 0, 255)

    return matched_img.astype(np.uint8)


def white_balance(img):
    """
    Applies a simple white balance algorithm to an image.

    Parameters:
        img: The source image to be white balanced.

    Returns:
        A white balanced image as a numpy array of type uint8.
    """
    # Convert image to float32 for processing
    img_float = img.astype(np.float32)
    # Calculate the average of each channel (R, G, B)
    avg_rgb = np.mean(img_float, axis=(0, 1))
    # Calculate the average intensity over all channels
    avg_intensity = np.mean(avg_rgb)
    # Scale each channel to adjust the white balance
    img_float[:, :, 0] *= avg_intensity / avg_rgb[0]
    img_float[:, :, 1] *= avg_intensity / avg_rgb[1]
    img_float[:, :, 2] *= avg_intensity / avg_rgb[2]
    # Clip values to [0, 255] and convert back to uint8
    balanced_img = np.clip(img_float, 0, 255).astype(np.uint8)
    return balanced_img


def correct_images(coloured_images, foreground_masks):
    """
      Corrects a set of colored images for white balance and color distribution.

      Parameters:
          coloured_images: A list of colored images (numpy arrays) to be corrected.
          foreground_masks: A list of binary foreground masks corresponding to the coloured images.

      Returns:
          A tuple containing:
          - A list of color and white balance corrected images.
          - The original list of foreground masks, unchanged.
      """
    coloured_images = np.array(coloured_images)
    foreground_masks = np.array(foreground_masks)
    # Apply white balance
    white_balanced_images = [white_balance(img) for img in coloured_images]
    reference_image = white_balanced_images[0]
    # Apply color matching
    coloured_images_corrected = [match_color_distribution(img, reference_image) for img in white_balanced_images]
    return coloured_images_corrected, foreground_masks


def online_phase(test_img, objp, mtx, dist, rvec, tvec):
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

    # Define 3D coordinates
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
    """
       Calibrates cameras based on chessboard patterns found in video sequences.


       Returns:
           A dictionary containing the calibration parameters ('mtx', 'dist', 'rvecs', 'tvecs')
           for each camera, keyed by camera ID.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_params = {}
    # Loop to calibrate for each camera
    for cam_id in range(1, 5):
        cap = cv2.VideoCapture(f'data/cam{cam_id}/intrinsics.avi')
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(0, total_frames, 100):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, img = cap.read()
            if ret:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

                if ret:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
        accepted_indices = []
        # Calibrate the camera
        if len(objpoints) > 0 and len(imgpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            # Calculate and check the reprojection error for each image
            max_acceptable_error = 0.075  #  maximum acceptable reprojection error
            for i, (rvec, tvec, objp1, imgp) in enumerate(zip(rvecs, tvecs, objpoints, imgpoints)):
                imgpoints2, _ = cv2.projectPoints(objp1, rvec, tvec, mtx, dist)
                error = cv2.norm(imgp, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                if error <= max_acceptable_error:
                    accepted_indices.append(i)  # Accept the image

            # Filter objpoints and imgpoints for accepted images
            filtered_objpoints = [objpoints[i] for i in accepted_indices]
            filtered_imgpoints = [imgpoints[i] for i in accepted_indices]
            # print(len(filtered_imgpoints))
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
    """
       Calculates extrinsic parameters for each camera based on user-selected corner points.

       Parameters:
           calibration_parameters: A dictionary containing the intrinsic calibration parameters
                                   for each camera, including the camera matrix and distortion
                                   coefficients.

       Returns:
           A dictionary containing the extrinsic parameters (rotation and translation vectors)
           for each camera, keyed by camera ID.
    """
    global corner_points, img
    extrinsic_parameters = {}

    for cam_id, params in calibration_parameters.items():
        mtx = params['mtx']
        dist = params['dist']
        img_path = f'data/cam{cam_id}/checkerboard.avi'
        cap = cv2.VideoCapture(img_path)
        if not cap.isOpened():
            print(f"Error opening video file {img_path}")
            continue
        ret, img = cap.read()
        cap.release()
        img_new = img.copy()
        if not ret or img is None:
            print(f"Error reading the first frame from {img_path}")
            continue

        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image', 1000, 1200)
        cv2.imshow('Image', img)
        cv2.setMouseCallback('Image', draw_circle, img)

        while len(corner_points) < 4:
            cv2.waitKey(1)

        _, points = find_corners(img, corner_points, chessboard_size=chessboard_size)
        corner_points = []

        # Find the rotation and translation vectors
        ret, rvecs, tvecs = cv2.solvePnP(objp, points, mtx, dist)

        if ret:
            extrinsic_parameters[cam_id] = {'rvecs': rvecs, 'tvecs': tvecs}
        else:
            print(f"Failed to calculate extrinsics for Camera {cam_id}.")

        cv2.destroyAllWindows()
        online_phase(img_new, objp, mtx, dist, rvecs, tvecs)
    return extrinsic_parameters

def create_background_model_with_GMM(video_path):
    """
    Creates a background model using Gaussian Mixture Models (GMM) from a video.

    Parameters:
        video_path: The path to the video file from which the background model is created.

    Returns:
        The background model as an image, where each pixel represents the background
        probability.
    """
    cap = cv2.VideoCapture(video_path)
    # Create the background subtractor object
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fgMask = backSub.apply(frame)

    background_model = backSub.getBackgroundImage()
    cap.release()
    return background_model

def subtraction(video_path, background_model_hsv, ground_image):
    """
    Subtracts the background from video frames using a pre-calculated background model in HSV.

    Parameters:
        video_path: Path to the input video file.
        background_model_hsv: The background model in HSV color space.
        ground_image: An image used for optimizing foreground segmentation thresholds.

    Returns:
        A tuple containing the final foreground mask and the last color frame processed.
    """
    cap = cv2.VideoCapture(video_path)
    frame_diffs = []
    colour_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV and calculate the absolute difference
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        diff = cv2.absdiff(frame_hsv, background_model_hsv)
        frame_diffs.append(diff)

    diffs = np.stack(frame_diffs, axis=0)
    th_hue, th_sat, th_val = optimize_thresholds(diff, ground_image)
    print(f"Optimal thresholds for video {video_path} are: " ,th_hue, th_sat, th_val)
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
        colour_frame = frame

    cap.release()
    return foreground_mask, colour_frame

def optimize_thresholds(video_frame, ground_truth):
    """
        Optimizes the thresholds for hue, saturation, and value to minimize the discrepancy
        between the generated mask and the ground truth.

        Parameters:
            video_frame: The video frame (in HSV color space) to be thresholded.
            ground_truth: The ground truth image to compare against the generated mask.

        Returns:
            A tuple containing the optimal thresholds for hue, saturation, and value.
    """

    optimal_thresholds = None
    minimal_discrepancies = float('inf')

    for th_hue in range(0, 30, 1):
        for th_sat in range(0, 30, 1):
            for th_val in range(0, 30, 1):
                # Generate the mask based on current thresholds
                mask = generate_mask(video_frame, int(th_hue), int(th_sat), int(th_val))
                # Apply morphological operations to reduce noise
                mask_cleaned = apply_morphological_ops(mask)
                # Evaluate the mask against the ground truth
                discrepancies = evaluate_segmentation(mask_cleaned, ground_truth)

                if discrepancies < minimal_discrepancies:
                    minimal_discrepancies = discrepancies
                    optimal_thresholds = (th_hue, th_sat, th_val)
    # print(optimal_thresholds)
    return optimal_thresholds


def evaluate_segmentation(mask, ground_truth_mask):
    """
    Evaluates the segmentation quality by comparing a mask against a ground truth mask.

    Parameters:
        mask: The segmentation mask generated from the image processing.
        ground_truth_mask: The ground truth mask for comparison.

    Returns:
        The number of discrepant pixels between the generated mask and the ground truth,
        indicating the level of segmentation error.
    """
    ground_truth_mask = np.array(ground_truth_mask)
    ground_truth_mask = cv2.cvtColor(ground_truth_mask, cv2.COLOR_BGR2GRAY)
    # Calculate the XOR to find differences
    xor_result = cv2.bitwise_xor(mask, ground_truth_mask)
    # Count the non-zero pixels in the XOR result
    discrepancies = np.count_nonzero(xor_result)
    return discrepancies


def apply_morphological_ops(foreground_mask):
    """
        Cleans up a foreground mask using morphological operations.

        Parameters:
            foreground_mask: The binary mask representing detected foreground objects.

        Returns:
            The cleaned binary mask after applying the morphological operations.
    """
    # Apply morphological operations to clean up the mask
    kernel = np.ones((2, 2), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((7, 7), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 3), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    return foreground_mask


def generate_mask(diff, th_hue, th_sat, th_val):
    """
       Generates a foreground mask by applying individual thresholds to the hue, saturation, and value channels.

       Parameters:
           diff: The difference image in HSV color space.
           th_hue: The threshold value for the hue channel.
           th_sat: The threshold value for the saturation channel.
           th_val: The threshold value for the value channel.

       Returns:
           A binary mask representing the detected foreground areas.
       """
    # Apply thresholding
    _, thresh_hue = cv2.threshold(diff[:, :, 0], th_hue, 255, cv2.THRESH_BINARY)
    _, thresh_saturation = cv2.threshold(diff[:, :, 1], th_sat, 255, cv2.THRESH_BINARY)
    _, thresh_value = cv2.threshold(diff[:, :, 2], th_val, 255, cv2.THRESH_BINARY)

    # Combine the thresholds to get the final mask
    foreground_mask = cv2.bitwise_and(thresh_hue, thresh_saturation)
    foreground_mask = cv2.bitwise_and(foreground_mask, thresh_value)
    return foreground_mask


def parse_camera_config_from_file(config_path):
    """
        Parses camera configuration parameters from an XML file.

        Parameters:
            config_path: The file path to the XML configuration file.

        Returns:
            A tuple containing:
            - The camera matrix as a numpy array.
            - The distortion coefficients as a numpy array.
            - The rotation vector as a numpy array.
            - The translation vector as a numpy array.
    """
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
    """
     Projects 3D points onto a 2D image plane using camera parameters.

     Parameters:
         points_3d: An array of 3D points to be projected.
         camera_matrix: The camera matrix (intrinsic parameters).
         dist_coeffs: The distortion coefficients of the camera.
         rvec: The rotation vector of the camera.
         tvec: The translation vector of the camera.

     Returns:
         An array of 2D points representing the projection of the 3D points onto the image plane.
     """
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)