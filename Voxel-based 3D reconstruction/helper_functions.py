import cv2
import numpy as np
import xml.etree.ElementTree as ET


def write_camera_configs(directory, calibration_parameters, extrinsic_parameters):
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
    all_camera_configs = {}

    for cam_id in range(1, 5):  # Adjust the range based on the number of cameras
        config_path = f"{directory}/cam{cam_id}/camera_properties.xml"
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