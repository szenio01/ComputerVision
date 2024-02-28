import time
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

from helper_functions import *

# Define the chess board size and square size
chessboard_size = (6, 8)
corner_points = []
directory = "data"
DEBUG = False

def create_background_models():
    background_models = {}
    for cam_id in range(1, 5):
        video_path = f'data/cam{cam_id}/background.avi'
        background_model = create_background_model_with_GMM(video_path)
        background_models[cam_id] = background_model
        # Optionally save the background model to a file
        cv2.imwrite(f'data/cam{cam_id}/background_model.jpg', background_model)
    return background_models

def background_subtraction():
    forground_masks = []
    coloured_images = []
    for cam_id in range(1, 5):
        background_model_path = f'data/cam{cam_id}/background_model.jpg'
        video_path = f'data/cam{cam_id}/video.avi'
        ground_image = f'parameters/{cam_id}.jpg'
        ground_image = cv2.imread(ground_image)

        # Read the background model and convert it to HSV
        background_model = cv2.imread(background_model_path)
        background_model_hsv = cv2.cvtColor(background_model, cv2.COLOR_BGR2HSV)
        forground_mask, coloured_image = subtraction(video_path, background_model_hsv, ground_image)
        forground_masks.append(forground_mask)
        coloured_images.append(coloured_image)
    return forground_masks, coloured_images


def background_subtraction_parallel():
    def process_camera(cam_id):
        background_model_path = f'data/cam{cam_id}/background_model.jpg'
        video_path = f'data/cam{cam_id}/video.avi'
        ground_image = f'parameters/{cam_id}.jpg'
        ground_image = cv2.imread(ground_image)

        # Read the background model and convert it to HSV
        background_model = cv2.imread(background_model_path)
        background_model_hsv = cv2.cvtColor(background_model, cv2.COLOR_BGR2HSV)
        forground_mask, coloured_image = subtraction(video_path, background_model_hsv, ground_image)

        return forground_mask, coloured_image

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_camera, cam_id) for cam_id in range(1, 5)]

        results = [future.result() for future in futures]

    forground_masks, coloured_images = zip(*results)  # Unzips the results into two lists

    return forground_masks, coloured_images

def create_lut(voxels):
    lut = {}
    for cam_id in range(1, 5):
        config_path = f"parameters/cam{cam_id}/camera_properties.xml"
        camera_matrix, dist_coeffs, rvec, tvec = parse_camera_config_from_file(config_path)

        # Project all voxels to 2D
        points_2d = project_to_2d(voxels, camera_matrix, dist_coeffs, rvec, tvec)
        # Store the 2D points in the Look up Table
        lut[cam_id] = points_2d

    return lut


def create_lut_parallel(voxels):
    def process_camera(cam_id):
        config_path = f"parameters/cam{cam_id}/camera_properties.xml"
        camera_matrix, dist_coeffs, rvec, tvec = parse_camera_config_from_file(config_path)

        # Project all voxels to 2D for this camera
        points_2d = project_to_2d(voxels, camera_matrix, dist_coeffs, rvec, tvec)
        return cam_id, points_2d

    lut = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_camera, cam_id) for cam_id in range(1, 5)]
        for future in futures:
            cam_id, points_2d = future.result()
            # Store the 2D points in the Look up Table
            lut[cam_id] = points_2d

    return lut


def check_voxel_visibility(voxel_index, lut, silhouette_masks, color_images):
    colors = []
    for cam_id, points_2d in lut.items():
        adjusted_cam_id = cam_id - 1  #
        point = points_2d[voxel_index]
        x, y = int(point[0]), int(point[1])

        if 0 <= x < silhouette_masks[adjusted_cam_id].shape[1] and 0 <= y < silhouette_masks[adjusted_cam_id].shape[0]:
            if silhouette_masks[adjusted_cam_id][y, x] == 255:
                # Get the color from the color image if the voxel is visible(white)
                color = color_images[adjusted_cam_id][y, x]
                colors.append(color)
            else:
                # If the voxel wasnt visible in one of the cameras
                return None

    if colors:
        return np.mean(colors, axis=0).astype(int)
    else:
        # If the voxel wasnt visible in every camera
        return None



def check_visibility_and_reconstruct(silhouette_masks, coloured_images):
    # Define the 3D grid
    x_range = np.linspace(-1024, 1024, num=200)
    y_range = np.linspace(-1024, 1024, num=200)
    z_range = np.linspace(0, 2048, num=200)
    voxels = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)

    # Create lookup table
    lookup_table = create_lut_parallel(voxels)

    visible_points = []
    for voxel_index in range(len(voxels)):
        x, y, z = voxels[voxel_index]  # Get voxel coordinates

        colour = check_voxel_visibility(voxel_index, lookup_table, silhouette_masks, coloured_images)
        if colour is not None:
            # Correct Indexing
            ix = int(x / 16)
            iy = int(y / 16)
            iz = int(z / 16)

            visible_points.append([ix, iy, iz, *colour])

    with open("parameters/voxels.txt", "w") as file:
        for point in visible_points:
            file.write(f"{point[0]} {point[1]} {point[2]} {point[3]} {point[4]} {point[5]}\n")
    print("Voxels saved in parameters/voxels.txt")
def main():
    global corner_points
    extrinsic_parameters = {}
    if DEBUG:
        calibration_parameters = calibrate_cameras()
        extrinsic_parameters = calculate_extrinsics(calibration_parameters)
        #
        write_camera_configs('parameters', calibration_parameters, extrinsic_parameters)
        all_camera_configs = read_all_camera_configs('parameters')

    # Task 2. Background subtraction
    # 1. Create a background image
    print("Creating Backround Models: ")
    create_background_models()

    # 2. Create a background image
    print("Creating Foreground Masks: ")
    foreground_masks, coloured_images = background_subtraction_parallel()

    # 3. Correct colouring of images
    print("Colour Correction of images: ")
    coloured_images_corrected, foreground_masks = correct_images(coloured_images, foreground_masks)

    # 4. Task 3: Check visibility and construct voxels
    print("Look up table creation and voxel visibility check:")
    check_visibility_and_reconstruct(foreground_masks, coloured_images_corrected)

if __name__ == "__main__":
    main()
