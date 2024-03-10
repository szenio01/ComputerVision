#new version
import glm
import random
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# global variables
block_size = 1.0
voxel_size = 60.0  # voxel every 3cm
lookup_table = []
camera_handles = []
background_models = []
shift = 20


# generate the floor grid locations
def generate_grid(width, depth):
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


# determines which voxels should be set
def set_voxel_positions(width, height, depth, curr_time):
    if len(lookup_table) == 0:
        create_lookup_table(width, height, depth)

    # initialize voxel list
    voxel_list = []
    print(curr_time)
    # swap y and z
    voxel_grid = np.ones((width, depth, height), np.float32)

    for i_camera in range(4):
        path_name = './data/cam' + str(i_camera + 1)

        if curr_time == 0:
            # train MOG2 on background video, remove shadows, default learning rate
            background_models.append(cv.createBackgroundSubtractorMOG2())
            background_models[i_camera].setShadowValue(0)

            # open background.avi
            camera_handle = cv.VideoCapture(path_name + '/background.avi')
            num_frames = int(camera_handle.get(cv.CAP_PROP_FRAME_COUNT))

            # train background model on each frame
            for i_frame in range(num_frames):
                ret, image = camera_handle.read()
                if ret:
                    background_models[i_camera].apply(image)

            # close background.avi
            camera_handle.release()

            # open video.avi
            camera_handles.append(cv.VideoCapture(path_name + '/video.avi'))
            num_frames = int(camera_handles[i_camera].get(cv.CAP_PROP_FRAME_COUNT))

        # read frame
        cap = camera_handles[i_camera]
        cap.set(cv.CAP_PROP_POS_FRAMES, curr_time)
        ret, image = cap.read()
        # if i_camera == 0:
        #     cv.imshow("Image", image)
        #     cv.waitKey(0)
        # determine foreground
        foreground_image = background_subtraction(image, background_models[i_camera])

        # set voxel to off if it is not visible in the camera, or is not in the foreground
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    if not voxel_grid[x, z, y]:
                        continue
                    voxel_index = z + y * depth + x * (depth * height)
                    # print(lookup_table[i_camera][voxel_index][0][0])
                    if not np.isinf(lookup_table[i_camera][voxel_index][0][0]):
                        projection_x = int(lookup_table[i_camera][voxel_index][0][0])
                    else:
                        projection_x = 1e6

                    if not np.isinf(lookup_table[i_camera][voxel_index][0][1]):
                        projection_y = int(lookup_table[i_camera][voxel_index][0][1])
                    else:
                        projection_y = 1e6
                    if projection_x < 0 or projection_y < 0 or projection_x >= foreground_image.shape[
                        1] or projection_y >= foreground_image.shape[0] or not foreground_image[
                        projection_y, projection_x]:
                        voxel_grid[x, z, y] = 0.0
    colors = []
    voxels_xz = []
    # put voxels that are on in list
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if voxel_grid[x, z, y] > 0:
                    voxel_list.append([x * block_size - width / 2 - shift, y * block_size, z * block_size - depth / 2])
                    voxels_xz.append([x * block_size - width / 2 - shift, z * block_size - depth / 2])
                    colors.append([x / width, z / depth, y / height])
    return voxel_list, colors


# create lookup table
def create_lookup_table(width, height, depth):
    # create 3d voxel grid
    voxel_space_3d = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                voxel_space_3d.append(
                    [voxel_size * (x * block_size - width / 2), voxel_size * (z * block_size - depth / 2),
                     - voxel_size * (y * block_size)])

    for i_camera in range(4):
        camera_path = './data/cam' + str(i_camera + 1)

        # use config.xml to read camera calibration
        file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)
        mtx = file_handle.getNode('CameraMatrix').mat()
        dist = file_handle.getNode('DistortionCoeffs').mat()
        rvec = file_handle.getNode('Rotation').mat()
        tvec = file_handle.getNode('Translation').mat()
        file_handle.release()

        # project voxel 3d points to 2d in each camera
        voxel_space_2d, jac = cv.projectPoints(np.array(voxel_space_3d, np.float32), rvec, tvec, mtx, dist)
        lookup_table.append(voxel_space_2d)


# applies background subtraction to obtain foreground mask
def background_subtraction(image, background_model):
    foreground_image = background_model.apply(image, learningRate=0)

    # remove noise through dilation and erosion
    erosion_elt = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dilation_elt = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    foreground_image = cv.dilate(foreground_image, dilation_elt)
    foreground_image = cv.erode(foreground_image, erosion_elt)

    return foreground_image


# Gets stored camera positions
def get_cam_positions():
    cam_positions = []

    for i_camera in range(4):
        camera_path = './data/cam' + str(i_camera + 1)
        file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)
        tvec = file_handle.getNode('Translation').mat()
        rvec = file_handle.getNode('Rotation').mat()
        file_handle.release()
        # obtain positions
        rotation_matrix = cv.Rodrigues(rvec)[0]
        positions = -np.matrix(rotation_matrix).T * np.matrix(tvec / voxel_size)
        cam_positions.append([positions[0][0] - shift, -positions[2][0], positions[1][0]])
    return cam_positions, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


# Gets stored camera rotations
def get_cam_rotation_matrices():
    cam_rotations = []

    for i in range(4):
        camera_path = './data/cam' + str(i + 1)
        file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)
        rvec = file_handle.getNode('Rotation').mat()
        file_handle.release()

        # # normalize rotations
        angle = np.linalg.norm(rvec)
        axis = rvec / angle

        # apply rotation to compensate for difference between OpenCV and OpenGL
        transform = glm.rotate(-0.5 * np.pi, [0, 0, 1]) * glm.rotate(-angle,
                                                                     glm.vec3(axis[0][0], axis[1][0], axis[2][0]))
        transform_to = glm.rotate(0.5 * np.pi, [1, 0, 0])
        transform_from = glm.rotate(-0.5 * np.pi, [1, 0, 0])
        cam_rotations.append(transform_to * transform * transform_from)
    return cam_rotations


def cluster_voxel_positions(voxel_locations):
    """
    Clusters voxel positions based on their (x, z) coordinates, ignoring the y coordinate.

    Parameters:
    - voxel_locations: A list of lists, where each inner list represents the (x, y, z) coordinates of a voxel.

    Returns:
    - labels: An array of labels indicating the cluster each voxel belongs to.
    - centers: The coordinates of the cluster centers.
    """
    # Convert the list of voxel locations into a NumPy array
    voxel_array = np.array(voxel_locations)

    # Extract the (x, z) coordinates, ignoring the y coordinate
    voxel_positions_xz = np.float32(voxel_array[:, [0, 2]])

    # Number of clusters (persons)
    K = 4

    # Define criteria = (type, max_iter, epsilon)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Number of attempts, to avoid local minima
    attempts = 10

    # Apply K-means clustering
    _, labels, centers = cv.kmeans(voxel_positions_xz, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS)

    return labels, centers, voxel_positions_xz


def plot_clusters(voxel_positions, labels, centers):
    """
    Plots the clustered voxel positions.

    Parameters:
    - voxel_positions: The (x, z) positions of the voxels.
    - labels: The cluster labels for each voxel.
    - centers: The coordinates of the cluster centers.
    """
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        # Select voxels belonging to the current cluster
        cluster_positions = voxel_positions[labels.flatten() == label]
        plt.scatter(cluster_positions[:, 0], cluster_positions[:, 1], label=f'Cluster {label}')

    # Plot the cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], s=100, color='black', marker='x', label='Centers')
    plt.title('Clustered Voxel Positions')
    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    plt.legend()
    plt.grid(True)
    plt.show()


def check_cluster_separation(centers, threshold=10.0):
    """
    Checks if any of the cluster centers are closer to each other than a specified threshold.

    Parameters:
    - centers: The coordinates of the cluster centers.
    - threshold: The minimum acceptable distance between any two centers.

    Returns:
    - Boolean indicating if any centers are too close.
    """
    from scipy.spatial.distance import pdist, squareform

    # Calculate pairwise distances between centers
    distances = squareform(pdist(centers))

    # Fill diagonal with a high value to ignore self-distance
    np.fill_diagonal(distances, np.inf)

    # Check if any distance is below the threshold
    too_close = np.any(distances < threshold)

    return too_close


def project_voxels_to_image(voxel_positions, camera_idx):
    # Use the camera calibration parameters for the selected camera
    camera_path = './data/cam' + str(camera_idx + 1)
    file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)
    mtx = file_handle.getNode('CameraMatrix').mat()
    dist = file_handle.getNode('DistortionCoeffs').mat()
    rvec = file_handle.getNode('Rotation').mat()
    tvec = file_handle.getNode('Translation').mat()
    file_handle.release()

    # Project the voxel positions to 2D
    voxel_positions_3d = np.array(voxel_positions, np.float32)
    projected_points, _ = cv.projectPoints(voxel_positions_3d, rvec, tvec, mtx, dist)
    return projected_points.reshape(-1, 2)  # Reshape for convenience


def create_color_models(image, projected_points, labels, K):
    color_models = []
    for k in range(K):
        # from the projected 2d points take the points that are for the k person (label = k)
        # Each point in projected_points coresponds to the label (same position)
        # cluster_points has the 2d points that have e.g. label = 0
        # print(projected_points.shape)
        # print(labels.shape)
        cluster_points = projected_points[labels.flatten() == k]

        # We get the colors of the image based on a specific cluster point image.shape = (height, width) = (y,x)
        colors = np.array([image[int(pt[1]), int(pt[0])] for pt in cluster_points if
                           0 <= int(pt[0]) < image.shape[1] and 0 <= int(pt[1]) < image.shape[0]])

        if colors.size > 0:
            # Reshape colors array to a proper format: a set of 1D images (vectors) for each color channel
            colors = colors.reshape(-1, 3)
            # Calculate histogram for each color channel
            hist = [cv.calcHist([colors[:, i]], [0], None, [256], [0, 256]) for i in range(3)]
            color_models.append(hist)
        else:
            color_models.append(None)
    return color_models


def process_frame_for_color_models(frame_number, voxel_positions, labels, K, camera_indices=[0]):
    """
    Processes a specific frame to create color models for each cluster (person) using selected camera(s).

    Parameters:
    - frame_number: The number of the frame to process.
    - voxel_positions: A list of 3D voxel positions.
    - labels: Cluster labels for each voxel.
    - K: The number of clusters (people).
    - camera_indices: List of indices of cameras to use.

    Returns:
    - color_models: A list of color models for each cluster.
    """
    color_models = [None] * K  # Initialize color models list

    for camera_idx in camera_indices:
        # Load the frame image from the selected camera
        camera_path = f'./data/cam{camera_idx + 1}/video.avi'
        cap = cv.VideoCapture(camera_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, image = cap.read()
        cap.release()
        if not ret:
            print(f"Failed to read frame {frame_number} from camera {camera_idx + 1}")

            continue
        # cv.imshow('h', image)
        # cv.waitKey(0)
        # Project voxel positions to the 2D image plane
        projected_points = project_voxels_to_image(voxel_positions, camera_idx)

        # Create color models for this frame and camera
        frame_color_models = create_color_models(image, projected_points, labels, K)

        # Update the overall color models with the frame's models
        for k in range(K):
            if frame_color_models[k] is not None:
                if color_models[k] is None:
                    color_models[k] = frame_color_models[k]
                else:
                    print('Empty Color model')
                    # Combine with existing model, e.g., by averaging histograms
                    color_models[k] = (color_models[k] + frame_color_models[k]) / 2

    return color_models


def visualize_color_models(color_models):
    num_clusters = len(color_models)

    for cluster_index, hist in enumerate(color_models):
        if hist is not None:
            plt.figure(figsize=(10, 4))
            plt.suptitle(f'Color Model for Cluster {cluster_index}', fontsize=16)

            for channel_index, color_channel in enumerate(['Red', 'Green', 'Blue']):
                plt.subplot(1, 3, channel_index + 1)
                plt.plot(hist[channel_index], color=color_channel)
                plt.title(f'{color_channel} Channel Histogram')
                plt.xlabel('Intensity')
                plt.ylabel('Frequency')

            plt.tight_layout()
            plt.show()
        else:
            print(f'No color model available for Cluster {cluster_index}')
