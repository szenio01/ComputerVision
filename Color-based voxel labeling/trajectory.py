import cv2
import matplotlib.pyplot as plt
from assignment import *
from engine.config import config

# Global dictionary to store the positions of each person over time
# Assuming we have 4 individuals to track
trajectories = {0: [], 1: [], 2: [], 3: []}
total_frames = 2700


def simulate(curr_time):
    while curr_time < 100:
        if curr_time == 0:
            print("OFFLINE PHASE")
            positions, colors = set_voxel_positions(config['world_width'], config['world_height'],
                                                    config['world_width'],
                                                    0)

            # k means and labels for each voxel
            labels, centers, voxel_positions_xz = cluster_voxel_positions(positions)
            # plot_clusters(voxel_positions_xz, labels, centers)

            # Filter positions based on the height to remove the trousers that don't have distinct colors
            filtered_positions = [pos for pos in positions if pos[1] > 12 and pos[1] < 25]
            filtered_colors = [[1, 0, 0] for pos in positions if pos[1] > 12 and pos[1] < 25]
            filtered_indices = [i for i, pos in enumerate(positions) if pos[1] > 12 and pos[1] < 25]
            filtered_labels = [labels[i] for i in filtered_indices]
            filtered_labels = np.array(filtered_labels)

            color_models_offline = process_frame_for_color_models_GMM(frame_number=0,
                                                                      voxel_positions=filtered_positions,
                                                                      labels=filtered_labels, K=4,
                                                                      camera_indices=[0])

            # visualize_color_models(color_models_offline)
            # plot_color_models_with_image(color_models_offline)
            new_colors = []
            for i in labels:
                if i[0] == 0:
                    new_colors.append([1, 0, 0])
                elif i[0] == 1:
                    new_colors.append([0, 1, 0])
                elif i[0] == 2:
                    new_colors.append([0, 0, 1])
                elif i[0] == 3:
                    new_colors.append([0, 1, 1])

            # Check if clusters are close (stuck in a local minimum)
            check_cluster_separation(centers, threshold=10.0)
            too_close = check_cluster_separation(centers)
            if too_close:
                print("Some clusters are too close to each other.")
            else:
                print("Clusters are adequately separated.")

            curr_time += 5
        else:
            print("ONLINE PHASE")
            # Process the current frame to re-cluster and generate new color models
            positions, _ = set_voxel_positions(config['world_width'], config['world_height'], config['world_width'],
                                               curr_time)
            labels, centers, voxel_positions_xz = cluster_voxel_positions(positions)

            # Filter positions based on height, similar to offline phase for consistency
            filtered_positions = [pos for pos in positions if pos[1] > 12 and pos[1] < 25]
            filtered_colors = [[1, 0, 0] for pos in positions if pos[1] > 12 and pos[1] < 25]
            filtered_indices = [i for i, pos in enumerate(positions) if pos[1] > 12 and pos[1] < 25]
            filtered_labels = [labels[i] for i in filtered_indices]
            filtered_labels = np.array(filtered_labels)
            filtered_positions, filtered_labels = filter_ghost_voxels(filtered_positions, filtered_labels )

            color_models_online = process_frame_for_color_models_GMM(frame_number=curr_time,
                                                                     voxel_positions=filtered_positions,
                                                                     labels=filtered_labels, K=4, camera_indices=[0])

            # Now, match the newly created online color models to the offline ones
            # first try
            # matches = match_online_to_offline(color_models_online, color_models_offline)
            # print(matches)
            # Second try
            try:
                matches = calculate_all_distancesGMM(color_models_online, color_models_offline)
                print("online: offline - ", matches)
            except ValueError as e:
                print(f"Error processing frame {curr_time}: {e}")

            new_colors = []
            labels_new = []
            for i in labels:
                if i[0] == 0:
                    labels_new.append([matches[0]])
                if i[0] == 1:
                    labels_new.append([matches[1]])
                if i[0] == 2:
                    labels_new.append([matches[2]])
                if i[0] == 3:
                    labels_new.append([matches[3]])

            for i in labels_new:
                if i[0] == 0:
                    new_colors.append([1, 0, 0])
                elif i[0] == 1:
                    new_colors.append([0, 1, 0])
                elif i[0] == 2:
                    new_colors.append([0, 0, 1])
                elif i[0] == 3:
                    new_colors.append([0, 1, 1])

            # Check if clusters are close (stuck in a local minimum)
            check_cluster_separation(centers, threshold=10.0)
            too_close = check_cluster_separation(centers)
            if too_close:
                print("Some clusters are too close to each other.")
            else:
                print("Clusters are adequately separated.")

            curr_time += 5
            if matches is not None:
                for online_id, offline_id in matches.items():
                    # Get the 2D center for the matched cluster (assuming centers are 2D)
                    new_center_2d = centers[online_id]
                    # Append the center position to the trajectory for the corresponding person
                    trajectories[offline_id].append(new_center_2d)
            else:
                # Handle the case where matches is None
                print("Warning: Matching failed for this frame. Skipping.")


def simulate_previews_position(curr_time):
    while curr_time<100:
        if curr_time == 0:
            print("OFFLINE PHASE")
            positions, colors = set_voxel_positions(config['world_width'], config['world_height'],
                                                    config['world_width'],
                                                    curr_time)

            # k means and labels for each voxel
            labels_of, centers_of, voxel_positions_xz_of = cluster_voxel_positions(positions)

            new_colors = []
            for i in labels_of:
                if i[0] == 0:
                    new_colors.append([1, 0, 0])
                elif i[0] == 1:
                    new_colors.append([0, 1, 0])
                elif i[0] == 2:
                    new_colors.append([0, 0, 1])
                elif i[0] == 3:
                    new_colors.append([0, 1, 1])

            curr_time += 5

        else:

            print("ONLINE PHASE")
            # Process the current frame to re-cluster and generate new color models
            positions, _ = set_voxel_positions(config['world_width'], config['world_height'], config['world_width'],
                                               curr_time)
            labels, centers, voxel_positions_xz = cluster_voxel_positions(positions)

            matches = match_clusters_by_proximity(centers_of, centers)
            print("centers_of", centers_of)
            print("centers new", centers)
            print(matches)
            new_colors = []
            labels_new = []
            centers_new = []
            for i in labels:
                if i[0] == 0:
                    labels_new.append([matches[0]])
                if i[0] == 1:
                    labels_new.append([matches[1]])
                if i[0] == 2:
                    labels_new.append([matches[2]])
                if i[0] == 3:
                    labels_new.append([matches[3]])

            for i, c in enumerate(centers):
                centers_new.append(centers[get_key_by_value(matches, i)])

            for i in labels_new:
                if i[0] == 0:
                    new_colors.append([1, 0, 0])
                elif i[0] == 1:
                    new_colors.append([0, 1, 0])
                elif i[0] == 2:
                    new_colors.append([0, 0, 1])
                elif i[0] == 3:
                    new_colors.append([0, 1, 1])

            # Check if clusters are close (stuck in a local minimum)
            check_cluster_separation(centers, threshold=10.0)
            too_close = check_cluster_separation(centers)
            if too_close:
                print("Some clusters are too close to each other.")
            else:
                print("Clusters are adequately separated.")
            labels_of = labels_new
            centers_of = centers_new
            voxel_positions_xz_of = voxel_positions_xz
            curr_time += 5
            if matches is not None:
                for online_id, offline_id in matches.items():
                    # Get the 2D center for the matched cluster (assuming centers are 2D)
                    new_center_2d = centers[online_id]
                    # Append the center position to the trajectory for the corresponding person
                    trajectories[offline_id].append(new_center_2d)
            else:
                # Handle the case where matches is None
                print("Warning: Matching failed for this frame. Skipping.")

# Open the video file

simulate_previews_position(0)
plt.figure(figsize=(10, 8))  # Optional: Specify the size of the plot
for person_id, trajectory in trajectories.items():
    x_positions = [pos[0] for pos in trajectory]
    y_positions = [pos[1] for pos in trajectory]
    plt.scatter(x_positions, y_positions, marker='o', label=f'Person {person_id}')

plt.legend()
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('2D positions of each person over time')
plt.grid(True)
plt.show()
