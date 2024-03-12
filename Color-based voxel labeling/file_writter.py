
# from assignment import *
# from engine.config import config
# def save_voxel_data(voxel_list, curr_time):
#     with open("voxel_data.txt", "a") as file:
#         file.write(f"Frame: {curr_time}, Voxels: {len(voxel_list)}\n")
#         for voxel in voxel_list:
#             file.write(f"{voxel}\n")
#
# # Example usage in a loop
# for curr_time in range(0, 2700,5):
#
#     voxel_list, _ = set_voxel_positions(config['world_width'], config['world_height'], config['world_width'],
#                                                     curr_time)
#     save_voxel_data(voxel_list, curr_time)

def load_voxel_data(file_path):
    voxel_data = {}
    current_frame = None
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Frame:"):
                # Extract the frame number from the line
                frame_info = line.split()
                current_frame = int(frame_info[1].rstrip(','))  # Assuming the line format is "Frame: <number>,"
                voxel_data[current_frame] = []
            else:
                # Assuming each line contains voxel coordinates in the format: [x, y, z]
                voxel = eval(line)  # Convert the string representation of the list into an actual list
                voxel_data[current_frame].append(voxel)
    return voxel_data

# Use the function to load voxel data
voxel_data = load_voxel_data("voxel_data.txt")

# Example of how to access voxel data for a specific frame
specific_frame_number = 10  # Example frame number
if specific_frame_number in voxel_data:
    specific_frame_voxels = voxel_data[specific_frame_number]
print(specific_frame_voxels)