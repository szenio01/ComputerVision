Assignment 1 Instructions:

1)Insert the data containing the videos in the main directory

2)Run main.py as it is

3)Calibration is already done and camera properties are saved in directory
parameters. If you want to calibrate the cameras again, change DEBUG = True
in main.py

4)Voxels are saved in parameters/voxels.txt in the format of x,y,z,r,g,b

5)To use them for visualisation in assignment.py of https://github.com/dmetehan/Computer-Vision-3D-Reconstruction , edit the function set_voxel_postions to:
def set_voxel_positions(width, height):
    data, colors = [], []
    with open("../parameters/voxels.txt", "r") as file:
        for line in file:
            # Read voxel position and color from each line
            x, z, y, b, g, r = map(float, line.split())  # Now includes r, g, b

            # Adjust voxel positions
            x = -(x * block_size)
            y = y * block_size
            z = z * block_size

            data.append([x, y, z])  # Add the voxel position

            # Normalize the RGB values to 0-1 range if necessary
            # Assuming RGB values are in 0-255 range, otherwise adjust accordingly
            color_r = r / 255.0
            color_g = g / 255.0
            color_b = b / 255.0
            colors.append([color_r, color_g, color_b])  # Use the actual color from the file

    return data, colors
	
