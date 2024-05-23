import open3d as o3d
import numpy as np

input_filename = "./44.txt"
output_filename = "./44.ply"

# load
points = []
colors = []
with open(input_filename, 'r') as file:
    for line in file:
        parts = line.split()
        x, y, z = map(float, parts[:3])  # read coordinate
        r, g, b = map(int, parts[3:])   # read RGB
        points.append([x, y, z])
        # normalize RGB
        colors.append([r / 255.0, g / 255.0, b / 255.0])

# create point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))
pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

# save
o3d.io.write_point_cloud(output_filename, pcd)

print("Point cloud with colors saved to", output_filename)
