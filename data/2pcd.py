import numpy as np

def write_pcd(filename, points, colors):
    header = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F I
COUNT 1 1 1 1
WIDTH {0}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {0}
DATA ascii
""".format(len(points))

    with open(filename, 'w') as f:
        f.write(header)
        for point, color in zip(points, colors):
            # Convert RGB to a single integer value
            rgb = (color[0] << 16) | (color[1] << 8) | color[2]
            f.write("{:.6f} {:.6f} {:.6f} {}\n".format(point[0], point[1], point[2], rgb))

# Read data
data = np.loadtxt("./44.txt")

# Separate XYZ and RGB
xyz = data[:, :3]
rgb = data[:, 3:].astype(int)

# Write to PCD
write_pcd("./44.pcd", xyz, rgb)

print("PCD file has been created successfully!")
