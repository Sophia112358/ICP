import open3d as o3d
import numpy as np

# 文件路径
input_filename = "./44.txt"
output_filename = "./44.ply"

# 读取数据
points = []
colors = []
with open(input_filename, 'r') as file:
    for line in file:
        # 解析每行的XYZ坐标和RGB颜色
        parts = line.split()
        x, y, z = map(float, parts[:3])  # 读取XYZ坐标
        r, g, b = map(int, parts[3:])   # 读取RGB颜色
        points.append([x, y, z])
        # 将RGB颜色从[0, 255]规范化到[0, 1]范围
        colors.append([r / 255.0, g / 255.0, b / 255.0])

# 创建Open3D点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))
pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

# 保存为PLY文件
o3d.io.write_point_cloud(output_filename, pcd)

print("Point cloud with colors saved to", output_filename)
