import numpy as np

# 读取点云数据
def load_point_cloud(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6]
    return points, colors

# 保存点云数据
def save_point_cloud(file_path, points, colors):
    data = np.hstack((points, colors))
    np.savetxt(file_path, data, fmt='%.8f')

# 创建绕 Y 轴旋转的旋转矩阵
def get_rotation_matrix_y(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle, 0, sin_angle],
        [0, 1, 0],
        [-sin_angle, 0, cos_angle]
    ])
    return rotation_matrix

# 应用旋转矩阵
def apply_rotation(points, rotation_matrix):
    rotated_points = points @ rotation_matrix.T
    return rotated_points

# 读取数据
points, colors = load_point_cloud('./data/44.txt')

# 旋转 90 度（绕 Y 轴）
angle = np.pi / 2
rotation_matrix = get_rotation_matrix_y(angle)
rotated_points = apply_rotation(points, rotation_matrix)

# 保存旋转后的点云数据
save_point_cloud('./data/44pitch.txt', rotated_points, colors)
