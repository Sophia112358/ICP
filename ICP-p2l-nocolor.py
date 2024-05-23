import numpy as np
import matplotlib.pyplot as plt

class KDTree:
    def __init__(self, points, depth=0):
        if len(points) > 0:
            k = points.shape[1]
            axis = depth % k
            sorted_points = points[points[:, axis].argsort()]
            self.point = sorted_points[len(sorted_points) // 2]
            self.left = KDTree(sorted_points[:len(sorted_points) // 2], depth + 1)
            self.right = KDTree(sorted_points[len(sorted_points) // 2 + 1:], depth + 1)
        else:
            self.point = None
            self.left = None
            self.right = None

    def query(self, point, depth=0, best=None):
        if self.point is None:
            return best

        k = len(point)
        axis = depth % k

        next_branch = None
        opposite_branch = None

        if point[axis] < self.point[axis]:
            next_branch = self.left
            opposite_branch = self.right
        else:
            next_branch = self.right
            opposite_branch = self.left

        if best is None:
            best = (self.point, np.sum((point - self.point) ** 2))

        best = self.closer_point(point, best, (self.point, np.sum((point - self.point) ** 2)))
        if next_branch is not None:
            best = next_branch.query(point, depth + 1, best)

        if opposite_branch is not None:
            if (point[axis] - self.point[axis]) ** 2 < best[1]:
                best = opposite_branch.query(point, depth + 1, best)

        return best


    def closer_point(self, point, p1, p2):
        if p1 is None:
            return p2
        if p2 is None:  
            return p1
        if np.sum((point - p1[0]) ** 2) < p2[1]:  
            return p1
        return p2


def nearest_neighbor_kd(src, dst):
    tree = KDTree(dst)
    indices = []
    distances = []
    for s in src:
        nearest_point, distance = tree.query(s, best=None) 
        idx = np.where((dst == nearest_point).all(axis=1))[0][0]
        indices.append(idx)
        distances.append(distance)
    return np.array(indices), np.array(distances)

# load data from txt file
def load_point_cloud(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    return points

# define downsampling
def voxel_down_sample(points, voxel_size):
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    dims = ((max_bound - min_bound) / voxel_size).astype(int) + 1
    voxel_indices = ((points - min_bound) / voxel_size).astype(int)
    voxel_keys = np.ravel_multi_index(voxel_indices.T, dims)
    unique_keys, unique_indices = np.unique(voxel_keys, return_index=True)
    downsampled_points = points[unique_indices]
    return downsampled_points

# estimate normals
def estimate_normals(points, max_nn=30):
    normals = np.zeros_like(points)
    for i, point in enumerate(points):
        diff = points - point
        dist = np.linalg.norm(diff, axis=1)
        neighbors = points[np.argsort(dist)[:max_nn]]
        if len(neighbors) < 3:
            continue
        cov_matrix = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        normals[i] = eigvecs[:, np.argmin(eigvals)]
    return normals

# calculate RMSE
def cal_rmse(source, target):
    distances = np.linalg.norm(source - target, axis=1)
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse

# convert axis-angle to rotation matrix
def axis_angle_to_matrix(axis, theta):
    w = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    rot = np.identity(3) + np.sin(theta) * w + (1 - np.cos(theta)) * np.dot(w, w)
    return rot

# load data
pcd1 = load_point_cloud('./data/16.txt')
pcd2 = load_point_cloud('./data/44pitch.txt')

# downsampling
pcd_s = voxel_down_sample(pcd1, voxel_size=0.1)
pcd_t = voxel_down_sample(pcd2, voxel_size=0.1)

# ensure both point cloud share same amount of points after downsampling
def match_point_cloud_size(src, dst):
    min_size = min(len(src), len(dst))
    return src[:min_size], dst[:min_size]

pcd_s, pcd_t = match_point_cloud_size(pcd_s, pcd_t)

# show initial status
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pcd_s[:, 0], pcd_s[:, 1], pcd_s[:, 2], c='g', marker='o', label='Source (16)')
ax.scatter(pcd_t[:, 0], pcd_t[:, 1], pcd_t[:, 2], c='b', marker='^', label='Target (44)')
ax.legend(loc="upper right")
plt.title('Initial Point Clouds')
plt.show()

# estimate normals
normals_t = estimate_normals(pcd_t)

# find nearest neighbors
indices, distances = nearest_neighbor_kd(pcd_s, pcd_t)
np_pcd_y = pcd_t[indices].copy()
np_normal_y = normals_t[indices].copy()

# matrix A
A = np.zeros((6, 6))
for i in range(len(pcd_s)):
    xn = np.cross(pcd_s[i], np_normal_y[i])
    xn_n = np.hstack((xn, np_normal_y[i])).reshape(-1, 1)
    A += np.dot(xn_n, xn_n.T)

# vector b
b = np.zeros((6, 1))
for i in range(len(pcd_s)):
    xn = np.cross(pcd_s[i], np_normal_y[i])
    xn_n = np.hstack((xn, np_normal_y[i])).reshape(-1, 1)
    nT = np_normal_y[i].reshape(1, -1)
    p_x = (np_pcd_y[i] - pcd_s[i]).reshape(-1, 1)
    b += xn_n * np.dot(nT, p_x)

# compute rotation axis w and rotation angle theta
u_opt = np.dot(np.linalg.inv(A), b)
theta = np.linalg.norm(u_opt[:3])
w = (u_opt[:3] / theta).reshape(-1)

# compute rotation matrix
rot = axis_angle_to_matrix(w, theta)

# construct 4x4 transformation matrix
transform = np.identity(4)
transform[0:3, 0:3] = rot.copy()
transform[0:3, 3] = u_opt[3:6].reshape(-1).copy()

# apply transformation
pcd_s_transformed = np.dot(pcd_s, transform[0:3, 0:3].T) + transform[0:3, 3]

# find nearest neighbors for transformed source
indices, distances = nearest_neighbor_kd(pcd_s_transformed, pcd_t)
nearest_target_points = pcd_t[indices]

# calculate RMSE and print
rmse = cal_rmse(pcd_s_transformed, nearest_target_points)
print(f"ICP Point-to-Plane RMSE: {rmse}")

# calculate mean error
mean_error = np.mean(distances)
print(f"Mean error: {mean_error}")

# visualize the final status
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pcd_t[:, 0], pcd_t[:, 1], pcd_t[:, 2], c='b', marker='^', label='Target (44)')
#ax.scatter(pcd_s[:, 0], pcd_s[:, 1], pcd_s[:, 2], c='g', marker='o', label='Source (16)')
ax.scatter(pcd_s_transformed[:, 0], pcd_s_transformed[:, 1], pcd_s_transformed[:, 2], c='r', marker='o', label='Transformed Source')
ax.legend(loc="upper right")
plt.title('Point Clouds After ICP')
plt.show()