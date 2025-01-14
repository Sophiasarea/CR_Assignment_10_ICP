import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def centroid(points):
    centroid = np.mean(points, axis=0)
    return centroid

def centered(points, centroid):
    centered = points - centroid
    return centered

# # compute closest Euclidean distances
# def closest_Euclidean_dist(X_centered, P0_centered, P0):
#     corresponding_points = []
#     for x in X_centered:
#         distances = np.linalg.norm(P0_centered - x, axis=1)
#         closest_point_idx = np.argmin(distances)
#         corresponding_points.append(P0[closest_point_idx])
        
def closest_point_to_point(X, P):
    tree = KDTree(P)
    distances, indices = tree.query(X)
    return P[indices]

def closest_point_to_line(X, P):
    result = []
    for x in X:
        distances = np.linalg.norm(P - x, axis=1)
        closest_idx = np.argmin(distances)
        next_idx = (closest_idx + 1) % len(P)
        prev_idx = (closest_idx - 1) % len(P)

        candidates = [
            project_point_to_line(x, P[closest_idx], P[next_idx]), 
            project_point_to_line(x, P[closest_idx], P[prev_idx])
        ]
        closest_candidate = min(candidates, key = lambda p: np.linalg.norm(x - p))
        result.append(closest_candidate)
    return np.array(result)

def project_point_to_line(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    return a + t *ab

# Compute R and T using SVD
def compute_transformation(X_centered, P_centered):
    W = np.dot(P_centered.T, X_centered)
    U, _, Vt = np.linalg.svd(W)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)
    t = np.zeros(X_centered.shape[1])
    return R, t

def apply_transformation(P, R, t):
    return np.dot(P, R.T)+t

def ICP(X, P, iterations=4, distance_metric="point"):
    for i in range(iterations):
        if distance_metric == "point":
            P_closest = closest_point_to_point(X, P)
        elif distance_metric == "line":
            P_closest = closest_point_to_line(X, P)
        R, t = compute_transformation(X, P_closest)
        P = apply_transformation(P, R, t)
        yield P, R, t

def visualize_iterations(X, P, iterations, title= "ICP Iterations"):
    plt.figure(figsize=(8,6))
    for i, (P_transformed, _, _) in enumerate(iterations):
        plt.scatter(X[:,0], X[:,1], c = "red", label = "X(fixed)" if i == 0 else "")
        plt.scatter(P_transformed[:,0], P_transformed[:,1], c = "blue", alpha = 0.6, label = f"P(iter{i+1})")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.show()

def rotation_experiments(X, P, angles):
    errors = []
    for angle in angles:
        theta = np.radians(angle)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        P_rotated = apply_transformation(P, R, np.zeros(2))
        iterations = list(ICP(X, P_rotated, distance_metric="point"))
        error = np.mean(np.linalg.norm(X - iterations[-1][0], axis=1))
        errors.append((angle, error))
    return errors

if __name__ == "__main__":
    X = np.array([[-1.5, 2.5], [0.5, 3.5], [1.5, 4.5], [2.5, 5.5], [3.5, 4.5], [4.5, 3.5], [5.5, 2.5]])
    P0 = np.array([[-0.5, 3], [0.4, 4.1], [1.3, 5.2], [2.2, 6.3], [3.1, 5.4], [4.0, 4.5], [4.9, 3.6]])

    X_centroid = centroid(X)
    X_centered = centered(X, X_centroid)

    P_centroid = centroid(P0)
    P_centered = centered(P0, P_centroid)

    #test
    print("X_centroid:", X_centroid)
    print("P_centroid:", P_centroid)
    print("X_centered:", X_centered)
    print("P_centered:", P_centered)


    # P2P icp
    iterations_point = list(ICP(X_centered, P_centered, distance_metric="point"))
    visualize_iterations(X_centered, P_centered, iterations_point, title="ICP (Point-to-Point)")

    # P2L icp
    iterations_line = list(ICP(X_centered, P_centered, distance_metric="line"))
    visualize_iterations(X_centered, P_centered, iterations_line, title="ICP (Point-to-Line)")

    # Rotation experiments
    angles = range(-90,91,10)
    errors = rotation_experiments(X_centered, P_centered, angles)
    for angle, error in errors:
        print(f"Rotation:{angle}°, Error:{error:.4f}")