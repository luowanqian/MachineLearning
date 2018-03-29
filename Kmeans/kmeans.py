import numpy as np


def distance_eclud(a, b):
    diff = a - b

    return np.linalg.norm(diff, ord=2)


def random_centroids(data, k):
    num_points, _ = data.shape
    idx = np.random.choice(num_points, k, replace=False)

    return data[idx, :]


def Kmeans(data, k, dist_measure=distance_eclud,
           create_centroids=random_centroids, max_iterations=100):
    num_points, dim = data.shape
    centroids = create_centroids(data, k)
    cluster_assignment = np.zeros(num_points, dtype=int)
    cluster_changed = False
    iterations = max_iterations

    for it in range(max_iterations):
        cluster_changed = False
        for i in range(num_points):
            min_distance = np.inf
            min_indx = -1
            for j in range(k):
                distance = distance_eclud(data[i, :], centroids[j, :])
                if (distance < min_distance):
                    min_distance = distance
                    min_indx = j
            if (min_indx != cluster_assignment[i]):
                cluster_changed = True
            cluster_assignment[i] = min_indx

        for i in range(k):
            cluster_data = data[cluster_assignment == i, :]
            centroids[i, :] = np.mean(cluster_data, axis=0)

        if (not cluster_changed):
            iterations = it
            break

    return centroids, cluster_assignment, iterations
