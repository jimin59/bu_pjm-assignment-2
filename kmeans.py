import numpy as np

# Euclidean distance between two points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

# Random initialization
def initialize_random(data, k):
    return data[np.random.choice(data.shape[0], k, replace=False)]

# Farthest First initialization
def initialize_farthest_first(data, k):
    centroids = [data[np.random.randint(len(data))]]
    for _ in range(1, k):
        distances = np.array([min([euclidean_distance(p, c) for c in centroids]) for p in data])
        next_centroid = data[np.argmax(distances)]
        centroids.append(next_centroid)
    return np.array(centroids)

# KMeans++ initialization
def initialize_kmeans_plusplus(data, k):
    centroids = [data[np.random.randint(len(data))]]
    for _ in range(1, k):
        distances = np.array([min([euclidean_distance(p, c)**2 for c in centroids]) for p in data])
        probabilities = distances / np.sum(distances)
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        next_centroid = data[np.where(cumulative_probabilities >= r)[0][0]]
        centroids.append(next_centroid)
    return np.array(centroids)

# KMeans algorithm
def kmeans(data, k, initialize):
    centroids = initialize(data, k)
    for _ in range(100):  # Max iterations
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)
        new_centroids = [np.mean(cluster, axis=0) if cluster else centroids[i] for i, cluster in enumerate(clusters)]
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return np.array(centroids), clusters
