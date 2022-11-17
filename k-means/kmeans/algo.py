import numpy as np
from .utils import get_logger


def get_init_centroids(observations, k: int):
    if len(observations) < k:
        raise Exception("Количество наблюдений меньше, чем количество кластеров - k.")

    centroids = []

    logger = get_logger("INIT CENTROIDS")

    first_centroid_index = np.random.randint(0, len(observations))
    centroids.append(observations[first_centroid_index])

    logger.debug(f"Выбрана первая случайная точка: {centroids[0]}")
    while len(centroids) < k:
        sum_of_squared_distances = 0

        logger.debug("Для каждого наблюдения считаем квадрат расстояния до ближайшего центра кластера")
        for index, observation in enumerate(observations):
            closer_centroid, closer_centroid_index, distance = find_closer_centroid(observation, centroids)
            sum_of_squared_distances += distance ** 2

        logger.debug("Выбераем наиболее удаленную точку")
        limit_for_find_next_centroid = np.random.random() * sum_of_squared_distances
        sum_of_squared_distances = 0
        founded_centroid = None
        for index, observation in enumerate(observations):
            closer_centroid, closer_centroid_index, distance = find_closer_centroid(observation, centroids)
            sum_of_squared_distances += distance ** 2
            if sum_of_squared_distances >= limit_for_find_next_centroid:
                founded_centroid = observation
                break

        logger.debug(f"Выбранная точка: {founded_centroid}")
        centroids.append(founded_centroid)

    return np.asarray(centroids)


def find_closer_centroid(observation, centroids):
    if len(centroids) == 1:
        return centroids[0], 0, count_distance_between_points(observation, centroids[0])

    closer_centroid = None
    closer_centroid_index = None
    min_distance = None
    centroid_index = 0
    while centroid_index < len(centroids):
        current_distance = count_distance_between_points(observation, centroids[centroid_index])
        if min_distance == None or min_distance > current_distance:
            min_distance = current_distance
            closer_centroid = centroids[centroid_index]
            closer_centroid_index = centroid_index
        centroid_index += 1
    return closer_centroid, closer_centroid_index, min_distance


def count_distance_between_points(first_point, second_point):
    return np.linalg.norm(first_point - second_point)


def assign_each_observation_to_nearest_centroid(observations, centroids):
    conclusions = []
    average_distances = [0] * len(centroids)

    for index, observation in enumerate(observations):
        centroid, conclusion, distance = find_closer_centroid(observation, centroids)
        average_distances[conclusion] += distance ** 2
        conclusions.append(conclusion)

    conclusions = np.asarray(conclusions)
    inertia = 0
    for conclusion, average_distance in enumerate(average_distances):
        inertia += average_distance

    return conclusions, inertia


def compute_centroids(observations, conclusions):
    centroids = []
    clusters = np.sort(np.unique(conclusions))
    for cluster in clusters:
        members = observations[conclusions == cluster]
        centroids.append(members.mean(axis=0))
    return centroids


def k_means(observations, k: int):
    logger = get_logger("GET CENTROIDS")
    logger.info("Формируем начальное приближение центров всех кластеров.")
    centroids = get_init_centroids(observations=observations, k=k)

    prev_conclusions = np.array([])
    logger.info("Относим каждое наблюдение к ближайшему кластеру")
    curr_conclusions, inertia = assign_each_observation_to_nearest_centroid(observations=observations,
                                                                            centroids=centroids)
    logger.debug(f"Кластеры {np.unique(curr_conclusions, return_counts=True)}")

    while not np.array_equal(prev_conclusions, curr_conclusions):
        logger.info("Пересчитывем центры кластеров")
        centroids = compute_centroids(observations, curr_conclusions)
        prev_conclusions = curr_conclusions
        logger.info("Относим каждое наблюдение к ближайшему кластеру")
        curr_conclusions, inertia = assign_each_observation_to_nearest_centroid(observations=observations,
                                                                                centroids=centroids)
        logger.debug(f"Кластеры {np.unique(curr_conclusions, return_counts=True)}")

    logger.info("Кластеризация завершена.")

    return centroids, curr_conclusions, inertia
