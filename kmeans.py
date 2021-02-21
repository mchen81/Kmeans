import random
import math 
import numpy as np
import sys

class KMeans:
    def __init__(self, k=5, max_iteration=100):
        self.k = k
        self.max_iteration = max_iteration
    
    def __init_centroids(self, input_X):
        centroids = []
        for i in range(self.k):
            r_index = random.randint(0, len(input_X)-1)
            while input_X[r_index] in centroids: #Avoid duplicating centroids
                r_index = random.randint(0, len(input_X)-1)
            centroids.append(input_X[r_index])
        return centroids
    
    def __get_euclidean_dist(self, c1, c2):
        p1 = np.array(c1) 
        p2 = np.array(c2)
        dist = np.linalg.norm(p1 - p2) 
        return dist
    
    def __get_centroid(self, group):
        group_len = len(group)
        centroid = [0]*len(group[0])
        for point in group:
            for i in range(len(point)):
                centroid[i] += point[i]
        for i in range(len(centroid)):
            centroid[i] /= group_len
        return centroid
        
    def __group_input_x(self, input_X, centroids, balanced):
        groups = []
        x_group = []
        
        for i in range(len(centroids)):
            groups.append([])
        for x in input_X:
            dist_to_centroids = [0] * len(centroids)
            for ci in range(len(centroids)):
                dist_to_centroids[ci] = self.__get_euclidean_dist(centroids[ci], x)
            
            min_dist = min(dist_to_centroids)
            min_centroid_i = dist_to_centroids.index(min_dist)
            
            if balanced:
                upper_bound = math.ceil(len(input_X) / len(centroids))
                while len(groups[min_centroid_i]) > upper_bound:
                    dist_to_centroids[min_centroid_i] = sys.maxsize
                    min_dist = min(dist_to_centroids)
                    min_centroid_i = dist_to_centroids.index(min_dist)
                
            groups[min_centroid_i].append(x)
            x_group.append(min_centroid_i)

        return (groups, x_group)
    
    def __update_centroids(self, groups, centroids):
        for i in range(len(centroids)):
            centroids[i] = self.__get_centroid(groups[i])
        return centroids
    
    def fit(self, X):
        centroids = self.__init_centroids(X)
        for i in range(self.max_iteration):
            (groups, x_group) = self.__group_input_x(X, centroids, False)
            centroids = self.__update_centroids(groups, centroids)
        return (x_group, centroids);
    
    def fit_extended(self, X, balanced=False):
        centroids = self.__init_centroids(X)
        for i in range(self.max_iteration):
            (groups, x_group) = self.__group_input_x(X, centroids, balanced)
            centroids = self.__update_centroids(groups, centroids)
        return (x_group, centroids);
    
    
