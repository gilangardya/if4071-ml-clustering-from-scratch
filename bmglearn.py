"""Clustering from scratch  
- K-Means
- K-Medoids dengan PAM
- DBScan
- Agglomerative clustering
"""

# Authors: Gilang Ardyamandala Al Assyifa <gilangardya@gmail.com>
#          Bobby Indra Nainggolan <>
#          Mico <>
#
# License: MIT License

import numpy as np

def euclidean_distance(X, Y):
    """
    Menghitung jarak euclidean dari 2 vektor

    Parameters
    ----------
    X : array berdimensi n
    Y : array berdimensi n

    Returns
    -------
    distance : Jarak euclidean
    """
    X = np.array(X)
    Y = np.array(Y)

    distance = np.sqrt(((X-Y)**2).sum())

    return distance

def manhattan_distance(X, Y):
    """
    Menghitung jarak manhattan dari 2 vektor

    Parameters
    ----------
    X : array berdimensi n
    Y : array berdimensi n

    Returns
    -------
    Jarak manhattan
    """
    X = np.array(X)
    Y = np.array(Y)

    distance = np.abs(X-Y).sum()

    return distance

def isMember(a, B):
    """
    """
    for b in B:
        if len(a)== len(b) :
            countTrue = 0
            for i in range(len(a)):
                 if (a[i] == b[i]):
                        countTrue +=1
            if countTrue == len(a):
                return True
    return False

class KMeans():
    """
    K-Means clustering

    Parameters
    ----------
    n_clusters : int, optional, default: 2
        Banyaknya kluster yang ingin dibentuk.

    tolerance : float, default: 0.0001
        Toleransi konvergen.

    max_iterations : int, default: 1000
        Banyak iterasi maksimum.

    Attributes
    ----------

    """
    def __init__(self, n_clusters=2, tolerance = 0.0001, max_iterations = 1000):
        self.k = n_clusters
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.Labels_ = []
        
        
    def fit(self,data):
        self.centroids = {}
        
        ## inisialisasi centroid menggunakan k data pertama
        for i in range(self.k):
            self.centroids[i] = data[i]
            
        for i in range(self.max_iterations):
            self.classifications = {}
            
            for i in range(self.k):
                self.classifications[i] = []
                
            for feature in data:
                distances = [euclidean_distance(feature,self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature)
            
            prev_centroids = dict(self.centroids)
            
            for classification in self.classifications :
                self.centroids[classification] = np.average(self.classifications[classification], axis =0)
                
            optimized = True
            
            # melakukan pengecekan apakah sudah konvergen atau belum dengan menggunakan tolerance
            for c in self.centroids :
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100) > self.tolerance:
                    optimized = False
            
            # Kalo sudah konvergen, break
            if optimized:
                self.Labels_ = [self.predict(instance) for instance in data]
                break
                
    def predict(self, instance) :
        distances = [euclidean_distance(instance,self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    def fit_predict(self, data) :
        self.fit(data)
        classifications = [self.predict(instance) for instance in data]
        return classifications




def _mDistance(start,end):
    return sum(abs(e - s) for s,e in zip(start,end))
    
def _random(bound,size):
    _rv = []
    _vis = []
    while True:
        r = np.random.randint(bound)
        if r in _vis:
            pass
        else:
            _vis.append(r)
            _rv.append(r)
        
        if len(_rv) == size:
            return _rv


class Medoid():
    """
    K-Medoid clustering dengan PAM (Partition Around Medoid)

    Parameters
    ----------
    n_clusters : int, optional, default: 2
        Banyaknya kluster yang ingin dibentuk.

    max_iterations : int, default: 1000
        Banyak iterasi maksimum.

    Attributes
    ----------

    """
    def __init__(self, n_clusters=2, max_iterations = 1000):
        self.k = n_clusters
        self.max_iterations = max_iterations
        self.Labels_ = []
        self._fit = False
   
    
    def fit(self,data):
        
        self.medoids = {}
        self.clusters = {}
        
        _med = _random(len(data),self.k)
        
        for i in range(self.k):
            self.medoids[i] = data[_med[i]]
        
        for _ in range(self.max_iterations):
            for j in range(self.k):
                self.clusters[j] = []
                
            for point in data:
                distances = [_mDistance(point, self.medoids[i]) if point.all() == self.medoids[i].all() else 0  for i in self.medoids]
                min_dist = np.min(distances)
                index = distances.index(min_dist)
                self.clusters[index].append(point)
                
            for i in self.clusters:
                distances = [[_mDistance(point, every_point) if point.all() == every_point.all() else 0 for every_point in self.clusters[i]] for point in self.clusters[i]] 
                costs = list(np.sum(distances, axis=1))
                index = costs.index(np.min(costs))
                self.medoids[i] = self.clusters[i][index]
    
        self._fit = True
                
                
    def predict(self, data):
        assert self._fit, "a program call to fit() must be given before calling predict()"
            
        predictions = []
        for point in data:
            distances = [_mDistance(point, self.medoids[i]) for i in self.medoids]
            min_dist = np.min(distances)
            index = distances.index(min_dist)
            predictions.append(index)
            
        return predictions


def linkage_distance(cluster_A, cluster_B, linkage, distance_function=euclidean_distance):
    
    # A ke kanan, B ke bawah
    matric_distance = []
    for i in range(len(cluster_A)):
        matric_distance.append([])
        for j in range(len(cluster_B)):
            matric_distance[i].append(distance_function(cluster_A[i],cluster_B[j]))
    
    if linkage == 'single':
        return min(min(matric_distance))
        
    elif linkage == 'complete':
        return max(max(matric_distance))
    
    elif linkage == 'average':
        x = [sum(a) for a in matric_distance]
        return min(x) / len(cluster_A)
    
    elif linkage == 'average group':
        pass

class Agglomerative():
    """
    Agglomerative clustering

    Parameters
    ----------
    n_clusters : int, optional, default: 2
        Banyaknya kluster yang ingin dibentuk.

    affinity : 

    linkage : 

    Attributes
    ----------

    """
    def __init__(self, n_clusters=2, affinity = linkage_distance, linkage = 'single'):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.classifications = []
        self.linkage = linkage
        self.clusters = []
    
    def fit(self,data):
        ## jika dilakukan fit ulang, data cluster direset
        self.clusters = []
        self.classifications = [0]* len(data)
        
        ## inisialisasi setiap data menjadi sebuah cluster
        for i in data:
            self.clusters.append([i])
        
        while len(self.clusters) != self.n_clusters:
            distances = [0] * len(self.clusters)

            for idx_cluster in range(len(self.clusters)): 
                distances[idx_cluster] = [self.affinity(self.clusters[idx_cluster],self.clusters[i],self.linkage) 
                             for i in range(len(self.clusters))]
                
            min_distances = [sorted(distance)[1]for distance in distances]
            
            x = min_distances.index(min(min_distances))
            y = distances[x].index(sorted(distances[x])[1])
                
            
            # cluster y dan x digabung ke x
            self.clusters[x] = self.clusters[x] + self.clusters[y]
            del self.clusters[y]
            
        for i in range(len(data)):
            for j in range(len(self.clusters)):
                if isMember(np.array(data[i]),self.clusters[j]) :
                    self.classifications[i] = j
                    
    def fit_predict(self, data) :
        self.fit(data)
        return self.classifications

class DBScan():
    """
    Agglomerative clustering

    Parameters
    ----------
    eps : float, optional, default: 0.5
        Nilai epsilon penentuan data dalam satu kluster.

    minPts : int, default: 5
        Banyaknya minimal data untuk membentuk core cluster

    distance_function : 
        Fungsi jarak yang digunakan

    Attributes
    ----------

    """
    def __init__(self, eps=0.5, minPts = 5, distance_function = euclidean_distance):
        self.eps = eps
        self.minPts = minPts
        self.classifications = []
        self.distance_function = distance_function
    
    def fit(self,data):
        # inisialisasi label dengan 0
        self.classifications = [0]*len(data)
        
        self.clusterID = 1
        
        ## inisialisasi centroid menggunakan k data pertama
        for point_id in range(len(data)):
            if (self.classifications[point_id] == 0):
                if self.canBeExpanded(point_id, data):
                    self.clusterID += 1
                    
        
    def canBeExpanded(self,point_id, data) :
        seeds = self.region_query(point_id,data)
        if len(seeds) < self.minPts:
            self.classifications[point_id] = -1
            return False
        else:
            self.classifications[point_id] = self.clusterID
            for seed_id in seeds : 
                self.classifications[seed_id] = self.clusterID
                
            while len(seeds) > 0:
                current_point = seeds[0]
                results = self.region_query(current_point,data)
                if len(results) >= self.minPts:
                    for i in range(len(results)):
                        result_point = results[i]
                        if self.classifications[result_point] <= 0 :
                            if self.classifications[result_point] == 0:
                                seeds.append(result_point)
                            self.classifications[result_point] = self.clusterID
                seeds = seeds[1:]
            return True
    
    def region_query(self,point_id,data):
        seeds = []
        for i in range(len(data)) :
            if self.distance_function(data[point_id],data[i]) <= self.eps:
                seeds.append(i)
        return seeds
                            
        
    def fit_predict(self,data):
        self.fit(data)
        return self.classifications