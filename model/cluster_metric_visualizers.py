import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance


class ClusterScoreVisualizers:
    @staticmethod
    def k_means_elbow_graph(data, from_k=1, to_k=20):
        # create new plot and data
        plt.plot()
        X = np.matrix(data)

        # k means determine k
        distortions = []
        K = range(from_k, to_k + 1)
        for k in K:
            print("-> Calculating k={}...".format(k))
            kmeans_model = KMeans(n_clusters=k)
            kmeans_model.fit(X)
            distortions.append(sum(np.min(distance.cdist(X, kmeans_model.cluster_centers_, 'euclidean'), axis=1))
                               / X.shape[0])

        print("-> Showing plot!")
        # Plot the elbow
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

    @staticmethod
    def sillhouette_coefficient(data, from_k=1, to_k=20):
        # create new plot and data
        plt.plot()
        X = np.matrix(data)
        colors = ['b', 'g', 'r']
        markers = ['o', 'v', 's']

        # k means determine k
        scs = []
        K = range(from_k, to_k + 1)
        for k in K:
            print("-> Calculating k={}...".format(k))
            clusterer = KMeans(n_clusters=k)
            preds = clusterer.fit_predict(data)

            scs.append(sklearn.metrics.silhouette_score(data, preds, metric='euclidean'))

        print("-> Showing plot!")
        # Plot the elbow
        plt.plot(K, scs, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Sillhouette Coefficient')
        plt.show()