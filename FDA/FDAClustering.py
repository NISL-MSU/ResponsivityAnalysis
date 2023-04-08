# import umap
import FDA.FDA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.cluster import AgglomerativeClustering


class AgglomerativeClust:

    def __init__(self, Rcurves: np.ndarray):
        """
        Agglomerative clustering for functional data.
        :param Rcurves: Array with shape (N, num, 3) (N resp. curves with a range of num values).
                    Upper bound response curve: Rcurves[:, :, 0],
                    Lower bound response curve: Rcurves[:, :, 1],
                    Predicted response curve: Rcurves[:, :, 2].
        """
        # Declare class variables
        self.fdata = Rcurves
        self.n_clusters = None
        self.distance_matrix = None
        self.n_samples, self.dim = self.fdata.shape

        self.centroids, self.centroids_ub, self.centroids_lb = None, None, None
        self._centroids_old, self._centroids_ub_old, self._centroids_lb_old = None, None, None

    def pairwise_distance(self):
        # Calculate distance matrix
        distance_matrix = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(i):
                dist = FDA.FDA.lp_distance(self.fdata[i, :], self.fdata[j, :], p=2)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        self.distance_matrix = distance_matrix

    def add_row_column(self, IndependentGroups, new_cluster_centroid):
        new_distance_matrix = np.zeros((self.distance_matrix.shape[0] + 1, self.distance_matrix.shape[1] + 1))
        new_distance_matrix[0:self.distance_matrix.shape[0], 0:self.distance_matrix.shape[1]] = self.distance_matrix
        for g in range(len(IndependentGroups) - 1):
            # Calculate distance from previous clusters to new cluster
            dist = FDA.FDA.lp_distance(np.mean(IndependentGroups[g], axis=0), new_cluster_centroid)
            new_distance_matrix[-1, g] = dist
            new_distance_matrix[g, -1] = dist
        self.distance_matrix = new_distance_matrix

    def fit(self):
        """Agglomerative algorithm"""
        self.pairwise_distance()
        clustering = AgglomerativeClustering(n_clusters=7).fit(self.distance_matrix)
        nc = len(np.unique(clustering.labels_))
        plt.figure()
        color = cm.Set1(np.linspace(0, 1, nc))
        for g in range(nc):
            for p in np.where(clustering.labels_ == g)[0]:
                plt.plot(self.fdata[p, :], c=color[g])
        plt.figure()
        nc1, nc2 = 1, 0
        for ind in np.where(clustering.labels_ == nc1)[0]:
            plt.plot(self.fdata[ind, :], color='r')
        for ind in np.where(clustering.labels_ == nc2)[0]:
            plt.plot(self.fdata[ind, :], color='g')


class FuzzyCMeans:

    def __init__(self, Rcurves: np.ndarray, c: int = 2, max_it: int = 100, tol: float = 1e-4, fuzzifier: float = 2):
        """
        Fuzzy C-means for functional data.
        :param Rcurves: Array with shape (N, num, 3) (N resp. curves with a range of num values).
                    Upper bound response curve: Rcurves[:, :, 0],
                    Lower bound response curve: Rcurves[:, :, 1],
                    Predicted response curve: Rcurves[:, :, 2].
        :param c: Number of desired clusters. Defaults to 2.
        :param max_it: Maximum number of iterations.
        :param tol: Tolerance used to compare previous and current centroids.
        :param fuzzifier: Scalar parameter used to specify the degree of fuzziness in the fuzzy algorithm. Defaults to 2.
        """
        # Declare class variables
        self.fdata = Rcurves[:, :, 2]
        self.fdata_ub = Rcurves[:, :, 0]
        self.fdata_lb = Rcurves[:, :, 1]
        self.widths = (Rcurves[:, :, 0] - Rcurves[:, :, 1])
        self.n_clusters = c
        self.n_samples, self.dim = self.fdata.shape
        self.max_it = max_it
        self.tol = tol
        self.fuzzifier = fuzzifier

        self.membership_matrix = np.zeros((self.n_samples, self.n_clusters))
        self.centroids, self.centroids_ub, self.centroids_lb = None, None, None
        self._centroids_old, self._centroids_ub_old, self._centroids_lb_old = None, None, None

    def _tolerance(self):
        """Define tolerance based on variance of the data"""
        variance = np.var(self.fdata, axis=0)
        return np.mean(variance * self.tol)

    def _distance_between_centroids(self):
        """Calculate all possible pairwise distances between centroids and old centroids"""
        dist = []
        for n in range(self.n_clusters):
            dist.append(FDA.FDA.lp_distance(self.centroids[n, :], self._centroids_old[n, :], p=2))
        return np.array(dist)

    def _prediction_from_membership(self):
        return np.argmax(self.membership_matrix, axis=1, )

    def fuzzy_silhouette(self):
        labels = self._prediction_from_membership()

        # Calculate silhouette coefficient
        s = np.zeros(self.n_samples)
        for ns in range(self.n_samples):
            # Average distance of object ns to all other objects belonging to the same cluster
            indices = np.where(labels == labels[ns])[0]  # Indices of samples with the same label
            dist = 0
            for ind in indices:
                if ind != ns:
                    dist += FDA.FDA.lp_distance(self.fdata[ns, :], self.fdata[ind, :], p=2)
            apj = dist / (len(indices) - 1)
            # Average distance of this object to all objects belonging to another cluster
            min_dist = np.infty
            for nc in range(self.n_clusters):
                if nc != labels[ns]:
                    indices = np.where(labels == nc)[0]  # Indices of samples with the same label
                    dist = 0
                    for ind in indices:
                        dist += FDA.FDA.lp_distance(self.fdata[ns, :], self.fdata[ind, :], p=2)
                    dist = dist / (len(indices))
                if dist < min_dist:
                    min_dist = dist
            bpj = min_dist
            # Silhouette
            s[ns] = (bpj - apj) / np.maximum(apj, bpj)

        # Calculate fuzzy silhouette
        nom, den = 0, 0
        for ns in range(self.n_samples):
            # Get two largest memberships
            u = self.membership_matrix[ns, :]
            [uqj, upj] = np.sort(u)[-2:]
            nom += (upj - uqj) * s[ns]
            den += (upj - uqj)

        FS = nom / den
        return FS

    def fit(self, random_state=7):
        """Fuzzy c-means algorithm"""
        repetitions = 0

        # Initialize centroids
        c_indices = np.arange(self.n_samples)
        np.random.seed(random_state)
        np.random.shuffle(c_indices)
        self.centroids = self.fdata[c_indices]
        self.centroids_ub = self.fdata_ub[c_indices]
        self.centroids_lb = self.fdata_lb[c_indices]
        self._centroids_old = self.centroids.copy()
        self._centroids_ub_old = self.centroids_ub.copy()
        self._centroids_lb_old = self.centroids_lb.copy()

        tolerance = self._tolerance()

        # Start iterations
        while repetitions == 0 \
                or (not np.all(self._distance_between_centroids() < tolerance) and repetitions < self.max_it):
            self._centroids_old = self.centroids.copy()
            self._centroids_ub_old = self.centroids_ub.copy()
            self._centroids_lb_old = self.centroids_lb.copy()

            # Compute distance from all curves to centroids
            distances_to_centroids = np.zeros((self.n_samples, self.n_clusters))
            for nc in range(self.n_clusters):
                for ns in range(self.n_samples):
                    dist = FDA.FDA.lp_distance(self.centroids[nc, :], self.fdata[ns, :], p=2)
                    distances_to_centroids[ns, nc] = dist

            ################
            # Perform update
            ################
            # Divisions by zero allowed
            with np.errstate(divide='ignore'):
                distances_to_centers_raised = (distances_to_centroids ** (2 / (1 - self.fuzzifier)))

            # Divisions infinity by infinity allowed
            with np.errstate(invalid='ignore'):
                self.membership_matrix[:, :] = (
                        distances_to_centers_raised / np.sum(distances_to_centers_raised, axis=1,
                                                             keepdims=True, ))
            # inf / inf divisions should be 1 in this context
            self.membership_matrix[np.isnan(self.membership_matrix)] = 1

            membership_matrix_raised = np.power(self.membership_matrix, self.fuzzifier, )

            slice_denominator = (
                    (slice(None),) + (np.newaxis,) * (self.fdata.ndim - 1))
            # Compute the new centroids and their corresponding PIs
            self.centroids = (
                    np.einsum('ij,i...->j...', membership_matrix_raised, self.fdata, ) /
                    np.sum(membership_matrix_raised, axis=0)[slice_denominator])
            self.centroids_ub = (
                    np.einsum('ij,i...->j...', membership_matrix_raised, self.fdata_ub, ) /
                    np.sum(membership_matrix_raised, axis=0)[slice_denominator])
            self.centroids_lb = (
                    np.einsum('ij,i...->j...', membership_matrix_raised, self.fdata_lb, ) /
                    np.sum(membership_matrix_raised, axis=0)[slice_denominator])

            repetitions += 1

        # Obtain labels
        labels = self._prediction_from_membership()
        highest_memberships = np.array([m[l] for m, l in zip(self.membership_matrix, labels)])
        # Get indices for each cluster
        indices = []
        for nc in range(self.n_clusters):
            indices.append(np.where(labels == nc)[0])
        return labels, indices, highest_memberships
