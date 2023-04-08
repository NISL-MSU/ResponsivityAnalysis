#  Helper functions used for Functional Data Analysis
# import umap
import skfda
import numpy as np
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA


def function_alignment(Rcurves: np.ndarray):
    """Align response curves according to the center of PIS
    @param Rcurves: Array with shape (N, num, 3) (N resp. curves with a range of num values).
                    Upper bound response curve: Rcurves[:, :, 0],
                    Lower bound response curve: Rcurves[:, :, 1],
                    Predicted response curve: Rcurves[:, :, 2].
    """
    # Calculate minimum value of each curve in ymean
    mins = np.min(Rcurves, axis=1)  # Rcurves[:, 0]
    mins = np.repeat(a=np.reshape(mins, (len(mins), 1)), repeats=Rcurves.shape[1], axis=1)
    # Perform alignment subtracting mins
    Rcurves[:, :] -= mins
    return Rcurves


def smooth_curves(Rcurves: np.ndarray) -> np.ndarray:
    """Smooth curves using B-Spline basis representation"""
    if Rcurves.ndim > 1:
        num = Rcurves.shape[1]
    else:
        num = Rcurves.shape[0]
    dataGrid = skfda.FDataGrid(data_matrix=Rcurves, grid_points=np.linspace(0, 1, num))
    basis = skfda.representation.basis.BSpline(n_basis=11)
    basis_fd = dataGrid.to_basis(basis)
    return basis_fd.to_grid(grid_points=np.linspace(0, 1, num)).data_matrix[:, :, 0]


def F_PCA(Rcurves: np.ndarray, n_components: int = 3):
    """Functional-PCA"""
    num = Rcurves.shape[1]
    dataGrid = skfda.FDataGrid(data_matrix=Rcurves, grid_points=np.linspace(0, 1, num))
    fpca = FPCA(n_components=n_components)
    transformed = fpca.fit_transform(dataGrid)
    # fpca.components_.plot()
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # color = cm.cool(np.linspace(0, 1, len(transformed)))
    # ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], alpha=.1)
    # c1, c2 = 50, 600
    # ax.scatter(transformed[c1, 0], transformed[c1, 1], transformed[c1, 2], s=51, c='r', alpha=1)
    # ax.scatter(transformed[c2, 0], transformed[c2, 1], transformed[c2, 2], s=51, c='r', alpha=1)
    # plt.figure()
    # plt.plot(Rcurves[50, :])
    # plt.plot(Rcurves[600, :])
    # # Plot using UMAP
    # reducer = umap.UMAP(metric='precomputed', n_components=1, n_neighbors=30, n_epochs=1000)
    # distance_matrix = pairwise_distance(Rcurves)
    # new_order = reducer.fit_transform(distance_matrix)
    # # Sort curves according to the new order
    # indices = np.argsort(new_order[:, 0])
    # plt.figure()
    # color = cm.Cool(np.linspace(0, 1, len(Rcurves)))
    # for c, g in Rcurves:
    #     plt.plot(g, c=color[c])
    return transformed, fpca


def lp_distance(Rcurve1, Rcurve2, p=2):
    """Lp distance between two response curves
    @param Rcurve1: Response curve 1.
    @param Rcurve2: Response curve 2.
    @param p: Value used to calculate the Lp distance.
    """
    diff = np.power(np.abs(Rcurve1 - Rcurve2), p)
    return np.power(np.sum(diff), 1 / p)


def pairwise_distance(Rcurves):
    n_samples = len(Rcurves)
    # Calculate distance matrix
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i):
            dist = lp_distance(Rcurves[i, :], Rcurves[j, :], p=2)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix


def t_statistic(RCurves1, RCurves2):
    """T-statistic for funcional data
    @param RCurves1: Group 1 of predicted response curves. Shape: (N1, num)
    @param RCurves2: Group 2 of predicted response curves. Shape: (N2, num)
    """

    # Calculate mean curves of each groups
    means1 = np.mean(RCurves1[:, :, 2], axis=0)
    means2 = np.mean(RCurves2[:, :, 2], axis=0)

    # Calculate standard errors
    std1 = np.std(RCurves1[:, :, 2], axis=0)
    std2 = np.std(RCurves2[:, :, 2], axis=0)
    # Difference
    diff = np.abs(means1 - means2)
    # Combined standard error
    std12 = np.sqrt(std1 ** 2 + std2 ** 2)
    # Check if means are significantly similar throughout t
    if np.all(diff - 1.96 * std12 <= 0):
        return True
    else:
        return False


def permutation_Ttest(RCurves, rep: int = 10000):
    Rcurves = RCurves.copy()
    # Calculate base statistic
    Tbase = t_statistic(Rcurves[0], Rcurves[1])

    # Start permutations
    T = np.zeros(rep)
    Rcurves_joined = np.concatenate((RCurves[0], RCurves[1]))
    for perm in range(rep):
        # Permute and separate into two new groups
        new_grouping = np.random.permutation(np.concatenate((np.zeros(len(RCurves[0]), ), np.ones(len(RCurves[1], )))))
        GroupA = Rcurves_joined[np.where(new_grouping == 0)[0]]
        GroupB = Rcurves_joined[np.where(new_grouping == 1)[0]]
        T[perm] = t_statistic(GroupA, GroupB)

    # Calculate mean pointwise p-value
    pvalues = np.mean(T >= Tbase)
    return np.mean(pvalues)
