import os
import sys
import time
import utils
import torch
import numpy as np
from FDA import FDA
from skfda import FDataGrid
from tqdm import trange
import matplotlib.pyplot as plt
from DataLoader import DataLoader
from models.NNModel import NNModel
from itertools import combinations
from sklearn.model_selection import KFold
from MOO import MixedVarsMOO, response_curves
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.visualization.scatter import Scatter
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination


class FeatureResponsivity:

    def __init__(self, dataset='Synth'):
        """
        @param dataset: Dataset name. Options: 'Insurance'.
        """
        self.dataset = dataset
        self.min_resp, self.max_resp = 0, 1

        # Load dataset
        dataLoader = DataLoader(name=dataset)
        self.X, self.Y = dataLoader.X, dataLoader.Y
        self.types, self.names, self.modelType = dataLoader.types, dataLoader.names, dataLoader.modelType

        # Load Model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.reset_model()

    def reset_model(self):
        if self.X.ndim == 2:
            return NNModel(device=self.device, nfeatures=self.X.shape[1], method='MCDropout',
                           modelType=self.modelType, dataset=self.dataset)
        else:
            return NNModel(device=self.device, nfeatures=self.X.shape[1:], method='MCDropout',
                           modelType=self.modelType, dataset=self.dataset)

    def impact(self, s: int, epsi: list, replace=False):
        """Estimate the impact of each feature on the responsivity of another feature.
        @param s: Index of the feature whose responsivity w.r.t response variable will be assessed.
        @param epsi: List of tolerance errors that will be tested.
        @param replace: If True, overwrite existing results.
        NOTE: This implementation assumes that the models are already trained using Trainer.py and 10x1 CV"""

        kfold = KFold(n_splits=10, shuffle=True, random_state=13)  # Initialize kfold object
        # If the folder does not exist, create it
        folder = "CVResults//" + self.dataset + "//" + 'MCDropout'
        if not os.path.exists("CVResults//" + self.dataset):
            sys.exit("The model does not exist. Use Trainer.py using 10x1 CV first.")
        folder2 = folder + "//Responsivity_feature-" + str(s)
        if not os.path.exists(folder2):
            os.mkdir(folder2)

        iterator = kfold.split(self.X)
        ntrain = 1
        Ids_cv = np.zeros((len(self.types) - 1, 11, len(epsi)))  # Save cross-validation results
        indexes = [fe for fe in np.arange(len(self.types)) if fe != s]  # List of variable features (!= s)
        Ids_cv[:, 0, :] = np.repeat(np.reshape(indexes, (len(indexes), 1)), len(epsi),
                                    axis=1)  # First column saves the feature indexes
        # Create list with all possible combination of feature indexes (up to a max. of 4 features)
        indx = np.arange(len(indexes))
        feat_combs = []
        for v in range(2, 5):
            feat_combs += list(combinations(indx, v))
        combs_matrix = np.zeros((len(feat_combs), 10, len(epsi)))

        # Iterate through each partition
        samples = None
        for first, second in iterator:
            print("\n******************************")
            print("Analyzing fold: " + str(ntrain))
            print("******************************")
            impact_results_path = [folder2 + "//Local_Ids_s-" + str(s) + "_sample_matrix_fold-" + str(ntrain) +
                                   "_epsi-" + str(np.round(eps, 1)) + '.npy' for eps in epsi]
            train = np.array(first)
            # Normalize using the training set. NOTE: These are the same partitions used during training (same seed).
            Xtrain, means, stds = utils.normalize(self.X[train])

            # Define path where the model is saved and load it
            filepath = folder + "//weights-MCDropout-" + self.dataset + "-" + str(ntrain)
            self.model.loadModel(filepath)

            # Get minimum and maximum values, and count how many possible values each feature has
            if self.X.ndim == 2:
                xmin, xmax, num = np.zeros((self.X.shape[1])), np.zeros((self.X.shape[1])), np.zeros((self.X.shape[1]))
                xmin_orig, xmax_orig = np.zeros((self.X.shape[1])), np.zeros((self.X.shape[1]))
                for f in range(self.X.shape[1]):
                    xmin[f], xmax[f], num[f] = np.min(Xtrain[:, f]), np.max(Xtrain[:, f]), len(np.unique(Xtrain[:, f]))
                    xmin_orig[f], xmax_orig[f] = np.min(self.X[train][:, f]), np.max(self.X[train][:, f])
            else:
                xmin, xmax, num = np.zeros((self.X.shape[2])), np.zeros((self.X.shape[2])), np.zeros((self.X.shape[2]))
                xmin_orig, xmax_orig = np.zeros((self.X.shape[2])), np.zeros((self.X.shape[2]))
                for f in range(self.X.shape[2]):
                    xmin[f], xmax[f], num[f] = np.min(Xtrain[:, :, f, :, :]), np.max(Xtrain[:, :, f, :, :]), \
                        len(np.unique(Xtrain[:, :, f, :, :]))
                    xmin_orig[f], xmax_orig[f] = np.min(self.X[train][:, :, f, :, :]), \
                        np.max(self.X[train][:, :, f, :, :])
                if self.dataset != 'Synth':
                    xmax[s] = (150 - means[s]) / stds[s]
                    xmin[s] = (0 - means[s]) / stds[s]
                    xmin_orig[s] = 0
                    xmax_orig[s] = 150

            ###########################################
            # MOO Problem
            ###########################################
            samples = len(train)
            if len(train) > 500:  # Limit the experiment to 500 samples
                samples = 500
            Ids = np.zeros((samples, len(self.types) - 1, len(epsi)))

            # If not all results are already saved, generate response curves
            if (not np.all([os.path.exists(path) for path in impact_results_path])) or replace:
                # Get response curves for all points in the dataset:
                print("Generating response curves for the entire training set...")
                start = time.time()
                Rcurves = response_curves(model=self.model, X=Xtrain, s=s, xmin=xmin[s], xmax=xmax[s], num=num[s],
                                          MC=True)
                end = time.time()
                print("It took " + str(end - start) + " s. to generate all the response curves of the s-th feature.")

                ###########################################
                # PRE-PROCESSING
                ###########################################
                # Response curves alignment
                Rcurves = FDA.function_alignment(Rcurves)
                # Apply FPCA
                Rcurves_transformed, fpca = FDA.F_PCA(Rcurves)

                ###########################################
                # CFE ANALYSIS
                ###########################################
                # Analyze each sample of the dataset
                for i in trange(samples):
                    x_orig = self.X[train][i, :]
                    # Calculate mutation probability of each variable given current values
                    probs = []
                    for f in range(len(self.types)):
                        if self.types[f] in ['binary', 'categorical']:
                            # Count how many samples exist in the dataset with different values than the current one
                            num_diff = len(np.where(self.X[train][:, f] != self.X[train][i, f])[0])
                            probs.append(num_diff / len(train))
                        else:
                            probs.append(None)  # If numerical, use default probability

                    # Find maximum distance between sample and other samples in the transformed space
                    x_orig_transformed = fpca.transform(FDataGrid(data_matrix=Rcurves[i],
                                                                  grid_points=np.linspace(0, 1, 50)))
                    ds = []
                    for t in Rcurves_transformed:
                        ds.append(FDA.lp_distance(x_orig_transformed, t))
                    epsi_max = np.max(ds)

                    curves, x_norm = [], None
                    for ne, eps in enumerate(epsi):
                        time.sleep(0.05)
                        print("\n**************************")
                        print("Analyzing epsilon=" + str(np.round(eps, 2)))
                        print("**************************")
                        n_threads = 20
                        pool = ThreadPool(n_threads)
                        runner = StarmapParallelization(pool.starmap)

                        # Define multi-objective optimization problem
                        if eps > epsi_max:
                            eps = epsi_max
                        moo = MixedVarsMOO(x_orig=x_orig, model=self.model, epsi=eps, transformer=fpca,
                                           probs=probs, resp_orig=Rcurves[i], s=s, types=self.types,
                                           xmin=xmin_orig, elementwise_runner=runner,
                                           xmax=xmax_orig, minS=xmin[s], maxS=xmax[s], num=num[s], stats=[means, stds])

                        # Optimize using NSGA-II
                        algorithm = NSGA2(pop_size=50,
                                          sampling=MixedVariableSampling(),
                                          mating=MixedVariableMating(
                                              eliminate_duplicates=MixedVariableDuplicateElimination()),
                                          eliminate_duplicates=MixedVariableDuplicateElimination(), )
                        res = minimize(moo, algorithm, ('n_gen', 80), seed=1, verbose=False)

                        # Select best solution
                        sols = res.F
                        sols[:, 0] = np.round(sols[:, 0], 2)
                        bestf1 = np.min(sols[:, 0])
                        f1sols = sols[(sols[:, 0] == bestf1)]  # Select solutions that produced the highest resp. change
                        bestf2 = np.min(f1sols[:, 1])  # The solution that required to change the fewer features
                        best_sol = np.where((sols[:, 0] == bestf1) & (sols[:, 1] == bestf2))[0][0]
                        # Assess which features changed the most
                        if self.modelType == 'NN':
                            x_opt = np.array(
                                [res.X[best_sol][f"x{k:02}"] if k != s else x_orig[s] for k in range(len(self.types))])
                        else:
                            x_opt = np.array([res.X[best_sol][f"x{k:02}"] if k != s else x_orig[0, s, 2, 2] for k in
                                              range(len(self.types))])
                            diff = x_orig[0, :, 2, 2] - x_opt
                            diff = np.multiply(np.ones(x_orig.shape), np.reshape(diff, (1, x_orig.shape[1], 1, 1)))
                            x_opt = x_orig[0, :, :, :] - diff
                        dist = utils.gower(x_orig, x_opt, types=self.types, ranges=(xmax_orig - xmin_orig))
                        Ids[i, :, ne] = np.delete(dist, s)  # Remove feature s
                        print(str(Ids[i, :, ne] > 0.01))
                        if self.modelType == 'NN':
                            x_norm = np.reshape(Xtrain[i, :], (1, len(self.types)))
                        else:
                            x_norm = Xtrain[i:i + 1, :, :, :, :]
                        x_norm2 = utils.applynormalize(np.reshape(x_opt, x_norm.shape), means, stds)
                        curve2 = response_curves(self.model, x_norm2, s, xmin=xmin[s], xmax=xmax[s], num=num[s]) \
                                  / 10 * np.max(self.Y[train])
                        curve2 = FDA.smooth_curves(curve2)
                        curves.append(FDA.function_alignment(curve2))

                    #############################################################
                    # PLOT Diference between original and resulting response curve
                    #############################################################
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111)
                    # curve = response_curves(self.model, x_norm, s, xmin=xmin[s], xmax=xmax[s], num=num[s]) \
                    #         / 10 * np.max(self.Y[train])
                    # curve = FDA.function_alignment(curve)
                    # curve = FDA.smooth_curves(curve)
                    # xlabels = np.linspace(xmin_orig[s], xmax_orig[s], curve.shape[1])
                    # ax.plot(xlabels, curve[0, :], linewidth=7, label=r'$\tilde{R}(\mathbf{x})$')
                    # for ne, eps in enumerate(epsi):
                    #     ax.plot(xlabels, curves[ne][0, :],
                    #             linewidth=7, label=r"$\tilde{R}(\mathbf{x}')$. $ϵ = $" + str(eps))
                    # ax.locator_params(axis='x', nbins=2)
                    # plt.xticks([xmin_orig[s], xmax_orig[s]], ['$x_s^{(min)}$', '$x_s^{(max)}$'], fontsize=16)
                    # plt.yticks(fontsize=16)
                    # plt.xlabel('$x_s$', fontsize=20)
                    # plt.ylabel('$\hat{y}$', fontsize=20)
                    # plt.legend(fontsize=16)
                    # ax.set_aspect(30 / 1)  # For fields A and B
                    # ax.set_aspect(1 / 6)  # For the synthetic dataset
                    # plt.xlabel('Nitrogen Rate')
                    # plt.ylabel('Predicted Yield')
                    # plt.pause(0.05)
            else:
                for npath, path in enumerate(impact_results_path):
                    Ids[:, :, npath] = np.load(path)

            ###################
            # Analyze results
            ###################
            for ne, eps in enumerate(epsi):
                Ids_bin = Ids[:, :, ne] > 0.005  # Hihglight the features that were changed for each sample
                plt.figure()
                plt.imshow(Ids_bin, aspect='auto')
                plt.xticks(np.arange(len(indexes)), [self.names[indexes[v]] for v in range(len(indexes))])
                plt.pause(0.05)
                # Save results
                np.save(impact_results_path[ne], Ids[:, :, ne])
                # Create an array of lists that stores the indexes of the features that changed
                Ilist = np.frompyfunc(list, 0, 1)(np.empty((Ids.shape[0],), dtype=object))
                for n in range(Ids.shape[0]):
                    Ilist[n] = list(np.where(Ids_bin[n])[0])
                # Calculate the percentage of times that each variable changed
                percentage = [100 * np.sum(Ids_bin[:, v]) / Ids.shape[0] for v in range(Ids.shape[1])]
                # Retrieve the unique combination of changing variables and how many times they appeared
                combs, comb_num = np.unique(Ilist, return_counts=True)
                sorted_comb_num = np.argsort(comb_num)
                sorted_combinations = combs[sorted_comb_num]  # Combinations of features sorted by repetition
                if len(sorted_combinations) >= 5:
                    top5combinations = sorted_combinations[-5:][::-1]
                else:
                    top5combinations = sorted_combinations[::-1]

                # Update number of multiple feature combinations
                for vi, v in enumerate(feat_combs):
                    combs_matrix[vi, ntrain - 1, ne] = 0
                    if list(v) in list(combs):  # If the combination appeared, count how many times
                        p = [p for p, vt in enumerate(combs) if vt == list(v)][0]
                        combs_matrix[vi, ntrain - 1, ne] = comb_num[p]

                # Print results
                print("**************************")
                print("Results for epsilon=" + str(np.round(eps, 2)))
                print("**************************")
                for v in range(len(indexes)):
                    print('\033[0m' + "The feature " + str(indexes[v]) + " (" + self.names[
                        indexes[v]] + ") was modified " +
                          '\033[1m' + str(np.round(percentage[v], 2)) + " % of the time." + '\033[0m')
                print("\nThe top-5 most repeated combination of passive features was:")
                for v in range(len(top5combinations)):
                    print(
                        "Combination " + str(v) + " : " + str([self.names[indexes[vi]] for vi in top5combinations[v]]))
                print("\nThe top-5 most repeated combination of multiple passive features was:")

                Ids_cv[:, ntrain, ne] = percentage
            ntrain += 1

        # Count the most frequent feature combinations
        for ne, eps in enumerate(epsi):
            print("\nThe top-5 most repeated combination of multiple passive features for epsilon = " + str(eps))
            combs_matrixn = combs_matrix[:, :, ne] / samples * 100
            combs_percentage = np.mean(combs_matrixn, axis=1)
            combs_std = np.std(combs_matrixn, axis=1)
            combs_indx = np.argsort(combs_percentage)
            sorted_combs = np.array(feat_combs)[combs_indx]  # Combinations of features sorted by repetition
            if len(sorted_combs) >= 5:
                top5combinations = sorted_combs[-5:][::-1]
            else:
                top5combinations = sorted_combs[::-1]
            for vi, v in enumerate(range(len(top5combinations))):
                print(
                    "Combination " + str(v) + " : " + str([self.names[indexes[vi]] for vi in top5combinations[v]]) +
                    ". Percentage = " + str(combs_percentage[combs_indx[-vi - 1]]) + " +/- " + str(
                        combs_std[combs_indx[-vi - 1]]))
        # Plot results of CV
        for ne, eps in enumerate(epsi):
            plt.figure()
            plt.imshow(Ids_cv[:, 1:, ne].T, aspect='auto')
            plt.xticks(np.arange(len(indexes)), [self.names[indexes[v]] for v in range(len(indexes))])

            np.save(folder2 + "//Global_Ids_s-" + str(s) + "_epsi-" + str(np.round(eps, 1)) + "_fold-" + str(ntrain),
                    Ids_cv[:, :, ne])
            # Save metrics
            file_name = folder2 + "//Global_Ids_s-" + str(s) + "_epsi-" + str(np.round(eps, 1)) + ".txt"
            with open(file_name, 'w') as x_file:
                for f in range(Ids_cv.shape[0]):
                    head = "Global Impact of feature " + str(indexes[f]) + " (" + self.names[indexes[f]] + \
                           ") on responsivity of feature " + str(s) + " = "
                    x_file.write(
                        head + "%.4f (+/- %.4f)" % (
                            float(np.mean(Ids_cv[f, 1:, ne])), float(np.std(Ids_cv[f, 1:, ne]))))
                    x_file.write('\n')


if __name__ == '__main__':
    name = 'Synth'
    fresp = FeatureResponsivity(dataset=name)
    fresp.impact(s=0, epsi=[0.4, 0.6, 0.8], replace=True)
    name = 'FieldA'
    fresp = FeatureResponsivity(dataset=name)
    fresp.impact(s=0, epsi=[0.6, 0.8, 1], replace=True)
    name = 'FieldB'
    fresp = FeatureResponsivity(dataset=name)
    fresp.impact(s=0, epsi=[0.6, 0.8, 1], replace=True)
