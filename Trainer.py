import os
import sys
import time
import utils
import torch
import pickle
import numpy as np
# import matplotlib.pyplot as plt
from DataLoader import DataLoader
from models.NNModel import NNModel
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


class PIGenerator:

    def __init__(self, dataset='Synth', method='DualAQD'):
        """Class used for training the NNs
        :param dataset: Name of the dataset. Options: 'Synth', 'FieldA', or 'FieldB'/
        :param method: Method used for generating prediction intervals. Options: 'MC-Dropout' or 'DualAQD'. For the CFE
        analysis, use 'MC-Dropout' only.
        """
        self.dataset = dataset
        self.method = method
        # Load dataset
        dataLoader = DataLoader(name=dataset)
        self.X, self.Y, self.types, self.names, self.modelType = dataLoader.X, dataLoader.Y, dataLoader.types, \
                                                                 dataLoader.names, dataLoader.modelType
        # Initialize kfold object
        self.kfold = KFold(n_splits=10, shuffle=True, random_state=13)
        # Initialize model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.reset_model()

    def reset_model(self):
        if self.X.ndim == 2:
            return NNModel(device=self.device, nfeatures=self.X.shape[1], method=self.method,
                           modelType=self.modelType, dataset=self.dataset)
        else:
            return NNModel(device=self.device, nfeatures=self.X.shape[1:], method=self.method,
                           modelType=self.modelType, dataset=self.dataset)

    def train(self, crossval='10x1', batch_size=16, epochs=500, alpha_=0.01, printProcess=True, val=False):
        """Train using cross validation
        @param crossval: Type of cross-validation. Options: '10x1' or '5x2'
        @param batch_size: Mini batch size. It is recommended a small number, like 16
        @param epochs: Number of training epochs
        @param alpha_: Hyperparameter(s) used by the selected PI generation method
        @param printProcess: If True, print the training process (loss and validation metrics after each epoch)
        @param val: If True, just perform validation, not training."""
        # Create lists to store metrics
        cvmse, cvpicp, cvmpiw, cvdiffs = [], [], [], []

        # If the folder does not exist, create it
        folder = "CVResults//" + self.dataset + "//" + self.method
        if not os.path.exists("CVResults//" + self.dataset):
            os.mkdir("CVResults//" + self.dataset)
        if not os.path.exists(folder):
            os.mkdir(folder)

        if crossval == "10x1":
            iterator = self.kfold.split(self.X)
            print("Using 10x1 cross-validation for this dataset")
        elif crossval == "5x2":
            # Choose seeds for each iteration is using 5x2 cross-validation
            seeds = [13, 51, 137, 24659, 347, 436, 123, 64, 958, 234]
            iterator = enumerate(seeds)
            print("Using 5x2 cross-validation for this dataset")
        else:
            sys.exit("Only '10x1' and '5x2' cross-validation are permited.")

        ntrain = 1
        # Iterate through each partition
        for first, second in iterator:
            if ntrain >= 1:
                if crossval == '10x1':
                    train = np.array(first)
                    test = np.array(second)
                else:
                    train, test = train_test_split(range(len(self.X)), test_size=0.50, random_state=second)
                    train = np.array(train)
                    test = np.array(test)

                print("\n******************************")
                print("Analyzing fold: " + str(ntrain))
                print("******************************")
                # Normalize using the training set
                Xtrain, means, stds = utils.normalize(self.X[train])
                Ytrain, maxs, mins = utils.minMaxScale(self.Y[train])
                Xval = utils.applynormalize(self.X[test], means, stds)
                Yval = utils.applyMinMaxScale(self.Y[test], maxs, mins)

                # Define path where the model will be saved
                filepath = folder + "//weights-" + self.method + "-" + self.dataset + "-" + str(ntrain)

                # Train the model using the current training-validation split
                self.model = self.reset_model()
                mse, PICP, MPIW = None, None, None
                if not val:
                    _, _, _, mse, PICP, MPIW = self.model.trainFold(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval,
                                                                    batch_size=batch_size, epochs=epochs,
                                                                    filepath=filepath, printProcess=printProcess,
                                                                    alpha_=alpha_, yscale=[maxs, mins])

                # Run the model over the validation set 'MC-samples' times and Calculate PIs and metrics
                if self.method != 'DualAQD' or val:  # DualAQD already performs validation and aggregation in "trainFold"
                    [mse, PICP, MPIW, _, _, _] = self.calculate_metrics(Xval, Yval, maxs, mins, filepath)
                print('PERFORMANCE AFTER AGGREGATION:')
                print("Val MSE: " + str(mse) + " Val PICP: " + str(PICP) + " Val MPIW: " + str(MPIW))

                # Add metrics to the list
                cvmse.append(mse)
                cvpicp.append(PICP)
                cvmpiw.append(MPIW)

                # Reset all weights
                self.model = self.reset_model()
            ntrain += 1

        # Save metrics of all folds
        np.save(folder + '//validation_MSE-' + self.method + "-" + self.dataset, cvmse)
        np.save(folder + '//validation_MPIW-' + self.method + "-" + self.dataset, cvmpiw)
        np.save(folder + '//validation_PICP-' + self.method + "-" + self.dataset, cvpicp)
        if self.dataset == "Synth":
            np.save(folder + '//validation_DIFFS-' + self.method + "-" + self.dataset, cvdiffs)
        # Save metrics in a txt file
        file_name = folder + "//regression_report-" + self.method + "-" + self.dataset + ".txt"
        with open(file_name, 'w') as x_file:
            x_file.write("Overall MSE %.6f (+/- %.6f)" % (float(np.mean(cvmse)), float(np.std(cvmse))))
            x_file.write('\n')
            x_file.write("Overall PICP %.6f (+/- %.6f)" % (float(np.mean(cvpicp)), float(np.std(cvpicp))))
            x_file.write('\n')
            x_file.write("Overall MPIW %.6f (+/- %.6f)" % (float(np.mean(cvmpiw)), float(np.std(cvmpiw))))
            if self.dataset == "Synth":
                x_file.write('\n')
                x_file.write("Overall DIFF %.6f (+/- %.6f)" % (float(np.mean(cvdiffs)), float(np.std(cvdiffs))))

        return cvmse, cvmpiw, cvpicp

    def calculate_metrics(self, Xval, Yval, maxs, mins, filepath=None):
        """Calculate metrics using MC-Dropout to measure model uncertainty"""
        startsplit = time.time()

        self.model.loadModel(filepath)  # Load model
        # Get outputs using trained model
        yout = self.model.evaluateFoldUncertainty(valxn=Xval, maxs=None, mins=None, batch_size=32, MC_samples=50)
        yout = np.array(yout)
        if self.method in ['AQD', 'DualAQD']:
            # Obtain upper and lower bounds
            if self.modelType == 'NN':
                y_u = np.mean(yout[:, 0], axis=1)
                y_l = np.mean(yout[:, 1], axis=1)
                # Obtain expected target estimates
                ypred = np.mean(yout[:, 2], axis=1)
            else:
                y_u = np.mean(yout[:, :, :, 0], axis=1)
                y_l = np.mean(yout[:, :, :, 1], axis=1)
                # Obtain expected target estimates
                ypred = np.mean(yout[:, :, :, 2], axis=1)
            ypred = utils.reverseMinMaxScale(ypred, maxs, mins)
            y_u = utils.reverseMinMaxScale(y_u, maxs, mins)
            y_l = utils.reverseMinMaxScale(y_l, maxs, mins)
        else:
            # Load validation MSE
            with open(filepath + '_validationMSE', 'rb') as f:
                val_MSE = pickle.load(f)
            # Obtain expected target estimates
            yout = utils.reverseMinMaxScale(yout, maxs, mins)
            ypred = np.mean(yout, axis=-1)
            # Obtain upper and lower bounds
            model_uncertainty = np.std(yout, axis=-1)
            y_u = ypred + 1.96 * np.sqrt(model_uncertainty ** 2 + val_MSE)
            y_l = ypred - 1.96 * np.sqrt(model_uncertainty ** 2 + val_MSE)

        # Reverse normalization process
        Yval = utils.reverseMinMaxScale(Yval, maxs, mins)

        # Calculate MSE
        mse = utils.mse(Yval, ypred)
        # Calculate the coverage vector
        y_true = torch.from_numpy(Yval).float().to(self.device)
        y_ut = torch.from_numpy(y_u).float().to(self.device)
        y_lt = torch.from_numpy(y_l).float().to(self.device)
        K_U = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_ut - y_true))
        K_L = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_true - y_lt))
        K = torch.mul(K_U, K_L)
        # Calculate MPIW
        MPIW = torch.mean(y_ut - y_lt).item()
        # Calculate PICP
        PICP = torch.mean(K).item()

        endsplit = time.time()
        print("It took " + str(endsplit - startsplit) + " seconds to execute this batch")

        return [mse, PICP, MPIW, ypred, y_u, y_l]


if __name__ == '__main__':
    name = 'Synth'
    predictor = PIGenerator(dataset=name, method='MCDropout')
    predictor.train(crossval='10x1', batch_size=32, epochs=1000, printProcess=True, alpha_=0.01, val=False)
    name = 'FieldA'
    predictor = PIGenerator(dataset=name, method='MCDropout')
    predictor.train(crossval='10x1', batch_size=32, epochs=120, printProcess=True, alpha_=0.01, val=False)
    name = 'FieldB'
    predictor = PIGenerator(dataset=name, method='MCDropout')
    predictor.train(crossval='10x1', batch_size=32, epochs=120, printProcess=True, alpha_=0.01, val=False)
