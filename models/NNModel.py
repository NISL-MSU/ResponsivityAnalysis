import utils
import torch
import pickle
import random
import numpy as np
from tqdm import trange
from models.network import *
# import matplotlib.pyplot as plt

np.random.seed(7)  # Initialize seed to get reproducible results
random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#######################################################################################################################
# Static functions and Loss functions
#######################################################################################################################


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def DualAQD_objective(y_pred, y_true, beta_, pe):
    """Proposed DualAQD loss function,
    @param y_pred: NN output (y_u, y_l)
    @param y_true: Ground-truth.
    @param pe: Point estimate (from the base model).
    @param beta_: Specify the importance of the width factor."""
    # Separate upper and lower limits
    valid = np.where(y_true.cpu().numpy() != 0)
    y_u = y_pred[:, 0][valid]
    y_l = y_pred[:, 1][valid]
    y_o = pe.detach().squeeze(1)[valid]
    y_true = y_true[valid]
    # Calculate objectives
    MPIW_p = torch.mean(torch.abs(y_u - y_true) + torch.abs(y_true - y_l))  # Calculate MPIW_penalty
    cs = torch.max(torch.abs(y_o - y_true).detach())
    Constraints = (torch.exp(torch.mean(-y_u + y_true) + cs) +
                   torch.exp(torch.mean(-y_true + y_l) + cs))
    # Calculate loss
    return MPIW_p + Constraints * beta_


#######################################################################################################################
# Class Definitions
#######################################################################################################################

class NNObject:
    """Helper class used to store the main information of a NN model."""

    def __init__(self, model, criterion, optimizer):
        self.network = model
        self.criterion = criterion
        self.optimizer = optimizer


class NNModel:

    def __init__(self, device, nfeatures, method, modelType='NN', dataset='FieldA'):
        self.method = method
        self.modelType = modelType
        self.device = device
        self.nfeatures = nfeatures
        self.basemodel = None  # DualAQD uses a base model trained only for target prediction
        self.dataset = dataset

        if self.method == 'DualAQD':
            self.output_size = 2
        else:  # MC-Dropout
            self.output_size = 1

        criterion = nn.MSELoss()
        if modelType == 'NN':
            if dataset == 'FieldA':  # 4-hidden-layer NN
                network = NN(input_shape=self.nfeatures, output_size=self.output_size)
            else:  # 3-hidden-layer NN
                network = NN2(input_shape=self.nfeatures, output_size=self.output_size)
        else:
            network = Hyper3DNetLiteReg(input_shape=self.nfeatures, output_size=5, output_channels=self.output_size)
        network.to(self.device)
        # Training parameters
        optimizer = optim.Adadelta(network.parameters(), lr=0.1)

        self.model = NNObject(network, criterion, optimizer)

    def trainFold(self, Xtrain, Ytrain, Xval, Yval, batch_size, epochs, filepath, printProcess, yscale, alpha_=0.01):
        if self.method in ['AQD', 'MCDropout']:  # Initialize seed to get reproducible results when using these methods
            np.random.seed(7)
            random.seed(7)
            torch.manual_seed(7)
            torch.cuda.manual_seed(7)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Otherwise, QD+ and QD need to generate different NNs each time in order to create a diverse ensemble

        indexes = np.arange(len(Xtrain))  # Prepare list of indexes for shuffling
        np.random.shuffle(indexes)
        Xtrain = Xtrain[indexes]
        Ytrain = Ytrain[indexes]

        indexes = np.arange(len(Xtrain))  # Prepare list of indexes for shuffling
        np.random.shuffle(indexes)
        T = np.ceil(1.0 * len(Xtrain) / batch_size).astype(np.int32)  # Compute the number of steps in an epoch

        val_mse = np.infty
        val_picp = 0
        val_mpiw = np.infty
        MPIWtr = []
        PICPtr = []
        MSEtr = []
        MPIW = []
        PICP = []
        MSE = []
        BETA = []
        widths = [0]
        picp, picptr, max_picptr, epoch_max_picptr = 0, 0, 0, 0
        first95 = True  # This is a flag used to check if validation PICP has already reached 95% during the training
        first100 = False  # This is a flag used to check if training PICP has already reached 100% during the training
        top = 1
        tau = 0.95
        alpha_0 = alpha_

        # If model AQD, start with the pre-trained network
        if self.method in ['DualAQD']:
            self.basemodel = NNModel(self.device, self.nfeatures, 'MCDropout', modelType=self.modelType)
            filepathbase = filepath.replace('DualAQD', 'MCDropout')
            if 'TuningResults' in filepathbase:
                filepathbase = filepathbase.replace('TuningResults', 'CVResults')
            self.basemodel.loadModel(filepathbase)
            for target_param, param in zip(self.model.network.named_parameters(),
                                           self.basemodel.model.network.named_parameters()):
                if 'out' not in target_param[0]:
                    target_param[1].data.copy_(param[1].data)
        err_prev, err_new, beta_, beta_prev, d_err = 0, 0, 1, 0, 1

        for epoch in trange(epochs):  # Epoch loop
            # Batch sorting
            if epoch > 0 and (self.method in ['DualAQD']):
                if self.modelType != 'NN':
                    widths = np.mean(widths, axis=(1, 2))
                indexes = np.argsort(widths)
            else:
                indexes = np.arange(len(Xtrain))  # Prepare list of indexes for shuffling
            np.random.shuffle(indexes)

            self.model.network.train()  # Sets training mode
            running_loss = 0.0
            for step in range(T):  # Batch loop
                # Generate indexes of the batch
                inds = indexes[step * batch_size:(step + 1) * batch_size]

                # Get actual batches
                Xtrainb = torch.from_numpy(Xtrain[inds]).float().to(self.device)
                Ytrainb = torch.from_numpy(Ytrain[inds]).float().to(self.device)

                # zero the parameter gradients
                self.model.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model.network(Xtrainb)
                if self.method == 'DualAQD':
                    point_estimates = self.basemodel.model.network(Xtrainb)
                    loss = DualAQD_objective(outputs, Ytrainb, beta_=beta_, pe=point_estimates)
                else:
                    outputs = outputs.squeeze(1)
                    loss = self.model.criterion(outputs, Ytrainb)
                loss.backward()
                self.model.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if printProcess and epoch % 10 == 0:
                    print('[%d, %5d] loss: %.5f' % (epoch + 1, step + 1, loss.item()))

            # Validation step
            with torch.no_grad():
                self.model.network.eval()
                enable_dropout(self.model.network)
                ypredtr, ypred, petr, pe = 0, 0, 0, 0
                samples = 5
                for r in range(samples):
                    ypredtr += self.model.network(torch.from_numpy(Xtrain).float().to(self.device)).cpu().numpy()
                    ypred += self.model.network(torch.from_numpy(Xval).float().to(self.device)).cpu().numpy()
                    if self.method in ['DualAQD']:  # Include base model outputs
                        self.basemodel.model.network.eval()
                        petr += self.basemodel.model.network(torch.from_numpy(Xtrain).float().
                                                             to(self.device)).cpu().numpy()
                        pe += self.basemodel.model.network(torch.from_numpy(Xval).float().
                                                           to(self.device)).cpu().numpy()
                ypredtr /= samples
                ypred /= samples
                if self.method in ['DualAQD']:  # Attach base model output to the last column
                    petr /= samples
                    pe /= samples
                    if self.modelType == 'NN':
                        ypredtr = np.concatenate((ypredtr, petr), axis=1)
                        ypred = np.concatenate((ypred, pe), axis=1)
                    else:
                        temptr = np.zeros((len(Xtrain), 5, 5, 3))
                        temptr[:, :, :, 0:2], temptr[:, :, :, 2] = ypredtr.transpose((0, 2, 3, 1)), petr
                        ypredtr = temptr
                        temp = np.zeros((len(Xval), 5, 5, 3))
                        temp[:, :, :, 0:2], temp[:, :, :, 2] = ypred.transpose((0, 2, 3, 1)), pe
                        ypred = temp
                # Reverse normalization
                Ytrain_original = utils.reverseMinMaxScale(Ytrain, yscale[0], yscale[1])
                Yval_original = utils.reverseMinMaxScale(Yval, yscale[0], yscale[1])
                ypredtr = utils.reverseMinMaxScale(ypredtr, yscale[0], yscale[1])
                ypred = utils.reverseMinMaxScale(ypred, yscale[0], yscale[1])

                # Calculate MSE
                if self.method == 'DualAQD':
                    if self.modelType == 'NN':
                        msetr = utils.mse(Ytrain_original, ypredtr[:, 2])
                        mse = utils.mse(Yval_original, ypred[:, 2])
                    else:
                        msetr = utils.mse(Ytrain_original, ypredtr[:, :, :, 2])
                        mse = utils.mse(Yval_original, ypred[:, :, :, 2])
                else:
                    if self.modelType == 'NN':
                        msetr = utils.mse(Ytrain_original, ypredtr[:, 0])
                        mse = utils.mse(Yval_original, ypred[:, 0])
                    else:
                        msetr = utils.mse(Ytrain_original, ypredtr)
                        mse = utils.mse(Yval_original, ypred)
                MSEtr.append(msetr)
                MSE.append(mse)
                if self.method == 'DualAQD':
                    # Calculate MPIW and PICP
                    y_true = torch.from_numpy(Ytrain_original).float().to(self.device)
                    if self.modelType == 'NN':
                        y_utr = torch.from_numpy(ypredtr[:, 0]).float().to(self.device)
                        y_ltr = torch.from_numpy(ypredtr[:, 1]).float().to(self.device)
                    else:
                        y_utr = torch.from_numpy(ypredtr[:, :, :, 0]).float().to(self.device)
                        y_ltr = torch.from_numpy(ypredtr[:, :, :, 1]).float().to(self.device)
                    K_U = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_utr - y_true))
                    K_L = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_true - y_ltr))
                    Ktr = torch.mul(K_U, K_L)
                    picptr = torch.mean(Ktr).item()
                    y_true = torch.from_numpy(Yval_original).float().to(self.device)
                    if self.modelType == 'NN':
                        y_u = torch.from_numpy(ypred[:, 0]).float().to(self.device)
                        y_l = torch.from_numpy(ypred[:, 1]).float().to(self.device)
                    else:
                        y_u = torch.from_numpy(ypred[:, :, :, 0]).float().to(self.device)
                        y_l = torch.from_numpy(ypred[:, :, :, 1]).float().to(self.device)
                    K_U = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_u - y_true))
                    K_L = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_true - y_l))
                    K = torch.mul(K_U, K_L)
                    # Update curves
                    MPIWtr.append((torch.sum(torch.mul((y_utr - y_ltr), Ktr)) / (torch.sum(Ktr) + 0.0001)).item())
                    PICPtr.append(picptr)
                    width = (torch.sum(torch.mul((y_u - y_l), K)) / (torch.sum(K) + 0.0001)).item()
                    picp = torch.mean(K).item()
                    MPIW.append(width)
                    PICP.append(picp)
                    # Get a vector of all the PI widths in the training set
                    widths = (y_utr - y_ltr).cpu().numpy()

            # Save model if PICP increases
            if self.method == 'DualAQD':
                # Criteria 1: If <95, choose max picp, if picp>95, choose any picp if width<minimum width
                if (((val_picp == picp < tau and width < val_mpiw) or (val_picp < picp < tau)) and first95) or \
                        (picp >= tau and first95) or \
                        (picp >= tau and width < val_mpiw and not first95):  # and val_std < std
                    if picp >= tau:
                        first95 = False
                    val_mse = mse
                    val_picp = picp
                    val_mpiw = width
                    if filepath is not None:
                        torch.save(self.model.network.state_dict(), filepath)
            else:  # Save model if MSE decreases
                if mse < val_mse:
                    val_mse = mse
                    if filepath is not None:
                        torch.save(self.model.network.state_dict(), filepath)

            # Check if picp has reached convergence
            if picptr > max_picptr:
                max_picptr = picptr
                epoch_max_picptr = epoch
            else:
                if epoch == epoch_max_picptr + 50 and \
                        picptr <= max_picptr:  # If 30 epochs have passed without increasing PICP
                    first100 = True
                    top = tau
                    alpha_0 = alpha_ / 2

            # Beta hyperparameter
            if picptr >= 0.999 and not first100:
                first100 = True
                top = tau
                alpha_0 = alpha_ / 2
            err_new = top - picptr
            beta_ = beta_ + alpha_0 * err_new
            # Update parameters
            BETA.append(beta_)

            # Print every 10 epochs
            if printProcess and epoch % 10 == 0:
                if self.method == 'MCDropout':
                    print('VALIDATION: Training_MSE: %.5f. MSE val: %.5f, Best_MSE: %.5f' % (msetr, mse, val_mse))
                else:
                    print('VALIDATION: Training_MSE: %.5f. Best_MSEval: %.5f. MSE val: %.5f. PICP val: %.5f. '
                          'MPIW val: %.5f'
                          % (msetr, val_mse, mse, picp, width))
                    print(val_picp)
                    print(val_mpiw)
                    print(picptr)
                    print(beta_)
                    print(first100)

        # Save model
        if filepath is not None:
            with open(filepath + '_validationMSE', 'wb') as fil:
                pickle.dump(val_mse, fil)
            # Save history
            np.save(filepath + '_historyMSEtr', MSEtr)
            np.save(filepath + '_historyMSE', MSE)
            if 'QD' in self.method:  # Average upper and lower limit to obtain expected output
                np.save(filepath + '_historyMPIWtr', MPIWtr)
                np.save(filepath + '_historyPICPtr', PICPtr)
                np.save(filepath + '_historyMPIW', MPIW)
                np.save(filepath + '_historyPICP', PICP)

        return MPIW, PICP, MSE, val_mse, val_picp, val_mpiw

    def evaluateFold(self, valxn, maxs, mins, batch_size):
        """Retrieve point predictions."""
        if maxs is not None and mins is not None:
            valxn = utils.reverseMinMaxScale(valxn, maxs, mins)

        ypred = []
        with torch.no_grad():
            self.model.network.eval()
            Teva = np.ceil(1.0 * len(valxn) / batch_size).astype(np.int32)
            indtest = np.arange(len(valxn))
            for b in range(Teva):
                inds = indtest[b * batch_size:(b + 1) * batch_size]
                if self.method == 'DualAQD':
                    ypred_batch = self.basemodel.model.network(torch.from_numpy(valxn[inds]).float().to(self.device))
                else:
                    ypred_batch = self.model.network(torch.from_numpy(valxn[inds]).float().to(self.device))
                ypred = ypred + (ypred_batch.cpu().numpy()).tolist()

        return ypred

    def evaluateFoldUncertainty(self, valxn, maxs, mins, batch_size, MC_samples):
        """Retrieve point predictions and PIs"""
        np.random.seed(7)  # Initialize seed to get reproducible results
        random.seed(7)
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if maxs is not None and mins is not None:
            valxn = utils.reverseMinMaxScale(valxn, maxs, mins)

        with torch.no_grad():
            if self.modelType == 'NN':
                preds_MC = np.zeros((len(valxn), MC_samples))
            else:
                preds_MC = np.zeros((len(valxn), 5, 5, MC_samples))
            if self.method == "DualAQD":
                if self.modelType == 'NN':
                    preds_MC = np.zeros((len(valxn), 3, MC_samples))
                else:
                    preds_MC = np.zeros((len(valxn), 3, 5, 5, MC_samples))
            for it in range(0, MC_samples):  # Test the model 'MC_samples' times
                ypred = []
                self.model.network.eval()
                # enable_dropout(self.model.network)  # Set Dropout layers to test mode
                Teva = np.ceil(1.0 * len(valxn) / batch_size).astype(np.int32)  # Number of batches
                indtest = np.arange(len(valxn))
                for b in range(Teva):
                    inds = indtest[b * batch_size:(b + 1) * batch_size]
                    ypred_batch = self.model.network(torch.from_numpy(valxn[inds]).float().to(self.device))
                    if self.method == "DualAQD":
                        self.basemodel.model.network.eval()
                        enable_dropout(self.basemodel.model.network)
                        ypred_batch = ypred_batch.cpu().numpy()
                        ypred_batchtmp = np.zeros((ypred_batch.shape[0], 3))
                        pe_batch = self.basemodel.model.network(torch.from_numpy(valxn[inds]).float().to(self.device))
                        pe_batch = pe_batch.cpu().numpy()
                        if self.modelType == 'NN':
                            ypred_batchtmp[:, :2] = ypred_batch
                            ypred_batchtmp[:, 2] = pe_batch.squeeze(1)
                        else:
                            ypred_batchtmp[:, :2, :, :] = ypred_batch
                            ypred_batchtmp[:, 2, :, :] = pe_batch.squeeze(1)
                        ypred_batch = ypred_batchtmp
                    else:
                        ypred_batch = ypred_batch.squeeze(1)
                    ypred = ypred + ypred_batch.tolist()

                if self.method == "DualAQD":
                    if self.modelType == 'NN':
                        preds_MC[:, :, it] = np.array(ypred)
                    else:
                        preds_MC[:, :, :, :, it] = np.array(ypred)
                else:
                    if self.modelType == 'NN':
                        preds_MC[:, it] = np.array(ypred)
                    else:
                        preds_MC[:, :, :, it] = np.array(ypred)
        return preds_MC

    def loadModel(self, path):
        self.model.network.load_state_dict(torch.load(path, map_location=self.device))

        if self.method == 'DualAQD':
            self.basemodel = NNModel(self.device, self.nfeatures, 'MCDropout', dataset=self.dataset, modelType=self.modelType)
            filepathbase = path.replace('DualAQD', 'MCDropout')
            if 'TuningResults' in filepathbase:
                filepathbase = filepathbase.replace('TuningResults', 'CVResults')
            self.basemodel.loadModel(filepathbase)
