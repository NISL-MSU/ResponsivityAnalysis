import sys
import numpy as np
import skfda.preprocessing.dim_reduction.feature_extraction

import FDA.FDA
from utils import gower, applynormalize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Binary


#############################################################################
# RESPONSE CURVE GENERATION
#############################################################################

def response_curves(model, X, s, xmin, xmax, num, MC=True):
    """Get the response curves of the s-th feature for each input sample
    @param model: Pytorch model.
    @param X: Data samples.
    @param s: Index of the feature whose responsivity will be assessed.
    @param xmin: Minimum value each feature can take.
    @param xmax: Maximum value each feature can take.
    @param num: Number of possible values the s-th feature can take.
    @param MC: If True, use MC-Dropout.
    """
    if MC:  # Activate dropout unit
        MCsamples = 1
    else:
        MCsamples = 1
    if num > 50:
        num = 50

    Y = np.zeros((X.shape[0], num))
    for n, x in enumerate(X):
        if model.modelType == 'NN':
            xvector = np.repeat(np.reshape(x, (1, len(x))), num, axis=0)
            for i, xs in enumerate(np.linspace(start=xmin, stop=xmax, num=num)):
                xvector[i, s] = xs  # Replace the s-th value
        else:
            xvector = np.repeat(np.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2], x.shape[3])), num, axis=0)
            for i, xs in enumerate(np.linspace(start=xmin, stop=xmax, num=num)):
                xvector[i, 0, s, :, :] = xs  # Replace the s-th value
            xvector = np.repeat(xvector, MCsamples, axis=0)

        # Pass through the NN
        y = model.evaluateFoldUncertainty(valxn=xvector, maxs=None, mins=None, batch_size=2000, MC_samples=1)
        if model.modelType == 'NN':
            Y[n, :] = np.mean(y, axis=-1)
        else:
            for t in range(num):
                y2 = [yy for yy in y[t * MCsamples:(t + 1) * MCsamples, 2, 2, :] if yy != 0]
                Y[n, t] = np.mean(y2)
    return Y


#############################################################################
# OBJECTIVE FUNCTIONS
#############################################################################

def L1(new_resp: np.ndarray, old_resp: np.ndarray, fpca: skfda.preprocessing.dim_reduction.feature_extraction.FPCA,
       threshold: int):
    """Objective function 1: Modify response curve"""
    # Convert curves into FDatagrid
    new_resp = skfda.FDataGrid(data_matrix=new_resp, grid_points=np.linspace(0, 1, len(new_resp)))
    old_resp = skfda.FDataGrid(data_matrix=old_resp, grid_points=np.linspace(0, 1, len(old_resp)))
    # Transform curves using fPCA
    new_resp = fpca.transform(new_resp)
    old_resp = fpca.transform(old_resp)
    # Calculate L2 distance in the new space
    d = FDA.FDA.lp_distance(new_resp, old_resp)
    if d > threshold:
        d = threshold
    return - d


def L2(x_orig, x_counter, types, ranges):
    """Objective function 2: Modify as few features as possible"""
    comp = []
    for i in range(len(types)):
        if x_orig.ndim <= 2:
            if types[i] == 'real':
                comp.append(int(np.abs(x_orig[i] - x_counter[i]) / ranges[i] > 0.01))
            else:
                comp.append(int(x_orig[i] != x_counter[i]))
        else:
            if types[i] == 'real':
                comp.append(int(np.mean(np.abs(x_orig[0][i] - x_counter[0][i])) / ranges[i] > 0.01))
            else:
                comp.append(int(x_orig[0][i] != x_counter[0][i]))

    return np.sum(comp)


def L3(new_resp, Resp_orig):
    """Objective function 3: Counterfactual explanations should have sound feature value combinations"""
    distances = [FDA.FDA.lp_distance(new_resp, resp) for resp in Resp_orig]
    return sorted(distances)[-1]


def L4(x_new, x_old, types, ranges):
    """Objective function 4: Counterfactual explanations should be close to the original feature values"""
    return np.mean(gower(x_new, x_old, types, ranges))


#############################################################################
# MULTI-OBJECTIVE OPTIMIZATION
#############################################################################

class MixedVarsMOO(ElementwiseProblem):

    def __init__(self, x_orig, transformer, model, epsi, probs, s, types, xmin, xmax, minS, maxS, num, stats,
                 resp_orig=None, centroids=None, **kwargs):
        """Initialize Multi-Objective Optimization object.
        @param x_orig: Original input values
        @param model: Pytorch model.
        @param epsi: Tolerance error.
        @param probs: Probability of mutation of each feature
        @param s: Index of the variable whose responsitivity is being analyzed.
        @param types: Array containing the types of input variables.
        @param xmin: Minimum original value each feature can take.
        @param xmax: Maximum original value each feature can take.
        @param minS: Minimum transformed value the s-th feature can take.
        @param maxS: Maximum transformed original value the s-th feature can take.
        @param num: Number of possible values the s-th feature can take.
        @param stats: Statistics used to apply z-score normalization before passing the input through the model.
        @param resp_orig: Original response curve.
        @param centroids: Centroid curves that will be used for comparison.
        """
        # Class variables
        self.x_orig = x_orig
        self.transformer = transformer
        self.model = model
        self.epsi = epsi
        # self.XObs = XObs
        self.probs = probs
        self.types = types
        self.s = s
        self.minS = minS
        self.maxS = maxS
        self.num = num
        self.stats = stats
        self.centroids = centroids
        self.ranges = xmax - xmin

        # Calculate original responsivity
        if model.modelType == 'NN':
            x_norm = applynormalize(np.reshape(self.x_orig, (1, len(self.types))), self.stats[0], self.stats[1])
        else:
            x_norm = applynormalize(np.reshape(self.x_orig, (1, self.x_orig.shape[0], self.x_orig.shape[1],
                                                             self.x_orig.shape[2], self.x_orig.shape[3])),
                                    self.stats[0], self.stats[1])
        self.resp_orig = resp_orig
        if resp_orig is None:
            self.resp_orig = self.ind_responsivity(x_norm, s, minS, maxS, self.num, MC=True)

        # Declare optimization variables
        variables = dict()
        for k in [i for i in range(len(self.types)) if i != s]:  # Consider all variables except the s-th variable
            if self.types[k] == 'real':
                var = Real(bounds=(xmin[k], xmax[k]))
            elif self.types[k] == 'integer':
                var = Integer(bounds=(int(xmin[k]), int(xmax[k])))
            elif self.types[k] == 'binary':
                var = Binary()
                var.prob = self.probs[k]  # Assign the probability of mutation of the variable
                ########################################################################################################
                # NOTE: Using pymoo 0.6.0
                # * The instance variable "prob" needs to be added to the pymoo.core.Variable class (self.prob=None)
                #
                # * The "pymoo.core.mutation.Mutation.get_prob_var" is modified as follows to support different
                #   individual mutation probabilities:
                #   if all([v.prob is not None for v in problem.vars]):  # Check if indiv. probabilities were assigned
                #       prob_var = [v.prob for v in problem.vars]
                #   else:
                #       prob_var = self.prob_var if self.prob_var is not None else min(0.5, 1 / problem.n_var)
                # * Finally, the pymoo.operators.mutation.bitflip.BitflipMutation._do method needs to consider different
                #   mutation probabilities as well so the following:
                #   prob_var = self.get_prob_var(problem, size=(len(X), len(self.get_prob_var(problem))))
                ########################################################################################################
            else:
                sys.exit("Only accepting real, integer and binary values for now.")
            variables[f"x{k:02}"] = var

        super().__init__(vars=variables, n_obj=3)  # , n_ieq_constr=1)

    def ind_responsivity(self, x, s, xmin, xmax, num, MC=True):
        """Evaluate individual responsivity of an input x"""
        # Obtain N-response curve
        Rcurve = response_curves(self.model, x.copy(), s, xmin, xmax, num, MC=MC)
        # Curve alignment
        return FDA.FDA.function_alignment(Rcurve)[0]

    def _evaluate(self, x, out, *args, **kwargs):
        # Generate response curve of the counterfactual
        if self.model.modelType == 'NN':
            # Complete input vector using a dummy value for the s-th feature
            x = np.array([x[f"x{k:02}"] if k != self.s else self.x_orig[self.s] for k in range(len(self.types))])
            x_norm = applynormalize(np.reshape(x, (1, len(self.types))), self.stats[0], self.stats[1])
        else:
            x = np.array([x[f"x{k:02}"] if k != self.s else self.x_orig[0, self.s, 2, 2] for k in range(len(self.types))])
            diff = self.x_orig[0, :, 2, 2] - x
            diff = np.multiply(np.ones(self.x_orig.shape), np.reshape(diff, (1, self.x_orig.shape[1], 1, 1)))
            x = self.x_orig[0, :, :, :] - diff
            # for k in range(len(self.types)):

            x_norm = applynormalize(np.reshape(x, (1, self.x_orig.shape[0], self.x_orig.shape[1],
                                                             self.x_orig.shape[2], self.x_orig.shape[3])),
                                    self.stats[0], self.stats[1])
        resp = self.ind_responsivity(x_norm, self.s, self.minS, self.maxS, self.num, MC=True)
        resp = FDA.FDA.smooth_curves(resp)[0, :]

        # Calculate first objective
        f1 = L1(resp, self.resp_orig, fpca=self.transformer, threshold=self.epsi)
        # Calculate second objective
        f2 = L2(x, self.x_orig, self.types, self.ranges)
        # Calculate THIRD objective
        f3 = L4(x, self.x_orig, self.types, self.ranges)
        out["F"] = [f1, f2, f3]
