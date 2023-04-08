import numpy as np
import matplotlib.pyplot as plt


def mse(imageA, imageB):
    """Calculate the 'Mean Squared Error' between the two images."""
    return np.mean((imageA - imageB) ** 2)


def normalize(trainx):
    """Normalize and returns the calculated means and stds for each feature"""
    trainxn = trainx.copy()
    if 1 < trainxn.ndim < 4:
        means = np.zeros((trainx.shape[1], 1))
        stds = np.zeros((trainx.shape[1], 1))
        for n in range(trainx.shape[1]):
            means[n, ] = np.mean(trainxn[:, n])
            stds[n, ] = np.std(trainxn[:, n])
            trainxn[:, n] = (trainxn[:, n] - means[n, ]) / (stds[n, ])
    elif trainxn.ndim > 4:
        means = np.zeros((trainx.shape[2], 1))
        stds = np.zeros((trainx.shape[2], 1))
        for n in range(trainx.shape[2]):
            means[n, ] = np.mean(trainxn[:, :, n, :, :])
            stds[n, ] = np.std(trainxn[:, :, n, :, :])
            trainxn[:, :, n, :, :] = (trainxn[:, :, n, :, :] - means[n, ]) / (stds[n, ])
    else:
        means = np.mean(trainxn)
        stds = np.std(trainxn)
        trainxn = (trainxn - means) / stds
    return trainxn, means, stds


def applynormalize(testx, means, stds):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    if 1 < testxn.ndim < 4:
        for n in range(testx.shape[1]):
            testxn[:, n] = (testxn[:, n] - means[n, ]) / (stds[n, ])
    elif testxn.ndim >= 4:
        for n in range(testx.shape[2]):
            testxn[:, :, n, :, :] = (testxn[:, :, n, :, :] - means[n, ]) / (stds[n, ])
    else:
        testxn = (testxn - means) / stds
    return testxn


def reversenormalize(testx, means, stds):
    """Reverse normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    if 1 < testxn.ndim < 4:
        for n in range(testx.shape[1]):
            testxn[:, n] = (testxn[:, n] * stds) + means
    elif testxn.ndim > 4:
        for n in range(testx.shape[2]):
            testxn[:, :, n, :, :] = (testxn[:, :, n, :, :] * stds[n, ]) + means[n, ]
    else:
        testxn = (testxn * stds) + means

    return testxn


def minMaxScale(trainx):
    """Normalize and returns the calculated max and min of the output"""
    trainxn = trainx.copy()
    maxs = np.max(trainxn)
    mins = np.min(trainxn)
    trainxn = (trainxn - mins) / (maxs - mins) * 10
    return trainxn, maxs, mins


def applyMinMaxScale(testx, maxs, mins):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    testxn = (testxn - mins) / (maxs - mins) * 10
    return testxn


def reverseMinMaxScale(testx, maxs, mins):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    if testxn.ndim > 1:
        for n in range(testx.shape[1]):
            testxn[:, n] = (testxn[:, n] * (maxs - mins) / 10) + mins
    else:
        testxn = (testxn * (maxs - mins) / 10) + mins

    return testxn


def create_synth_data(n=1000, plot=False):
    """Create a synthetic sinusoidal dataset with varying PI width"""
    np.random.seed(7)
    X = np.linspace(-5, 5, num=n)
    randn = np.random.normal(size=n)
    gauss = (2 + 2 * np.cos(1.2 * X))
    noise = gauss * randn
    orig = 10 + 5 * np.cos(X + 2)
    Y = orig + noise
    P1 = orig + 1.96 * gauss
    P2 = orig - 1.96 * gauss
    if plot:
        plt.figure(figsize=(9.97, 7.66))
        plt.fill_between(X, P1, P2, color='gray', alpha=0.5, linewidth=0, label='Ideal 95% PIs')
        plt.scatter(X, Y, label='Data with noise')
        plt.plot(X, orig, 'r', label='True signal')
        plt.legend()
    return X, Y, P1, P2


def gower(x1, x2, types, ranges):
    """Calculate Gower distance"""
    dist = []
    for j in range(len(types)):
        if (types[j] == 'categorical') or (types[j] == 'binary'):
            dist.append(int(x1[j] != x2[j]))
        else:
            if x1.ndim <= 2:
                dist.append((np.abs(x1[j] - x2[j])) / ranges[j])
            else:
                dist.append((np.abs(x1[0, j, 2, 2] - x2[0, j, 2, 2])) / ranges[j])
    return np.array(dist)


def plot_numpy_matrix(cm, cmap=plt.cm.Reds):
    """
    This function prints and plots a numpy array as an image.
    Normalization can be applied by setting `normalize=True`.
    """
    label_font = {'size': '15'}
    fig, ax = plt.subplots(figsize=(2.5, 3.4))
    ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto', vmin=0, vmax=100)
    plt.xticks(np.arange(6), ['S\n$r_2$', 'TPI\n$r_3$', 'A\n$r_4$', 'P\n$r_5$', 'VH\n$r_6$', 'VV\n$r_7$'], fontsize=13)
    ax.xaxis.tick_top()
    plt.yticks(np.arange(10), np.arange(1, 11), fontsize=11)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, '{0:.0f}'.format(cm[i, j]),
                         horizontalalignment="center",
                         verticalalignment="center", fontsize=11,
                         color="white" if cm[i, j] > 50 else "black",fontweight='bold')

    plt.tight_layout()
    plt.ylabel('CV Iteration', fontdict=label_font)
    plt.xlabel('$ϵ = 0.6$', fontdict=label_font)
