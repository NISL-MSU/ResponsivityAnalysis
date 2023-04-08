import h5py
import numpy as np
# import matplotlib.pyplot as plt


class DataLoader:

    def __init__(self, name):
        self.X, self.Y, self.types, self.names, self.modelType = np.zeros(0), np.zeros(0), None, None, None
        if name == "Synth":
            self.synthetic()
            self.modelType = "NN"
        elif name == "FieldA":
            self.fieldA()
            self.modelType = "Hyper3DNetLiteReg"
        elif name == "FieldB":
            self.fieldB()
            self.modelType = "Hyper3DNetLiteReg"

        # Shuffle dataset
        indexes = np.arange(len(self.X))
        np.random.seed(7)
        np.random.shuffle(indexes)
        self.X = self.X[indexes]
        self.Y = self.Y[indexes]

    def fieldA(self):
        hdf5_file = h5py.File('Datasets//fieldA_dataset.hdf5', mode='r')
        X = np.array(hdf5_file["data"][...]).astype(np.float32)
        Y = np.array(hdf5_file["target"][...])
        # Remove inputs where y==0
        valid = [np.all(Y[ind, :, :] > 0) for ind in range(Y.shape[0])]
        valid = np.where(valid)[0]
        self.X = X[valid][:, :, [0, 1, 3, 4, 5, 6, 7], :, :]
        self.Y = Y[valid][:, :, :]
        self.names = ['aa_n', 'slope', 'tpi', 'aspect_rad', 'prec_cy_g', 'vv_cy_f', 'vh_cy_f']
        self.types = ['real'] * self.X.shape[2]
        # Code to save dataset
        # import h5py
        # hdf5_path = 'fieldA_dataset.hdf5'
        # hdf5_file = h5py.File(hdf5_path, mode='w')
        # hdf5_file.create_dataset("data", trainx.shape, np.float32)
        # hdf5_file.create_dataset("target", train_y.shape, np.float32)
        # hdf5_file["target"][:, ...] = train_y  # Save yield data
        # hdf5_file["data"][:, ...] = trainx  # Save agg data
        # hdf5_file.close()

    def fieldB(self):
        hdf5_file = h5py.File('Datasets//fieldB_dataset.hdf5', mode='r')
        X = np.array(hdf5_file["data"][...]).astype(np.float32)
        Y = np.array(hdf5_file["target"][...])
        # Remove inputs where y==0
        valid = [np.all(Y[ind, :, :] > 0) for ind in range(Y.shape[0])]
        valid = np.where(valid)[0]
        self.X = X[valid][:, :, [0, 1, 3, 4, 5, 6, 7], :, :]
        self.Y = Y[valid][:, :, :]
        self.names = ['aa_n', 'slope', 'tpi', 'aspect_rad', 'prec_cy_g', 'vv_cy_f', 'vh_cy_f']
        self.types = ['real'] * self.X.shape[2]

    def synthetic(self, n=10000):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        np.random.seed(7)
        # Define features
        x1 = np.random.uniform(0, 1, size=n)  # x1 \in [0, 1]
        x2 = np.random.uniform(-3, 3, size=n)  # x2 \in [1, 2]
        x3 = np.random.uniform(1, 2, size=n)  # x3 \in [1, 2]
        x4 = np.random.uniform(1, 2, size=n)  # x4 \in [1, 2]
        x5 = np.random.uniform(0, 2, size=n)  # x5 \in [1, 2]
        self.X = np.array([x1, x2, x3, x4, x5]).T
        # Calculate output
        self.Y = sigmoid((10 * x1 - 5) + x2) * (x3 ** 2) * x4 + 10 * x5
        self.names = ['x1', 'x2', 'x3', 'x4', 'x5']
        self.types = ['real'] * self.X.shape[1]
        # Draw target response curves
        # Y = np.zeros((n, 50))
        # for i, x in enumerate(self.X):
        #     xvector = np.repeat(np.reshape(x, (1, len(x))), 50, axis=0)
        #     for j, xs in enumerate(np.linspace(start=0, stop=1, num=50)):
        #         xvector[j, 0] = xs  # Replace the s-th value
        #         Y[i, j] = sigmoid((10 * xvector[j, 0] - 5) + xvector[j, 1]) * (xvector[j, 2] ** 2) * xvector[j, 3] + \
        #                   10 * xvector[j, 4]
        # plt.figure()
        # for p in Y:
        #     plt.plot(p - p[0])
