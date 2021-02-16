import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, nbr_iters=1000):
        self.lr =learning_rate
        self.lambda_param= lambda_param
        self.nbr_iters =nbr_iters
        self.w=None
        self.b=None

    def fit(self, X, y):

        # making sure y has value of 1 or -1
        y_=np.where(y<=0, -1,1)

        # getting number of samples and features from numpy array X 
        nbr_samples, nbr_features =X.shape

        # initializing 'w' weights and 'b' bias with value 0 for all features
        self.w =np.zeros(nbr_features)
        self. b=0

        # gradient descent
        for _ in range(self.nbr_iters):
            for idx, x_i in enumerate(X):
                condition=y_[idx]*(np.dot(x_i, self.w)-self.b)>= 1
                if condition:
                    self.w -= self.lr*(2*self.lambda_param*self.w)
                else:
                    self.w -= self.lr * (2*self.lambda_param*self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):

        #wx-b using dot product
        linear_output =np.dot(X, self.w) - self.b

        return np.sign(linear_output)