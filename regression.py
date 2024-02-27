# This class allows you to apply two fundamental Machine Learning models- Linear Regression and Logistic Regression.
import numpy as np

class Regression:

    def __init__(self, X, w, b, model = 'linear'):
        self.X = np.array(X)
        self.w = np.array(w)
        self.b = b
        self.model = model

# linear regression function
    def f(self):
        linear_output = []
        for i in range(len(self.X)):
            linear_output.append(np.dot(self.X[i], self.w) + self.b)
        return linear_output

# logistic regression function 
    def g(self):
        logistic_output = []
        for i in range(len(self.f())):
            logistic_output.append(1/(1+np.exp(-self.f()[i])))
        return logistic_output

    def prediction(self):
        if self.model == 'linear':
            return self.f()
        elif self.model == 'logistic':
            return self.g()

class Compute_cost:

    def __init__(self, X, Y, w, b, model = 'linear'):
        """
        X: test data used to predict
        Y: true prediction values
        w: tunable parameter for ML algorithm (array)
        b: tunable parameter for ML algorithm (float)
        model: ML model used to predict. Current options: 'linear' and 'logistic' regression.
        """
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.w = np.array(w)
        self.b = b
        self.model = model
         
    cost = 0

    def compute_cost(self):

        self.m = self.X.shape[0]
    
        for i in range(self.m):
            self.cost += (Regression(self.X, self.w, self.b, self.model).prediction()[i] - self.Y[i])**2

        self.total_cost = self.cost / (2 * self.m)

        return self.total_cost

class Compute_gradient:

    def __init__(self, X, Y, w, b, model = 'linear', alpha = 0.01):
        """
        X: test data used to predict
        Y: true prediction values
        w: tunable parameter for ML algorithm (array)
        b: tunable parameter for ML algorithm (float)
        model: ML model used to predict. Current options: 'linear' and 'logistic' regression.
        """
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.w = np.array(w)
        self.b = b
        self.model = model
        self.alpha = alpha

    w_history = []
    b_history = []

    def w_gradient(self):
        
        self.w_history.append(self.w)
        self.m = self.X.shape[0]
        self.partial_d_dw = 0

        for i in range(self.m):
            self.partial_d_dw += (Regression(self.X, self.w, self.b, self.model).prediction()[i] - self.Y[i]) * self.X[i]
        
        self.d_dw = self.partial_d_dw / self.m

        self.w = self.w - self.alpha * self.d_dw
        self.w_history.append(self.w)

        return self.w
    


    def min_w(self):

        self.counter = 0

        while self.w_gradient()/self.w_history[-2] < 0.99 or self.w_gradient()/self.w_history[-2] > 1.01:
            self.w_gradient()
            self.counter +=1
            if self.counter > 1000:
                break

        return self.w

    def b_gradient(self):
        
        self.b_history.append(self.b)
        self.m = self.X.shape[0]
        self.partial_d_db = 0

        for i in range(self.m):
            self.partial_d_db += (Regression(self.X, self.w, self.b, self.model).prediction()[i] - self.Y[i])
        
        self.d_db = self.partial_d_db / self.m

        self.b = self.b - self.alpha * self.d_db
        self.b_history.append(self.b)

        return self.b
    


    def min_b(self):

        self.counter = 0

        while self.b_gradient()/self.b_history[-2] < 0.999 or self.b_gradient()/self.b_history[-2] > 1.001:
            self.b_gradient()
            self.counter +=1
            if self.counter > 1000:
                break

        return self.b