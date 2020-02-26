# check the following link for using cvxopt qp solver
# http://cvxopt.org/examples/tutorial/qp.html
from sklearn import datasets # For Using for Iris data
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import random
import math
import copy


class SVM(object):

    def __init__(self, kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def gram_matrix(self, X, k, p):
        gram_matrix_initiliase = np.zeros((k, k))
        for m in range(k):
            for n in range(k):
                gram_matrix_initiliase[m, n] = self.kernel(X[m], X[n])
        return gram_matrix_initiliase

    def fit(self, X, y):
        print(X.shape)
        n_samples, n_features = X.shape

        # Kernel/Gram matrix
        # to do - compute the kernel matrix given the choice of the kernel function
        gram_matrix_initiliase = self.gram_matrix(X, n_samples, n_features)

        # Compute the parameters to be sent to the solver
        # P, q, A, b - refer to lab for more information.
        # remember to use cvxopt.matrix for storing the vector and matrices.
        product_p = np.outer(y, y) * gram_matrix_initiliase
        P = cvxopt.matrix(product_p)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)

        if self.C is not None:
            # soft margin case
            # Compute the parameters to be sent to the solver
            # G, h - refer to the lab for information about it.
            # remember to use cvxopt.matrix for storing the vector and matrices.
            # cvxopt does not work with numpy matrix
            matrix_new = np.ones(n_samples) * -1
            new_matrix = np.diag(np.ones(matrix_new))
            old_matrix = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((new_matrix, old_matrix)))
            new_matrix = np.zeros(n_samples)
            old_matrix = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((new_matrix, old_matrix)))

        else:
            # linear separable case
            # Compute the parameters to be sent to the solver
            # G, h - refer to the lab for more information.
            # remember to use cvxopt.matrix for storing the vector and matrices.
            # cvxopt does not work with numpy matrix
            matrix_new = np.ones(n_samples) * -1
            G = cvxopt.matrix(np.diag(matrix_new))
            h = cvxopt.matrix(np.zeros(n_samples))

        # solve QP problem once we have all the parameters of the solver specifed, let us solve it!
        # uncomment the line below once you have specified all the parameters.
        # A=A.astype(double)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Obtain the Lagrange multipliers from the solution.
        alpha = np.ravel(solution['x'])

        # Support vectors have non zero Lagrange multipliers
        # apply a threshold on the value of alpha and identify the support vectors
        # print the fraction of support vectors.
        sv = alpha > 1e-7
        # y.reshape((1, -1))
        ind = np.arange(len(alpha))[sv]
        self.alpha = alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Weight vector
        # compute the weight vector using the support vectors only when using linear kernel
        if self.kernel != linear_kernel:
            self.w = None
        else:
            #print(n_features)
            self.w = np.zeros(n_features)
            for n in range(len(self.alpha)):
                self.w += self.alpha[n] * self.sv_y[n] * self.sv[n]

        # Intercept
        # computer intercept term by taking the average across all support vectors
        self.W0 = 0
        for n in range(len(self.alpha)):
            self.W0 += self.sv_y[n]
            self.W0 -= np.sum(self.alpha * self.sv_y * gram_matrix_initiliase[ind[n], sv])
        self.W0 /= len(self.alpha)

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.W0
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for alpha, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                    s += alpha * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.W0

    def predict(self, X):
        # implement the function to predict the class label for a test set.
        # return the class label and the output f(x) for a test data point
        M = self.project(X)
        return (M)

    def predict1(self, X):
        # implement the function to predict the class label for a test set.
        # return the class label and the output f(x) for a test data point
        M = self.project(X)
        return np.sign(M)



if __name__ == "__main__":

    def linear_kernel(x1, x2):
        # Implement the linear kernel
        dot_product = np.dot(x1, x2)
        return dot_product


    def polynomial_kernel(x1, x2, q=3):
        # Implement the polynomial kernel
        dot_product = np.dot(x1, x2)
        dot_product = 1 + dot_product
        # Power of the resultant Dot Product
        return dot_product ** q


    def gaussian_kernel(x1, x2, s=2):
        # Implement the radial basis function kernel
        resultant = -1 * linalg.norm(x1 - x2) ** 2
        resultant = resultant / (s ** 2)
        resultant = np.exp(resultant)
        return resultant


    def split_train_test(X, y):
        # Since 100 Instances is mentioned in Question, Considering 75 Instances as Training data
        # And Remaining data 25 Instances as Test data
        train_len = math.floor(3 / 4 * (150))
        X1 = [[]]
        X2 = [[]]
        y1 = []
        y2 = []
        # Since we are Taking Half from Positive and Half from Negative point)
        # Creating Splits For Training Data
        for i in range(train_len):
            key = random.randint(0, 149)
            X1.append(X[key])
            y1.append(y[key])
        train_len1 = 150 - train_len
        for i in range(train_len1):
            key = random.randint(0, 149)
            X2.append(X[key])
            y2.append(y[key])
        X1 = X1[1:]
        X2 = X2[1:]
        X1 = np.array(X1)
        X2 = np.array(X2)
        y1 = np.array(y1)
        y2 = np.array(y2)
        return X1, y1, X2, y2


    def linear_svm(X, y,X_train, X_test, y_train, y_test , count):
        # 1. generate linearly separable data
        # 2. split the data into train and test sets
        # 3. create an SVM object called model (uses linear kernel)
        # 4. train the SVM using the fit function and the training data
        # 5. compute the classes of the model for the test data
        # 6. compute the accuracy of the model
        # 7. plot the training data points, and the margin.
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        if count == 1:
            model = SVM(linear_kernel)
        if count == 2:
            model = SVM(polynomial_kernel)
        if count == 3:
            model = SVM(gaussian_kernel)
        model.fit(X_train, y_train)
        model_output_prediction = model_prediction(model, X_test, y_test)
        return model_output_prediction


    def kernel_svm(X, y,X_train,y_train, X_test, y_test ,  count):
        # 1. generate non-linearly separable data
        # 2. split the data into train and test sets
        # 3. create an SVM object called model using an appropriate kernel function
        # 4. train the SVM using the fit function and the training data
        # 5. compute the classes of the model for the test data
        # 6. compute the accuracy of the model
        # 7. plot the contours of the model's output using the plot_contour function
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        #solvers.options['show_progress'] = False
        if count == 1:
            model = SVM(linear_kernel)
        if count == 2:
            model = SVM(polynomial_kernel)
        if count == 3:
            model = SVM(gaussian_kernel)
        model.fit(X_train, y_train)
        model_output_prediction = model_prediction(model, X_test, y_test)
        return model_output_prediction


    def soft_svm(X_train, y_train, X_test, y_test,count):
        # 1. generate linearly separable overlapping data
        # 2. split the data into train and test sets
        # 3. create an SVM object called model (uses linear kernel, and the box penalty parameter)
        # 4. train the SVM using the fit function and the training data
        # 5. compute the classes of the model for the test data
        # 6. compute the accuracy of the model
        # 7. plot the training data points, and the margin.
        #X_train, y_train, X_test, y_test = split_train_test(X1, y1, X2, y2)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        X = X_train
        y = y_test
        if count == 1:
            model = SVM(linear_kernel)
        if count == 2:
            model = SVM(polynomial_kernel)
        if count == 3:
            model = SVM(gaussian_kernel)
        model.fit(X_train, y_train)
        model_output_prediction = model_prediction(model, X_test, y_test)
        return model_output_prediction


    def model_prediction(model, X_test, y_test):
        y_predict = model.predict(X_test)
        #print(y_predict)
        correct = np.sum(y_predict == y_test)
        y_predict1 = model.predict1(X_test)
        # print(y_predict)
        correct1 = np.sum(y_predict1 == y_test)
        accuracy = correct1 / len(y_predict1)
        #print("Accuracy is " + str(float(100 * accuracy)))
        return y_predict





    # Load the Data from Scikit Learn
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    #print(y_train)
    # Implementing Multi class SVM based on One Vs All approach and taking max of all
    options = [1,2,3]
    for option in options :
        if option == 1:
            print_kernel = "Linear Kernel"
        elif option == 2:
            print_kernel = "Polynomial Kernel"
        else:
            print_kernel = "Gaussian Kernel"
        y1 = []
        print("\n")
        print("Started Implementing Multi Class SVM using Kernel Function is "+print_kernel)
        print("\n")
        for i in y_train:
            if i == 0:
                y1.append(1)
            else:
                y1.append(-1)
        #print(y1)
        y2 = []
        for i in y_test:
            if i == 0:
                y2.append(1)
            else:
                y2.append(-1)
        X_train1 = copy.deepcopy(X_train)
        X_test1 = copy.deepcopy(X_test)
        y_train1 = copy.deepcopy(y1)
        y_test1 = copy.deepcopy(y_test)
        model1_result = kernel_svm(X, y1,X_train1, y1, X_test1, y2, option)
        y1 = []
        for i in y_train:
            if i == 1:
                y1.append(1)
            else:
                y1.append(-1)
        y2 = []
        for i in y_test:
            if i == 1:
                y2.append(1)
            else:
                y2.append(-1)
        X_train2 = copy.deepcopy(X_train)
        X_test2= copy.deepcopy(X_test)
        y_train2 = copy.deepcopy(y1)
        y_test2 = copy.deepcopy(y2)
        model2_result = kernel_svm(X, y1,X_train2, y1, X_test2, y2, option)
        y1 = []
        for i in y_train:
            if i == 2:
                y1.append(1)
            else:
                y1.append(-1)
        y2 = []
        for i in y_test:
            if i == 1:
                y2.append(1)
            else:
                y2.append(-1)
        X_train3 = copy.deepcopy(X_train)
        X_test3 = copy.deepcopy(X_test)
        y_train3 = copy.deepcopy(y1)
        y_test3= copy.deepcopy(y_test)
        model3_result = kernel_svm(X, y1,X_train3, y1, X_test3, y2, option)
        #Uncomment the Follwing to see Model result predicitons in 3 cases (Because 3 classes):
        print(model1_result)
        print(model2_result)
        print(model3_result)
        class_predicted_multi_class_svm = []
        for i in range(len(model1_result)) :
            p =[]
            m = model1_result[i]
            n = model2_result[i]
            o = model3_result[i]
            p.append(m)
            p.append(n)
            p.append(o)
            k = p.index(max(p))
            #print(k)
            class_predicted_multi_class_svm.append(k)
        count=0
        for m in range(len(X_test)) :
            #print((y_test[m]))
            #print((class_predicted_multi_class_svm[m]))
            if y_test[m] == class_predicted_multi_class_svm[m] :
                count=count+1
        accuracy = count/len(y_test)
        print("Number of Correct prediction by multi class SVM using Kernel function :  "+print_kernel+" is :   " +str(100*accuracy))


