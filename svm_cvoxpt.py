
# check the following link for using cvxopt qp solver
# http://cvxopt.org/examples/tutorial/qp.html


import numpy as np
from numpy import linalg
import math
from pylab import rand
import cvxopt
import cvxopt.solvers
import random
             

class support_vectorM(object):
    def __init__(self, kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)


    def linear_kernel_case(self,k):
        w = np.zeros(k)
        for n in range(len(self.alpha)):
            w += self.alpha[n] * self.support_vector_y[n] * self.support_vector[n]
        return w


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Kernel/Gram matrix
        # to do - compute the kernel matrix given the choice of the kernel function
        gram_matrix_initiliase = self.gram_matrix(X,n_samples,n_features)

        # Compute the parameters to be sent to the solver
        # P, q, A, b - refer to lab for more information.
        # remember to use cvxopt.matrix for storing the vector and matrices.
        product_p = np.outer(y,y) * gram_matrix_initiliase
        P = cvxopt.matrix(product_p)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples),'d')
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
            old_matrix = np.ones(n_samples) * self.C
            new_matrix = np.zeros(n_samples)
            h = cvxopt.matrix(np.hstack((new_matrix, old_matrix)))
            
        else : 
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
        #A=A.astype(double)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Obtain the Lagrange multipliers from the solution.
        alpha = np.ravel(solution['x'])

        # Support vectors have non zero Lagrange multipliers
        # apply a threshold on the value of alpha and identify the support vectors
        # print the fraction of support vectors.
        support_vector = alpha > 1e-3
        ind = np.arange(len(alpha))[support_vector]
        self.alpha = alpha[support_vector]
        self.support_vector = X[support_vector]
        self.support_vector_y = y[support_vector]
        
        # Weight vector
        # compute the weight vector using the support vectors only when using linear kernel
        if self.kernel != linear_kernel:
            self.w = None
        else:
            #linear_kernel_case(self,n_features,alpha,support_vector_y,support_vector)
            self.w = self.linear_kernel_case(n_features)

        # Intercept
        # computer intercept term by taking the average across all support vectors
        self.W0 = 0
        self.W0 = self.update_w0(gram_matrix_initiliase,ind,support_vector)

    
    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.W0
        else:
            y_prediction = self.prediction_w(X)
            return y_prediction + self.W0


    def update_w0(self,gram,ind,sv):
        for n in range(len(self.alpha)):
            self.W0 += self.support_vector_y[n]
            self.W0 -= np.sum(self.alpha * self.support_vector_y * gram[ind[n],sv])
        self.W0 /= len(self.alpha)
        return self.W0
        
    def prediction_w(self,X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for alpha, support_vector_y, support_vector in zip(self.alpha, self.support_vector_y, self.support_vector):
                s += alpha * support_vector_y * self.kernel(X[i], support_vector)
            y_predict[i] = s
        return y_predict


    def predict(self, X):
        # implement the function to predict the class label for a test set.
        # return the class label and the output f(x) for a test data point
        M = self.project(X)
        return np.sign(M)


    def gram_matrix(self, X, k,p):
        gram_matrix_initiliase = np.zeros((k, k))
        for m in range(k):
            for n in range(k):
                gram_matrix_initiliase[m, n] = self.kernel(X[m], X[n])
        return gram_matrix_initiliase



if __name__ == "__main__":
    import pylab as pl

    
    def linear_kernel(x1, x2):
        #Implement the linear kernel
        dot_product = np.dot(x1,x2)
        return dot_product

    def polynomial_kernel(x1, x2, q=3):
        #Implement the polynomial kernel
        dot_product = np.dot(x1,x2)
        dot_product = 1 + dot_product
        #Power of the resultant Dot Product
        return dot_product**q

    def gaussian_kernel(x1, x2, s=0.9):
        # Implement the radial basis function kernel
        resultant = -1*linalg.norm(x1-x2)**2
        resultant = resultant/(2*s**2)
        resultant = np.exp(resultant)
        return resultant


      
    def lin_separable_data():
        # generate linearly separable 2D data
        # remember to assign class labels as well. 
        n=100
        y1 =[]
        y2 = []
        x_base = (rand(n)*10-1)/10-1
        y_base = (rand(n)*10-1)/10+1
        x_r_base = (rand(n)*10-1)/10+1
        y_r_base = (rand(n)*10-1)/10-1
        X1=[[]]
        X2 =[[]]
        for i in range(len(x_base)):
            X1.append([x_base[i],y_base[i]])      
            X2.append([x_r_base[i],y_r_base[i]])
            y2.append(-1)
            y1.append(1)
        X1 = X1[1:]
        X2 = X2[1:]
        print(len(X1))
        print(len(X2))
        print("Positive Points  - Linear Seperable Data")
        print(X1)
        print("Negative Points  - Linear Seperable Data")
        print(X2)
        return X1,y1,X2,y2

      
    def lin_separable_overlap_data():
        # for testing the soft margin implementation, 
        # generate linearly separable data where the instances of the two classes overlap.
        n=50
        y1 =[]
        y2 = []
        X1=[[]]
        X2 =[[]]
        for i in range(100) :
          point = []
          m1 = random.uniform(-1,2.1)
          n1 = random.uniform(-1,2.1)
          point.append(m1)
          point.append(n1)
          X1.append(point)
          y1.append(1)
        for i in range(100) :
          point = []
          m1 = random.uniform(1,4)
          n1 = random.uniform(1,4)
          point.append(m1)
          point.append(n1)
          X2.append(point)
          y2.append(-1)
        X1 = X1[1:]
        X2 = X2[1:]
        print("Positive Points  - Linear Seperable Overlap Data")
        print(X1)
        print("Negative Points  - Linear Seperable Overlap Data")
        print(X2)
        return X1,y1,X2,y2
        
    
    
      
    def circular_data():
        # let us complicate things a little to study the advantage of using Kernel functions
        # generate data that is separable using a circle
        value_center_x = random.randint(1,2)
        value_center_y = random.randint(3,5)
        X1 = [[]]
        X2 = [[]]
        y1=[]
        y2 =[]
        angles_postive = []
        angles_negative = []
        for m in range(25) :
          np.random.seed(m)
          angle = random.uniform(0,360)
          angles_postive.append(angle)
        for m in range(25) :
          np.random.seed(m)
          angle = random.uniform(0,360)
          angles_negative.append(angle)
          
        # Generating the Positive Points 
        np.random.seed(99)
        for i in range(8) :
          radius = random.uniform(2,4.5)
          for k in angles_postive :
            list_one_postive = []
            x = value_center_x + radius*math.cos(k)
            y = value_center_y + radius*math.sin(k)
            list_one_postive.append(x)
            list_one_postive.append(y)
            y1.append(1)
            X1.append(list_one_postive)
          
        # Generating the Negative Points
        for i in range(8) :
          radius = random.uniform(5,6)
          np.random.seed(i)
          for k in angles_negative :
            list_one_negative = []
            x = value_center_x + radius*math.cos(k)
            y = value_center_y + radius*math.sin(k)
            list_one_negative.append(x)
            list_one_negative.append(y)
            y2.append(-1)
            X2.append(list_one_negative)
        X1 = X1[1:]
        X2 = X2[1:]
        print("Positive Points  - Circular Data")
        print(X1)
        print("Negative Points  - Circular Data")
        print(X2)
        return X1,y1,X2,y2
      
    

    def split_train_test(X1, y1, X2, y2):
        # Since 100 Instances is mentioned in Question, Considering 75 Instances as Training data
        # And Remaining data 25 Instances as Test data
        train_len = 3/4*(200)
        # Since we are Taking Half from Positive and Half from Negative points
        train_len = math.floor(train_len/2)
        # Creating Splits For Training Data
        X1_train = X1[:train_len]
        y1_train = y1[:train_len]
        X1_test = X1[train_len:]
        y1_test = y1[train_len:]
        # Creating Splits For Test Data
        X2_train = X2[:train_len]
        y2_train = y2[:train_len]
        X2_test = X2[train_len:]
        y2_test = y2[train_len:]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        print("Number of Points in Training Data are : ")
        print(len(X_train))
        return X_train, y_train, X_test, y_test
    
    

    def plot_margin(X1_train, X2_train, model,count_value,count):
        # plot the margin boundaries (for the linear separable and overlapping case)
        # plot the data points
        # plot the w^Tx+w_0 = 1, w^Tx+w_0 = -1, and w^Tx+w_0 = 0, lines
        # highlight the support vectors.

        # w.x + b = 0
        a0 = -1; a1 = f(a0, model.w, model.W0)
        b0 = 1; b1 = f(b0, model.w, model.W0)
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = -1
        a0 = -1; a1 = f(a0, model.w, model.W0, -1)
        b0 = 1; b1 = f(b0, model.w, model.W0, -1)
        pl.plot([a0,b0], [a1,b1], "k--")
        pl.plot(X1_train[:, 0], X1_train[:, 1], "bo")
        pl.plot(X2_train[:, 0], X2_train[:, 1], "co")
        pl.scatter(model.support_vector[:, 0], model.support_vector[:, 1], s=100, c="pink")
        pl.axis("tight")

        # w.x + b = 1
        a0 = -1;
        a1 = f(a0, model.w, model.W0, 1)
        b0 = 1;
        b1 = f(b0, model.w, model.W0, 1)
        pl.plot([a0, b0], [a1, b1], "k--")
        #pl.show()
        if count_value == 1 :
          if count == 1 : 
            pl.savefig("./figures/linear_sep_linear_kernel.png")
          if count == 2 : 
            pl.savefig("./figures/linear_sep_overlap_linear_kernel.png")
          if count == 3 : 
            pl.savefig("./figures/linear_kernel_circular_data.png")

        
        

    def plot_contour(X1_train, X2_train, model,count_value,count):
        # plot the contours of the classifier
        # create a meshgrid and for every point in the grid compute the output f(x) of the classifier
        # use the classifier's output to plot the contours on the meshgrid

        X1, X2 = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
        pl.scatter(model.support_vector[:,0], model.support_vector[:,1], s=100, c="g")

        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z_matrix = model.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z_matrix + 1, [0.0], colors='red', linewidths=2, origin='lower')
        pl.contour(X1, X2, Z_matrix, [0.0], colors='k', linewidths=2, origin='lower')
        pl.contour(X1, X2, Z_matrix - 1, [0.0], colors='red', linewidths=2, origin='lower')
        pl.plot(X1_train[:, 0], X1_train[:, 1], "bo")
        pl.plot(X2_train[:, 0], X2_train[:, 1], "co")
        pl.axis("tight")
        #pl.show()
        if count_value == 2 :
            if count == 1:
                pl.savefig("./figures/linear_sep_polynomial_kernel.png")
            if count == 2:
                pl.savefig("./figures/linear_sep_overlap_polynomial_kernel.png")
            if count == 3:
                pl.savefig("./figures/polynomial_kernel_circular_data.png")
        if count_value == 3 :
            if count == 1:
                pl.savefig("./figures/linear_sep_gaussian_kernel.png")
            if count == 2:
                pl.savefig("./figures/linear_sep_overlap_gaussian_kernel.png")
            if count == 3:
                pl.savefig("./figures/gaussian_kernel_circular_data.png")
        if count_value == 1:
            if count == 1:
                pl.savefig("./figures/linear_sep_linear_kernel.png")
            if count == 2:
                pl.savefig("./figures/linear_sep_overlap_linear_kernel.png")
            if count == 3:
                pl.savefig("./figures/linear_kernel_circular_data.png")
        pl.close()
    
    
    
    def f(x, w, b, c=0):
        # given x, return y such that [x,y] in on the line
        return (-w[0] * x - b + c) / w[1]
       
        

    def linear_svm(count):
        # 1. generate linearly separable data
        # 2. split the data into train and test sets
        # 3. create an support_vectorM object called model (uses linear kernel)
        # 4. train the support_vectorM using the fit function and the training data
        # 5. compute the classes of the model for the test data
        # 6. compute the accuracy of the model
        # 7. plot the training data points, and the margin.
        if count == 1 :
            X1, y1, X2, y2 = lin_separable_data()
            X_train, y_train, X_test, y_test = split_train_test(X1, y1, X2, y2)
            model = support_vectorM(linear_kernel)
        if count == 2 :
            X1, y1, X2, y2 = lin_separable_overlap_data()
            X_train, y_train, X_test, y_test = split_train_test(X1, y1, X2, y2)
            model = support_vectorM(linear_kernel)
        if count == 3 :
            X1, y1, X2, y2 = circular_data()
            X_train, y_train, X_test, y_test = split_train_test(X1, y1, X2, y2)
            model = support_vectorM(linear_kernel)
        model.fit(X_train, y_train)
        model_prediction(model,X_test,y_test)
        if count == 1 :
          plot_margin(X_train[y_train==1], X_train[y_train==-1], model,1,count)
        else :
          plot_contour(X_train[y_train==1], X_train[y_train==-1], model,1,count)
        pl.close()
        
        

    def kernel_svm(count):
        # 1. generate non-linearly separable data
        # 2. split the data into train and test sets
        # 3. create an support_vectorM object called model using an appropriate kernel function
        # 4. train the support_vectorM using the fit function and the training data
        # 5. compute the classes of the model for the test data
        # 6. compute the accuracy of the model
        # 7. plot the contours of the model's output using the plot_contour function
        if count == 1 :
            X1, y1, X2, y2 = lin_separable_overlap_data()
            X_train, y_train, X_test, y_test = split_train_test(X1, y1, X2, y2)
            model = support_vectorM(polynomial_kernel)
        if count == 2 :
            X1, y1, X2, y2 = lin_separable_data()
            X_train, y_train, X_test, y_test = split_train_test(X1, y1, X2, y2)
            model = support_vectorM(polynomial_kernel)
        if count == 3 :
                X1, y1, X2, y2 = circular_data()
                X_train, y_train, X_test, y_test = split_train_test(X1, y1, X2, y2)
                model = support_vectorM(polynomial_kernel)
        model.fit(X_train, y_train)
        model_prediction(model,X_test,y_test)
        plot_contour(X_train[y_train==1], X_train[y_train==-1], model,2,count)
        

    def soft_svm(count):
        # 1. generate linearly separable overlapping data
        # 2. split the data into train and test sets
        # 3. create an support_vectorM object called model (uses linear kernel, and the box penalty parameter)
        # 4. train the support_vectorM using the fit function and the training data
        # 5. compute the classes of the model for the test data
        # 6. compute the accuracy of the model
        # 7. plot the training data points, and the margin.
        if count == 1 :
            X1, y1, X2, y2 = lin_separable_data()
            X_train, y_train, X_test, y_test = split_train_test(X1, y1, X2, y2)
            model = support_vectorM(gaussian_kernel)
        if count == 2 :
            X1, y1, X2, y2 = lin_separable_overlap_data()
            X_train, y_train, X_test, y_test = split_train_test(X1, y1, X2, y2)
            model = support_vectorM(gaussian_kernel)
        if count == 3 :
            X1, y1, X2, y2 = circular_data()
            X_train, y_train, X_test, y_test = split_train_test(X1, y1, X2, y2)
            model = support_vectorM(gaussian_kernel)
        model.fit(X_train, y_train)
        model_prediction(model,X_test,y_test)
        plot_contour(X_train[y_train==1], X_train[y_train==-1], model,3,count)
        
    def model_prediction(model,X_test,y_test) :
        y_predict = model.predict(X_test)
        correct = np.sum(y_predict == y_test)
        accuracy = correct/len(y_predict)
        print("Accuracy is "+str(float(100*accuracy)))

        
    # after you have implemented the kernel and fit functions let us test the implementations
    # uncomment each of the following lines as and when you have completed their implementations.
    # In the Below Implementation for each support_vectorM for 3 different Datasets is Used.
    #print("\n")
    print("Plots for Linear support_vectorM for 3 different dataset are started generating ")
    linear_svm(1)
    linear_svm(2)
    linear_svm(3)
    print("\n")
    print("Plots for Linear support_vectorM for 3 different dataset are generated  Successfully")
    print("\n")
    print("Plots for Kernel support_vectorM for 3 different dataset are started generating ")
    kernel_svm(1)
    kernel_svm(2)
    kernel_svm(3)
    print("\n")
    print("Plots for Kernel support_vectorM for 3 different dataset are generated  Successfully")
    print("\n")
    print("Plots for Soft support_vectorM for 3 different dataset are started generating ")
    soft_svm(1)
    soft_svm(2)
    soft_svm(3)
    print("\n")
    print("Plots for Soft support_vectorM for 3 different dataset are generated  Successfully")			
			
