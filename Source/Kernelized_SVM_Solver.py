import numpy as np
from cvxpy import *
from Kernels import *
from PrimalSVM import *

def dualqp(K,yTr,C):
    """
    function alpha = dualqp(K,yTr,C)
    constructs the SVM dual formulation and uses a built-in 
    convex solver to find the optimal solution. 
    
    Input:
        K     | the (nxn) kernel matrix
        yTr   | training labels (nx1)
        C     | the SVM regularization parameter
    
    Output:
        alpha | the calculated solution vector (nx1)
    """
    y = yTr.flatten()
    N, _ = K.shape
    alpha = Variable(N)
    
    Q = mul_elemwise(np.dot(yTr,yTr.T),K)
    objective = Minimize(0.5*quad_form(mul_elemwise(yTr,alpha),K)-sum_entries(alpha))
    constraints = [alpha>=0,alpha<=C,sum_entries(mul_elemwise(yTr,alpha))==0]
    prob = Problem(objective, constraints)
    prob.solve()
    return np.array(alpha.value).flatten()


def recoverBias(K,yTr,alpha,C):
    """
    function bias=recoverBias(K,yTr,alpha,C);
    Solves for the hyperplane bias term, which is uniquely specified by the 
    support vectors with alpha values 0<alpha<C
    
    INPUT:
    K : nxn kernel matrix
    yTr : 1xn input labels
    alpha  : nx1 vector of alpha values
    C : regularization constant
    
    Output:
    bias : the scalar hyperplane bias of the kernel SVM specified by alphas
    """


    alpha = alpha.reshape(alpha.shape[0],1)
    yTr = yTr.reshape(yTr.shape[0],1)
    
    su = np.sum(alpha*yTr*K,axis=0)
    su = su.reshape(su.shape[0],1)
    
    index = np.argmin(np.absolute(alpha-(C/2)))
    bias = yTr[index]-su[index]


    return bias

def dualSVM(xTr,yTr,C,ktype,lmbda):
    """
    function classifier = dualSVM(xTr,yTr,C,ktype,lmbda);
    Constructs the SVM dual formulation and uses a built-in 
    convex solver to find the optimal solution. 
    
    Input:
        xTr   | training data (nxd)
        yTr   | training labels (nx1)
        C     | the SVM regularization parameter
        ktype | the type of kernelization: 'rbf','polynomial','linear'
        lmbda | the kernel parameter - degree for poly, inverse width for rbf
    
    Output:
        svmclassify | usage: predictions=svmclassify(xTe);
    """
    #svmclassify = lambda x: x #Dummy code
    

    K = computeK(kerneltype=ktype, X=xTr, Z=xTr, kpar=lmbda)
    alpha = dualqp(K,yTr,C)
    b = recoverBias(K,yTr,alpha,C)
    alpha = alpha.reshape(alpha.shape[0],1)
    yTr = yTr.reshape(yTr.shape[0],1)
    svmclassify = lambda x: np.sum(alpha*yTr*computeK(kerneltype=ktype, X=xTr, Z=x, kpar=lmbda),axis=0)
    return svmclassify
