import numpy as np
from cvxpy import *


def primalSVM(xTr, yTr, C=1):
    """
    function (classifier,w,b) = primalSVM(xTr,yTr;C=1)
    constructs the SVM primal formulation and uses a built-in 
    convex solver to find the optimal solution. 
    
    Input:
        xTr   | training data (nxd)
        yTr   | training labels (nx1)
        C     | the SVM regularization parameter
    
    Output:
        fun   | usage: predictions=fun(xTe);
        wout  | the weight vector calculated by the solver
        bout  | the bias term calculated by the solver
    """
    N, d = xTr.shape
    y = yTr.flatten()
    # dummy code: example of establishing objective and constraints, and let the solver solve it.
    w = Variable(d)
    b = Variable(1)
    objective = sum_squares(w)
    #constraints = [w >= 0]
    
    loss = sum_entries(pos(1 - mul_elemwise(yTr, xTr*w - b)))
    
    #prob = Problem(Minimize(objective), constraints)
        
    #prob = Problem(Minimize(objective), constraints)
    
    prob = Problem(Minimize(loss + (1/C)*objective))

    prob.solve()
    wout = w.value
    bout = b.value
    # End of dummy code
    
    # TODO 1
    
    fun = lambda x: x.dot(wout) + bout
    return fun, wout, bout




