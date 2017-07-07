# Kernels.py:
A function to compute the Kernel Matrix

# PrimalSVM.py
Simply the implementation of linear SVM (funtion primalSVM(xTr, yTr, C=1))

# Kernelized_SVM_Solver
*dualqp* solves the dual problem user [cvxpy](http://www.cvxpy.org/en/latest/install/) solver package
*recoverBias* is a function to recover the bias (b) after finding the weights that optimize the dual problem.
*dualSVM* uses the weights and bias to construct the classfier.  
# Dataio.py
Use this code to test the classifiers on the simulated data. 
# Simulate_Data.py
Functions to simulate linearly separable data and spiral data. (imported from my machine learning class at Cornell)
