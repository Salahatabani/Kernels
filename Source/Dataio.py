import numpy as np
from PrimalSVM import *
from Simulate_Data import *
from Kernelized_SVM_Solver import *

# Test the linear SVM function on linearly separable data
print 'Linearly Separable Data with Linear SVM'
xTr,yTr=genrandomdata()
fun,w,b=primalSVM(xTr,yTr,C=20)
err=np.mean(np.array((np.sign(fun(xTr))).flatten()!=yTr))
print "Training error: %2.1f%%" % (err*100) 

# Now let's try linear SVM function on spiral data
print 'Spiral Data with Linear SVM'
xTr,yTr,xTe,yTe=spiraldata()
fun,w,b=primalSVM(xTr,yTr,C=10)
err=np.mean(np.array(np.sign(fun(xTr))).flatten()!=yTr)
print("Training error: %2.1f%%" % (err*100))

# Now let's try RBF-kernelied SVM function on spiral data
print 'Spiral Data with RBF-kernelied SVM'
xTr,yTr,xTe,yTe=spiraldata()
C=10.0
sigma=0.25
ktype="rbf"
svmclassify=dualSVM(xTr,yTr,C,ktype,sigma)

# compute training and testing error
predsTr=svmclassify(xTr)
trainingerr=np.mean(np.sign(predsTr)!=yTr)
print("Training error: %2.4f" % trainingerr)

predsTe=svmclassify(xTe)
testingerr=np.mean(np.sign(predsTe)!=yTe)
print("Testing error: %2.4f" % testingerr)
