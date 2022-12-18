import numpy as np


#================================================================================================
#================================================================================================
def computeError(y,yPred,w):
    # y : actual value
    # yPred : predicted value
    # w : weights of samples
    # error = sum of weight of misclassified samples / sum of weight of all samples
    return np.sum(w[y!=yPred])/np.sum(w)

def computeAlpha(error):
    # compute alpha (weight) for the current weak classifier m
    return 0.5*np.log((1-error)/error)

def updateWeights(y,yPred,w,alpha):
    # update weights of samples
    # w = w*exp(-alpha*y*yPred)
    w = w * np.exp(-alpha * (np.not_equal(y,yPred)).astype(int))
    return w


#================================================================================================
#================================================================================================

class DecisionStump:
    # Stump is a weak classifier
    # It is a decision tree with only one split (depth = 1)
    # It is used to classify samples into two classes
    def __init__(self):
        return

    def classificationError(self,y,yPred):
        # y : target variable
        # yPred : predicted value
        return len(y[y!=yPred])/len(y)

    def predict(self,X):
        # X : feature matrix
        # n : number of samples
        # m : number of features
        return


#================================================================================================
#================================================================================================

class AdaBoost:

    #weakClassifier : weak classifier (Gm)
    #M : number of boosting iterations

    def __init__(self):
        self.alphas = []
        self.weakClassifiers = []
        self.M = None
        self.trainingErrors = []
        self.predictionErrors = []

    def fit(self,X,y,M = 100):
        return 
        # X : Independent variables
        # y : target variable which is to be predicted
        # M : number of boosting iterations default = 100

        self.alphas = []
        self.trainingErrors = []
        self.M = M
        
        
        for m in range(M):
            #set weights of samples
            if m == 0:
                #first iteration set all weights to 1/n (equal weights)
                w = np.ones( len(y) ) / len(y)
            else:
                w = updateWeights(y,yPred,w,alpha)
            
            # train weak classifier Gm
            #implement your own weak classifier
            weakClassifier = DecisionTreeClassifier(max_depth=1)
            weakClassifier.fit(X,y,sample_weight=w)
            yPred = weakClassifier.predict(X)
            error = computeError(y,yPred,w)
            alpha = computeAlpha(error)
            w = updateWeights(y,yPred,w,alpha)
            self.alphas.append(alpha)
            self.weakClassifiers.append(weakClassifier)
            self.trainingErrors.append(error)


#================================================================================================
#================================================================================================