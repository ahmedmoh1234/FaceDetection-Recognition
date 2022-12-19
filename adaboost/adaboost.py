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

def integralImage(img):
    # img : image
    # integralImage : integral image of the input image
    # integralImage[i,j] = sum of all pixels in the rectangle from (0,0) to (i,j)
    # test time complexity of this function


    #np.cumsum() is much faster than the for loop
    # intImg = np.zeros(img.shape)
    # for i in range(0,img.shape[0]):
    #     for j in range(0,img.shape[1]):
    #         if i==0 and j==0:
    #             intImg[i,j] = img[i,j]
    #         elif i==0 and j!=0:
    #             intImg[i,j] = img[i,j] + intImg[i,j-1]
    #         elif i!=0 and j==0:
    #             intImg[i,j] = img[i,j] + intImg[i-1,j]
    #         else:
    #             intImg[i,j] = img[i,j] + intImg[i-1,j] + intImg[i,j-1] - intImg[i-1,j-1]
    # return intImg

    # intImg2 = np.zeros(img.shape)
    # for i in range(1,img.shape[0]):
    #     for j in range(1,img.shape[1]):
    #             intImg2[i,j] = img[i,j] + intImg2[i-1,j] + intImg2[i,j-1] - intImg2[i-1,j-1]

    # return np.cumsum(np.cumsum(img,axis=0),axis=1)


    #calculate integral image with first row and first column as zeros
    #because this is a convention
    return np.cumsum(np.cumsum(np.pad(img,((1,0),(1,0)),'constant'),axis=0),axis=1)

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