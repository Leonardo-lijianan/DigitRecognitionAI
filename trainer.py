import numpy as np
from typing import List

class Layer:
    def __he_init(self, in_dim, out_dim):
        std = np.sqrt(2.0 / in_dim)
        W = np.random.normal(0.0, std, size=(out_dim, in_dim))
        b = np.zeros((out_dim, 1))
        return W, b

    def __init__(self, selfLen, beforeLen):
        self.selfLen = selfLen
        self.beforeLen = beforeLen
        (self.weights, self.biases) = self.__he_init(self.beforeLen, self.selfLen)
        self.preData = np.zeros((self.selfLen,1), dtype=float)
        self.postData = np.zeros((self.selfLen,1), dtype=float)

### developing...###
class Connection:
    def __init__(self):
        pass

class InputLayer(Layer):
    def __init__(self, selfLen, beforeLen):
        super().__init__(selfLen, beforeLen)

class HiddenLayer(Layer):
    def __init__(self, selfLen, beforeLen):
        super().__init__(selfLen, beforeLen)

class OutputLayer(Layer):
    def __init__(self, selfLen, beforeLen):
        super().__init__(selfLen, beforeLen)
### developing...###


class Trainer:

    def __label2OneHot(self, label: int):
        onehot = np.zeros((10,1))
        onehot[label] = 1
        return onehot
    
    def __ReLU(self, X):
        return np.maximum(0, X)
    
    def __dReLU(self, X):
        return np.where(X<0, 0, 1)

    def __LReLU(self, X):
        return np.maximum(0.01*X, X)
    
    def __dLReLU(self, X):
        return np.where(X<0, 0.01, 1)
    
    def __softmax(self, X):
        exp_X = np.exp(X - np.max(X))
        return exp_X / np.sum(exp_X)
    
    def __crossEntropyLoss(self, Yhat, label: int):
        return -np.log(Yhat[label] + 1e-8)  # 加小数防止 log(0) # one-hot vector only one one.

    def __init__(self):
        self.layers: List[Layer] = []
        self.actFuncs: List[str] = []
        self.iDataSet = []
        self.oLabelSet = []
        self.loss = 10
        self.dataSetIndex = 0
        self.losses = [100] *10

    def loadData(self,sourceDataPath: str, labelDataPath: str):
        self.iDataSet = np.load(sourceDataPath)
        self.oLabelSet = np.load(labelDataPath)

    def loadLayer(self, lens: List[int]):
        for i in range(len(lens)):
            self.layers.append(Layer(lens[i],( lens[i] if i==0 else lens[i-1] )))

    def loadActFunc(self, funcs: List[str]):
        self.actFuncs = funcs

    def forward(self, inputData: np.ndarray=np.array([])):
        if inputData.size == 0:
            self.layers[0].postData = self.iDataSet[self.dataSetIndex].flatten().reshape((1024,1))
        else:
            self.layers[0].postData = inputData.flatten().reshape((1024,1))

        for i in range(len(self.layers)-1):
            self.layers[i+1].preData = self.layers[i+1].weights @ self.layers[i].postData + self.layers[i+1].biases
            if self.actFuncs[i] == "LReLU":
                self.layers[i+1].postData = self.__LReLU(self.layers[i+1].preData)
            elif self.actFuncs[i] == "softmax":
                self.layers[i+1].postData = self.__softmax(self.layers[i+1].preData)
            else:
                print("error!!!!!!!!")

        #print(self.layers[-1].postData)
        return np.argmax(self.layers[-1].postData.flatten())

    def backward(self):
        self.loss = self.__crossEntropyLoss(self.layers[-1].postData, self.oLabelSet[self.dataSetIndex])[0]
        if self.loss < self.losses[self.oLabelSet[self.dataSetIndex]]:
            self.losses[self.oLabelSet[self.dataSetIndex]] = (float)(self.loss)

        eta = 1e-3 # learning rate
        weightGrads = np.ndarray
        biasGrads = np.ndarray
        nextGrads = np.ndarray

        nextGrads = self.layers[-1].postData - self.__label2OneHot(self.oLabelSet[self.dataSetIndex])

        for i in range(len(self.layers)-1):
            weightGrads = nextGrads @ self.layers[-(2+i)].postData.T
            biasGrads = nextGrads * 1 # x求导为1
            nextGrads = (self.layers[-(1+i)].weights.T @ nextGrads) * self.__dLReLU(self.layers[-(2+i)].preData)
        
            self.layers[-(1+i)].weights += -eta*weightGrads
            self.layers[-(1+i)].biases += -eta*biasGrads
        
    def saveModelParas(self):
        layerNumber = np.zeros((1,1), dtype=int)
        layerNumber[0][0] = len(self.layers)
        np.save("modelParas/layer_number.npy",layerNumber)
        np.save("modelParas/actFuncs.npy",np.array(self.actFuncs))
        for i in range(len(self.layers)):
            np.save("modelParas/weightsSet/layer"+str(i)+"_weights.npy",self.layers[i].weights)
            np.save("modelParas/biasesSet/layer"+str(i)+"_biases.npy",self.layers[i].biases)

    def loadModelParas(self):
        layerNumber = np.load("modelParas/layer_number.npy")
        self.actFuncs = np.load("modelParas/actFuncs.npy")
        for i in range(layerNumber[0][0]):
            weights = np.load("modelParas/weightsSet/layer"+str(i)+"_weights.npy")
            biases = np.load("modelParas/biasesSet/layer"+str(i)+"_biases.npy")

            nowLayer = Layer(weights.shape[0],weights.shape[1])
            nowLayer.weights = weights
            nowLayer.biases = biases
            self.layers.append(nowLayer)