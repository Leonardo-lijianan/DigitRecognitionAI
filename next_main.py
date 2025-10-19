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

class Trainer:

    def __label2OneHot(self, label: int):
        onehot = np.zeros((10,1))
        onehot[label] = 1
        return onehot

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
            #print(inputData)
            self.layers[0].postData = inputData.flatten().reshape((1024,1))

        for i in range(len(self.layers)-1):
            self.layers[i+1].preData = self.layers[i+1].weights @ self.layers[i].postData + self.layers[i+1].biases
            if self.actFuncs[i] == "LReLU":
                self.layers[i+1].postData = self.__LReLU(self.layers[i+1].preData)
            elif self.actFuncs[i] == "softmax":
                self.layers[i+1].postData = self.__softmax(self.layers[i+1].preData)
            else:
                print("error!!!!!!!!11")

        #print(self.layers[-1].postData)
        return np.argmax(self.layers[-1].postData.flatten())

    def backward(self):
        self.loss = self.__crossEntropyLoss(self.layers[-1].postData, self.oLabelSet[self.dataSetIndex])

        eta = 1e-3 # learning rate
        weightGrads = np.ndarray
        biasGrads = np.ndarray
        nextGrads = np.ndarray

        # dLossDZc = self.layers[-1].postData - self.__label2OneHot(self.oLabelSet[self.dataSetIndex])
        nextGrads = self.layers[-1].postData - self.__label2OneHot(self.oLabelSet[self.dataSetIndex])

        for i in range(len(self.layers)-1):
            weightGrads = nextGrads @ self.layers[-(2+i)].postData.T
            biasGrads = nextGrads * 1 # x求导为1
            #print(self.layers[-(1+i)].weights.T.shape, nextGrads.shape, self.__dLReLU(self.layers[-(2+i)].preData).shape, end="\n", sep=" ")
            nextGrads = (self.layers[-(1+i)].weights.T @ nextGrads) * self.__dLReLU(self.layers[-(2+i)].preData)
        
            self.layers[-(1+i)].weights += -eta*weightGrads
            self.layers[-(1+i)].biases += -eta*biasGrads



        # self.layers[-1].postData[5] = -1.0
        # dX = self.__dLReLU(self.layers[-1].postData)
        # print(self.layers[-1].postData,dX,sep="\n666\n")
        
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
        print(layerNumber[0][0],self.actFuncs)
        for i in range(layerNumber[0][0]):
            weights = np.load("modelParas/weightsSet/layer"+str(i)+"_weights.npy")
            biases = np.load("modelParas/biasesSet/layer"+str(i)+"_biases.npy")

            nowLayer = Layer(weights.shape[0],weights.shape[1])
            nowLayer.weights = weights
            nowLayer.biases = biases
            self.layers.append(nowLayer)

            print(i,weights.shape,biases.shape)


if __name__ == "__main__":
    def test():
        trainer = Trainer()
        trainer.loadModelParas()
        trainer.loadData("handwritten_data.npy", "handwritten_labels.npy")

        trainer.forward()
        print(trainer.loss)

    def forwards():
        print("hello world!")
        trainer = Trainer()
        # trainer.loadModelParas()
        trainer.loadData("handwritten_data.npy", "handwritten_labels.npy")
        trainer.loadLayer([1024, 64, 16, 16, 10])
        trainer.loadActFunc(["LReLU", "LReLU", "LReLU", "softmax"])

        for i in range(1000):
            print("The ",i,"th TRAIN")
            for j in range(len(trainer.oLabelSet)-1):
                if trainer.oLabelSet[j] !=  -1 :
                    trainer.dataSetIndex = j # 忘记这个了，AI只学5(第零号数据是5),笑死！
                    trainer.forward()
                    print("before Backward, Loss: ",trainer.loss)
                    if trainer.loss >= 1.0e-5:
                        trainer.backward()
                        trainer.forward()
                        print("after Backward, Loss: ",trainer.loss)
                    else:
                        trainer.oLabelSet[j] = -1

        trainer.saveModelParas()
        # print(trainer.layers[-1].postData)
        # print(trainer.loss)

    # test()
    forwards()