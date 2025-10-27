import numpy as np

def he_init(in_dim, out_dim):
        std = np.sqrt(2.0 / in_dim)
        W = np.random.normal(0.0, std, size=(out_dim, in_dim))
        b = np.zeros((out_dim, 1))
        return W, b

class Layer:
    def __init__(self, selfLen, beforeLen):
        self.selfLen = selfLen
        self.beforeLen = beforeLen
        (self.weights, self.biases) = he_init(self.beforeLen, self.selfLen)
        self.preData = np.zeros((self.selfLen,1), dtype=float)
        self.postData = np.zeros((self.selfLen,1), dtype=float)
        self.loss = 0
        # print(self.weights)


    
    


def LReLU(x):
    return (np.maximum(0.01*x, x), 0)

def softmaxEntropyLoss(X, ladel):
    exp_X = np.exp(X - np.max(X))
    output = exp_X / np.sum(exp_X)
    loss = np.log(output[ladel])
    return (output, loss)

def dLReLU(x):
    return np.where(x < 0, 0.01, 1.0)  # x < 0 时返回 0.01，否则返回 1.0

def Grads(L, Z):
    return Z - L



if __name__ == "__main__":
    iDataSet = np.load("handwritten_data.npy")
    oLabelSet = np.load("handwritten_labels.npy")
    setSize = oLabelSet.size

    layer1 = Layer(64, 1024)
    # relu
    layer2 = Layer(16, 64)
    # relu
    layer3 = Layer(16, 16)
    # relu
    layer4 = Layer(10, 16)
    # softmax
    layer5 = Layer(10, 10)
    data5 = np.zeros((10,1), dtype=float)
    data5_mean = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    # init weights and biases
    # ...
    
    preds = []
    expects = []
    losses = []
    dLs = []

    def layerForward(curLayer: Layer, iData: np.array, func):
        curLayer.preData = curLayer.weights @ iData + curLayer.biases
        (curLayer.postData, curLayer.loss) = func(curLayer.preData)

    eta = 10^3
    def layerBackward(curLayer: Layer, preLayer: Layer, lastGrads):
        newWeight = curLayer.weights - eta * lastGrads * preLayer.preData.T
        newBias = 0

    print(type(layer1.weights))
        
         
    i = 0
    for inputData in iDataSet :
        # forward
        flattenedInputData = inputData.flatten().reshape((1024,1))

        layerForward(layer1, flattenedInputData, LReLU)
        layerForward(layer2, layer1.postData, LReLU)
        layerForward(layer3, layer2.postData, LReLU)
        layerForward(layer4, layer3.postData, LReLU)
        layerForward(layer5, layer4.postData, softmaxEntropyLoss)

        
        (data5, loss) = softmaxEntropyLoss(layer4.postData,oLabelSet[i])
        preds.append(np.argmax(data5))

        one_hot = np.zeros((10,1))
        one_hot[oLabelSet[i]] = 1.0
        losses.append(loss)
        dLs.append(Grads(one_hot, data5))
        i+=1

        layerBackward()
    
    # w = 0
    # for d in losses:
    #     if d == -0.0:
    #         print(d)
    #         print(dLs[10])
    #     w+=1
    
    # print("预测值\t实际值\t相同?")
    # rs = oLabelSet.size
    # for i in range(rs):
    #     print(int(preds[i]),oLabelSet[i],(int(preds[i])==oLabelSet[i]),sep="\t")


    # === 反向传播（从后往前）===
    # grad_output: ∂L/∂z4 （因为 softmax + cross entropy 导数为 p - y）
    # 注意：layer4.data 是 relu(z4)，但我们已经用 grad_output = p - y


