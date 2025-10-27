import trainer as tr

def test():
    trainer = tr.Trainer()
    trainer.loadModelParas()
    trainer.loadData("handwritten_data.npy", "handwritten_labels.npy")

    trainer.forward()
    print(trainer.loss)

def forwards():
    trainer = tr.Trainer()
    # trainer.loadModelParas()
    trainer.loadData("handwritten_data.npy", "handwritten_labels.npy")
    trainer.loadLayer([1024, 64, 16, 16, 10])
    trainer.loadActFunc(["LReLU", "LReLU", "LReLU", "softmax"])

    for i in range(1000):
        print("The ",i,"th TRAIN\n")
        for j in range(len(trainer.oLabelSet)-1):
            if trainer.oLabelSet[j] !=  -1 :
                trainer.dataSetIndex = j # 忘记这个了，AI只学5(第零号数据是5),笑死！
                trainer.forward()
                # print("before Backward, Loss: ",trainer.loss)
                if trainer.loss >= 1.0e-5:
                    trainer.backward()
                    trainer.forward()
                    # print("after Backward, Loss: ",trainer.loss)
                else:
                    trainer.oLabelSet[j] = -1
        print(trainer.losses)

    trainer.saveModelParas()

if __name__ == "__main__":
    #test()
    forwards()