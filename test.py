# import numpy as np
# import matplotlib.pyplot as plt

    ###################################### test Demo  ####################333
    # weights1 = np.array([
    #     [1, 2,  3,  4], # 1-4+9-16=-10
    #     [5, 6,  7,  8],
    #     [9, 10, 11, 12]
    # ])

    # flattenedInputData = np.array([
    #     [1], 
    #     [-2],
    #     [3],
    #     [-4]
    # ])

    # biases1 = np.array([
    #     [0.1],
    #     [-0.2],
    #     [0.3]
    # ])

    # data1 = relu(weights1 @ flattenedInputData + biases1)

    # print(data1)

    # 
    # np.random.normal(0, 0.01, size=(3, 4))

# # data = np.zeros((32, 32), dtype=int)

# # for i in range(32):
# #         for j in range(32):
# #             data[i, j] = i*1000 + j
# # print(data)

# # 读取图像数据
# data = np.load("handwritten_data.npy")     # shape: (N, 32, 32)
# labels = np.load("handwritten_labels.npy") # shape: (N,)


# # 显示第0张图
# plt.imshow(data[0], cmap='gray')  # cmap='gray' 表示灰度图
# plt.title(f"Label: {labels[0]}")  # 标题显示标签
# plt.show()

import numpy as np

def correcten():
    # 1. 加载原始数据
    labels = np.load("handwritten_labels.npy")
    corrected_last_label = 8
    index = -1

    print("原始的标签:", labels[index])  # 查看值
    labels[index] = corrected_last_label  # 修改元素
    print("修正后的标签:", labels[index])
    np.save("handwritten_labels.npy", labels)

labels = np.load("handwritten_labels.npy")
times = np.zeros((10,1))

for num in labels:
    times[num]+=1

print(times.tolist())
