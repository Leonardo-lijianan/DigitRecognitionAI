import pygame
import numpy as np


iDataSet = np.load("handwritten_data.npy")
oLabelSet = np.load("handwritten_labels.npy")

# ------------------- 数据收集模式 -------------------
 # 存储所有样本: 每个是 (image, label)
data = []
labels = []

data.extend(iDataSet)
labels.extend(oLabelSet)

rects = np.zeros((32,32), dtype=int)

pygame.init()
screen = pygame.display.set_mode((600,600))
pygame.display.set_caption("Hello")

light_black = (25, 25, 25)
white = (255,255,255)
blue = (0,0,255)

# 字体显示提示
font = pygame.font.Font("ch_zn.ttf", 36)

size = 15
start = 5
gap = 1
running = True
mouseDowning = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouseDowning = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouseDowning = False
        elif event.type == pygame.KEYDOWN:

            # 按 0~9 来标记当前画的是什么数字
            if pygame.K_0 <= event.key <= pygame.K_9:
                label = event.key - pygame.K_0
                # 保存当前图像（归一化到 0-1，或 0-255）
                sample = rects.copy().astype(np.float32)  # 32x32
                data.append(sample)
                labels.append(label)
                print(f"✅ 保存数字 '{label}', 当前数据量: {len(labels)}")

                if data:
                    data_array = np.array(data)  # (N, 32, 32)
                    labels_array = np.array(labels)  # (N,)
                    np.save("handwritten_data.npy", data_array)
                    np.save("handwritten_labels.npy", labels_array)
                    print(f"💾 数据已保存: {len(labels)} 个样本")
                    rects.fill(0)
                else:
                    print("⚠️ 无数据可保存")
            # 按 S 保存到文件
            elif event.key == pygame.K_s:
                if data:
                    data_array = np.array(data)  # (N, 32, 32)
                    labels_array = np.array(labels)  # (N,)
                    np.save("handwritten_data.npy", data_array)
                    np.save("handwritten_labels.npy", labels_array)
                    print(f"💾 数据已保存: {len(labels)} 个样本")
                    rects.fill(0)
                else:
                    print("⚠️ 无数据可保存")
            # 按 C 清空画布
            elif event.key == pygame.K_c:
                rects.fill(0)
                print("🧹 画布已清空")

    if mouseDowning:
        (mx, my) = pygame.mouse.get_pos()
        j = (mx - start) // (size + gap)
        i = (my - start) // (size + gap)
        if((0 <= i < 32) & (0 <= j < 32)):
            rects[i, j] = 1

    screen.fill(light_black)

    for i in range(32):
        for j in range(32):
            if rects[i][j] == 1:
                pygame.draw.rect(screen, white, (start+j*(size+gap), start+i*(size+gap), size, size))
    
    # 显示提示
    tip = font.render("按 0-9 标记数字, S 保存, C 清空", True, (100, 200, 100))
    screen.blit(tip, (10, 550))

    pygame.display.flip()

pygame.quit()