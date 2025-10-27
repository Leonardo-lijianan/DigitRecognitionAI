import pygame
import numpy as np

rects = np.zeros((32,32), dtype=int)
iDataSet = np.load("handwritten_data.npy")
oLabelSet = np.load("handwritten_labels.npy")
setSize = oLabelSet.size

summary = np.zeros((10,1))
for num in oLabelSet:
    summary[num] +=1

print(summary)



pygame.init()
screen = pygame.display.set_mode((600,600))
pygame.display.set_caption("Hello")

light_black = (25, 25, 25)
white = (255,255,255)
blue = (0,0,255)

# 字体显示提示
font = pygame.font.Font("ch_zn.ttf", 36)

index = 0

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


            if event.key == pygame.K_LEFT:
                if index > 0:
                    index-=1
                    rects = iDataSet[index]

            elif event.key == pygame.K_RIGHT:
                if index < setSize-1:
                    index+=1
                    rects = iDataSet[index]

    screen.fill(light_black)

    for i in range(32):
        for j in range(32):
            if rects[i][j] == 1:
                pygame.draw.rect(screen, white, (start+j*(size+gap), start+i*(size+gap), size, size))
    
    # 显示提示
    tip = font.render("目前"+index.__str__()+"的图形代表的数字："+oLabelSet[index].__str__(), True, (100, 200, 100))
    screen.blit(tip, (10, 550))

    pygame.display.flip()

pygame.quit()