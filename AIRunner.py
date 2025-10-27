import trainer as tr
import numpy as np
import pygame

rects = np.zeros((32,32), dtype=int)
          #########
size = 15 # The   #
start = 5 # Rects #
gap = 1   # Paras #
          #########

light_black = (25, 25, 25)
white = (255,255,255)
blue = (0,0,255)

guessNumber = 0

if __name__ == "__main__":
    trainer = tr.Trainer()
    trainer.loadModelParas()
    trainer.loadData("sourceData/handwritten_data.npy", "sourceData/handwritten_labels.npy")

    pygame.init()
    screen = pygame.display.set_mode((600,600))
    pygame.display.set_caption("AIRunner")
    # 字体显示提示
    font = pygame.font.Font("ch_zn.ttf", 36)    

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
                if event.key == pygame.K_s:
                    guessNumber = trainer.forward(np.array(rects))
                    print(guessNumber)
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
        tip = font.render("你写的数字是："+str(guessNumber), True, (100, 200, 100))
        screen.blit(tip, (10, 550))

        pygame.display.flip()

    pygame.quit()