import numpy as np
import pygame
import random


def neural_network(input, weights):
    return np.clip(input.dot(weights), -1, 1)

weights = np.array([0.5, 0.5, 0.5])
data_points = []

def train():
    for point in data_points:
        input = np.array([point[0], point[1], 1])
        pred = neural_network(input, weights)
        delta = pred - point[2]
        for i in range(len(weights)):
            weight_delta = delta * input[i] * 0.1
            weights[i] -= weight_delta


# Инициализация Pygame
pygame.init()

# Размеры окна и массива фона
window_size = (512, 512)
background_size = (64, 64)

# Создание окна
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Chapter 5 Project")


# Основной цикл программы
running = True
while running:
    train()
    print(f"Weights: {weights}")
    # Отрисовка фона
    for y in range(background_size[1]):
        for x in range(background_size[0]):
            pred = neural_network(np.array([x/background_size[0], y/background_size[1], 1]), weights)
            if pred > 0:
                color = [0, 0, np.clip(pred*255, 0, 255)]
            else:
                color = [np.clip(-pred*255, 0, 255), np.clip(-pred*127, 0, 255), 0]
            pygame.draw.rect(screen, color, (x * 8, y * 8, 8, 8))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in (1, 3):
                x, y = event.pos
                if 0 <= x < window_size[0] and 0 <= y < window_size[1]:
                    if event.button == 1:
                        data_points.append([x/512, y/512, -1])
                    elif event.button == 3:
                        data_points.append([x/512, y/512, 1])
    
    for point in data_points:
        # Создание кружка
        pygame.draw.circle(screen, (204, 85, 0) if point[2] == -1 else (0, 0, 102), (point[0]*512, point[1]*512), 16)
    
    # Обновление экрана
    pygame.display.flip()

# Завершение работы Pygame
pygame.quit()