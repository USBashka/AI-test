import numpy as np
import pygame



np.random.seed(1)

def relu(x):
    return (x > 0) * x

def relu2deriv(output):
    return output > 0

streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1]])

walk_vs_stop = np.array([[1, 1, 0, 1]]).T

alpha = 0.2
hidden_size = 4

weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

for iter in range(60):
    layer_2_error = 0
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)

        layer_2_delta = layer_2 - walk_vs_stop[i:i+1]
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)

def neural_network(input):
    layer_0 = input
    layer_1 = relu(np.dot(layer_0, weights_0_1))
    layer_2 = np.dot(layer_1, weights_1_2)
    return layer_2


# Инициализация pygame
pygame.init()

# Цвета
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

# Размеры экрана
WIDTH = 720
HEIGHT = 600

# Создание экрана
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chapter 6 Project")

# Переменная для отслеживания текущего состояния светофора
current_colors = [0, 0, 0]

font = pygame.font.Font(None, 36)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Обработка клика мышки
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if 150 <= mouse_pos[0] <= 250:
                if 50 <= mouse_pos[1] <= 250:
                    current_colors[0] = not current_colors[0]
                elif 250 <= mouse_pos[1] <= 450:
                    current_colors[1] = not current_colors[1]
                elif 450 <= mouse_pos[1] <= 650:
                    current_colors[2] = not current_colors[2]

    # Очистка экрана
    screen.fill(BLACK)

    # Рисование светофора
    pygame.draw.rect(screen, "gray", (150, 50, 100, 200))
    pygame.draw.rect(screen, "gray", (150, 250, 100, 200))
    pygame.draw.rect(screen, "gray", (150, 450, 100, 200))

    # Включение цвета светофора
    if current_colors[0]:
        pygame.draw.circle(screen, RED, (200, 100), 40)
    else:
        pygame.draw.circle(screen, BLACK, (200, 100), 40)
    if current_colors[1]:
        pygame.draw.circle(screen, YELLOW, (200, 300), 40)
    else:
        pygame.draw.circle(screen, BLACK, (200, 300), 40)
    if current_colors[2]:
        pygame.draw.circle(screen, GREEN, (200, 500), 40)
    else:
        pygame.draw.circle(screen, BLACK, (200, 500), 40)

    if neural_network(current_colors) > 0.5:
        text = font.render("Идти", True, "white")
    else:
        text = font.render("Стоять", True, "white")
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT - 30))
    screen.blit(text, text_rect)

    # Обновление экрана
    pygame.display.flip()

# Завершение pygame
pygame.quit()
