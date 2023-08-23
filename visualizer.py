import matplotlib.pyplot as plt
import numpy as np


def visualize(layers, weights):
    plt.figure(figsize=(10, 6))
    
    for i, layer in enumerate(layers):
        for j in range(layer):
            neuron_x = i
            neuron_y = j / layer  # Распределяем нейроны по вертикали

            if i < len(layers) - 1:
                next_layer = layers[i + 1]
                next_weights = weights[i]
                for k in range(next_layer):
                    next_neuron_x = i + 1
                    next_neuron_y = k / next_layer
                    next_weight = next_weights[j][k]
                    if next_weight > 0:
                        plt.plot([neuron_x, next_neuron_x], [neuron_y, next_neuron_y], color='green', lw=np.arctan(next_weight)*8)
                    elif next_weight < 0:
                        plt.plot([neuron_x, next_neuron_x], [neuron_y, next_neuron_y], color='red', lw=-np.arctan(next_weight)*8)
            
            plt.scatter(neuron_x, neuron_y, color='blue', s=300, zorder=10)

    plt.title('Визуализация Нейронной Сети', fontsize=16)
    plt.xlabel('Слои', fontsize=14)
    plt.ylabel('Нейроны', fontsize=14)
    plt.xticks(range(len(layers)), [f'Слой {i+1}' for i in range(len(layers))], fontsize=12)
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    neural_network_layers = [2, 3, 1]  # Входной слой, скрытый слой, выходной слой
    neural_network_weights = [[[6, 2, -1],
                               [-8, 4, 1]],
                               [[3], [7], [-0.5]]]
    visualize(neural_network_layers, neural_network_weights)
