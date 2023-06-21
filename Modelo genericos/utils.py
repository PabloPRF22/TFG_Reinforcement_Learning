import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Tuple
import numpy as np

def plot_mean_reward_per_algorithm(
        results: List[Tuple[str, float]],
        env_name: str,
        savepath: Path,
        reward_threshold: float = 200,
        bar_width: float = 0.3,
        bar_distance: float = 0.1
):
    names, values = zip(*results)

    fig, ax = plt.subplots(figsize=(20, 12))

    n_bars = len(results)
    x_positions = [i * (bar_width + bar_distance) for i in range(n_bars)]
    bars = ax.bar(x_positions, values, width=bar_width,
                  color=plt.cm.tab20.colors[:n_bars])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(names)
    ax.axhline(y=reward_threshold, color='red', linestyle='--', linewidth=1.5)

    ax.axhline(y=0, color='black', linewidth=0.8)

    ax.set_ylabel('Mean Reward')
    ax.set_title(f'Mean Reward per Algorithm on the {env_name} environment')

    plt.savefig(savepath)


import numpy as np



def plot_train_results(rewards, savepath, titulo, env):
    # Crear una figura.
    fig = plt.figure()
    # Recorrer cada lista de recompensas y plotearla en el gráfico.
    for i, rewards_list in enumerate(rewards):
        # Obtener la longitud de la lista de recompensas actual.
        current_length = len(rewards_list)

        # Generar un array de episodios para el eje x de la lista de recompensas actual.
        current_episodes = np.arange(current_length)
        # Obtener los índices de los elementos que se seleccionarán (1 de cada 20)
        selected_indices = np.arange(0, current_length, 20)

        # Obtener los valores de recompensas correspondientes a los índices seleccionados
        selected_rewards = [rewards_list[j] for j in selected_indices]

        # Plotear las recompensas.
        plt.plot(selected_indices, selected_rewards, label=titulo[i])

    # Añadir leyenda al gráfico.
    plt.legend()

    # Establecer los títulos de los ejes y del gráfico.
    plt.xlabel('Episodios')
    plt.ylabel('Recompensas')
    plt.title('Perdida por algoritmo de aprendizaje por refuerzo')

    # Guardar la figura en el archivo especificado.
    plt.savefig(savepath)

def plot_train_eval_results(rewards, savepath, titulo, env):
    # Crear una figura.
    fig = plt.figure()
    # Recorrer cada lista de recompensas y plotearla en el gráfico.
    for i, rewards_list in enumerate(rewards):
        # Obtener la longitud de la lista de recompensas actual.
        current_length = len(rewards_list)

        # Como se recogen las recompensas cada 100 episodios, generamos el array de episodios multiplicando por 100.
        current_episodes = np.arange(current_length) * 100

        # Plotear las recompensas.
        plt.plot(current_episodes, rewards_list, label=titulo[i])

    # Añadir leyenda al gráfico.
    plt.legend()

    # Establecer los títulos de los ejes y del gráfico.
    plt.xlabel('Episodios')
    plt.ylabel('Recompensas')
    plt.title('Recompensa media por algoritmo de aprendizaje por refuerzo')

    # Guardar la figura en el archivo especificado.
    plt.savefig(savepath)



