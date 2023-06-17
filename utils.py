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
    # Obtener la longitud de la lista de recompensas más larga.
    max_length = max(len(x) for x in rewards)

    # Generar un array de episodios para el eje x basado en la longitud máxima y el intervalo dado.
    episodes = np.arange(max_length) * 2

    # Recorrer cada lista de recompensas y plotearla en el gráfico.
    for i, rewards_list in enumerate(rewards):
        # Obtener la longitud de la lista de recompensas actual.
        current_length = len(rewards_list)

        # Generar un array de episodios para el eje x de la lista de recompensas actual.
        current_episodes = np.arange(current_length)

        # Plotear las recompensas.
        plt.plot(current_episodes, rewards_list, label=titulo[i])

    # Añadir leyenda al gráfico.
    plt.legend()

    # Establecer los títulos de los ejes y del gráfico.
    plt.xlabel('Episodios')
    plt.ylabel('Recompensas')
    plt.title('Recompensas por algoritmo de aprendizaje por refuerzo')

    # Guardar la figura en el archivo especificado.
    plt.savefig(savepath)

  



