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

    # Mostrar la figura.
    plt.show()


    

def plot_train_results2(recompensas_medias, savepath, titulo,env):
    # Crear una figura
    fig  = plt.figure()
    lenmax =0
    for i in range (len(recompensas_medias)):
        if lenmax<len(recompensas_medias[i]):
            lenmax = len(recompensas_medias[i])
    # Itera sobre cada conjunto de recompensas_medias
    for i in range(len(recompensas_medias)):
        # Gráfica de la recompensa media
        plt.plot([j*5 for j in range(len(recompensas_medias[i]))], recompensas_medias[i], label=titulo[i])

    # Añadir una leyenda
    plt.legend(loc='upper right')
    ax = fig.gca()
    ax.set_xticks([50,100,150,200,250,300,350,400,450,500])
    
    # Título de la figura
    plt.title(env)

    # Etiquetas de los ejes
    plt.xlabel('Episodio')
    plt.ylabel('Valor')

    # Mostrar la figura
    plt.show()

    # Guardar la figura
    plt.savefig(savepath)


if __name__ == '__main__':
    data = [
        ("Algorithm1", 50.5),
        ("Algorithm2", 75.3),
        ("Algorithm3", 60.2),
        ("Algorithm4", 200.12),
        ("Algorithm5", 0.0),
        ("Algorithm6", -200),
        ("Algorithm7", 10.3),
        ("Algorithm8", 500),
        ("Algorithm9", 348),
    ]
    #plot_mean_reward_per_algorithm(data, 'Taxi-v3', Path("results_plot.png"))
