import matplotlib.pyplot as plt
import numpy as np

def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_rewards(rewards, save_path, window_size=100):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Recompensa por episodio')
    plt.plot(np.arange(window_size - 1, len(rewards)), moving_average(rewards, window_size),
             label=f'Recompensa media cada {window_size} episodios')
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa acumulada')
    plt.title('Recompensas a lo largo del entrenamiento')
    plt.legend()
    plt.savefig(save_path)
    plt.show()