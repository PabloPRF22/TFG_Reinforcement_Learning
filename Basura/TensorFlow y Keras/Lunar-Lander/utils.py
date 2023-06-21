import matplotlib.pyplot as plt
import numpy as np

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Episodio", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])
    
    # Cálculo del average_reward cada 10 episodios
    n = 10
    running_avg_50 = np.empty(int(N/n))
    for i in range(int(N/n)):
        running_avg_50[i] = np.mean(running_avg[i*n:(i+1)*n])
    
    # Graficar el average_reward cada 50 episodios
    ax2.plot(x[::n], running_avg_50, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Recompnesa media', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
