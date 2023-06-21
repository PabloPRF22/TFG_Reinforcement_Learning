import gym
import pybullet_envs
import imageio

# Crear el entorno
env_names = pybullet_envs
print(env_names)
env = gym.make('KukaCamBulletEnv-v0')

# Crear una lista para almacenar los frames de la ejecución
frames = []

# Reiniciar el entorno
obs = env.reset()

# Ejecutar el entorno durante 1000 pasos
for i in range(1000):
    # Renderizar el entorno
    img = env.render(mode='rgb_array')
    
    # Añadir el frame actual a la lista
    frames.append(img)
    
    # Tomar una acción aleatoria
    action = env.action_space.sample()
    
    # Ejecutar la acción y obtener la observación y la recompensa
    obs, reward, done, info = env.step(action)
    
    # Salir del bucle si el episodio ha terminado
    if done:
        break

# Guardar la ejecución en un archivo de video
imageio.mimsave('kuka_execution.mp4', frames, fps=60)

# Cerrar el entorno
env.close()
