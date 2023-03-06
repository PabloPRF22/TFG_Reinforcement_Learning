import numpy as np
import gym


start_epsilon = 1.0
n_episodes = 10000
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1
alpha = 0.1
gamma = 0.99
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001

# Initialize Taxi-v3 environment
env = gym.make('Taxi-v3').env
rewards_history = list()

def initialize_Q_table(env):
    """
    For the given environment (Taxi-v3), the possible actions are integers from 0-5, 
    and the possible states are integers from 0-499.
    Therefore, the Q-Value of, for example, taking action 3 in state 156 can be found as Q[156, 3]
    """
    return np.zeros([env.observation_space.n, env.action_space.n])

def get_epsilon_greedy_action(Q, state):
    
    # Pick random action
    if np.random.uniform(0, 1) < start_epsilon:
        return env.action_space.sample()
    
    # Pick action with highest Q-Value
    return np.argmax(Q[state])

    
def execute_episode(Q):
    global start_epsilon,alpha,gamma
    # Initialize Rewards historic
    rewards = list()
    
    # Reset environment
    state, _ = env.reset()
    done = False
    print("Episodio")
    while not done:
        
        # 1) Pick action following epsilon-greedy policy
        action = get_epsilon_greedy_action(Q, state)
        
        # 2) Perform action and receive new state S' and reward R
        new_state, reward, done, info, _ = env.step(action)
        rewards.append(reward)
        
        # 3) Update Q-values 
        old_value = Q[state, action]
        next_max = np.max(Q[new_state])
        next_action = get_epsilon_greedy_action(Q,new_state)

        Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[new_state, next_action] - Q[state, action])
        
        state = new_state
    
    return Q, sum(rewards)


def train_q_learning():
    
    Q = initialize_Q_table(env)
    global start_epsilon,min_epsilon,max_epsilon,decay_rate
    for episode in range(n_episodes):
        print(f'Training on Episode {episode+1}... Epsilon: {start_epsilon}', end="\r")

        Q, reward = execute_episode(Q)
        start_epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        rewards_history.append(reward)
    
    return Q, rewards_history

trained_Q, rewards_history = train_q_learning()
env = gym.make('Taxi-v3')
state, _ = env.reset()
def execute_episodes_on_trained_agent(Q, n_episodes):
    
    episode_rewards = dict()
    for episode in range(n_episodes):
        print(f'Executing Episode {episode+1}...')
        
        # Initialize historics
        rewards = list()

        state, _ = env.reset()

        done = False
        truncated = False
        n_steps = 0

        while not done and not truncated:

            action = np.argmax(Q[state])
            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            n_steps += 1
            
        episode_rewards[episode] = (sum(rewards), n_steps)
        print(f'Episode {episode+1} took {n_steps} steps, and got a reward of {sum(rewards)}\n')

    return episode_rewards
env = gym.make('Taxi-v3')
state, _ = env.reset()
def execute_episodes_on_trained_agent(Q, n_episodes):
    
    episode_rewards = dict()
    for episode in range(n_episodes):
        print(f'Executing Episode {episode+1}...')
        
        # Initialize historics
        rewards = list()

        state, _ = env.reset()

        done = False
        truncated = False
        n_steps = 0
        
        while not done and not truncated:

            action = np.argmax(Q[state])
            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            n_steps += 1
            
        episode_rewards[episode] = (sum(rewards), n_steps)
        print(f'Episode {episode+1} took {n_steps} steps, and got a reward of {sum(rewards)}\n')

    return episode_rewards
    

n_test_episodes = 200
trained_agent_rewards = execute_episodes_on_trained_agent(trained_Q, n_test_episodes)

avg_steps = sum([episode_info[1] for episode_info in trained_agent_rewards.values()])/n_test_episodes
avg_reward = sum([episode_info[0] for episode_info in trained_agent_rewards.values()])/n_test_episodes

print('\n\n')
print('Average number of timesteps per episode: ', avg_steps)
print('Average reward per episode: ', avg_reward)