import gymnasium as gym
import numpy as np
from ppo import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    N = 20
    batch_size =  32
    n_epochs =  4
    alpha = 0.0003
    
    # observation = env.reset(seed=42)
    # print(env.observation_space.shape) # output: (4,)
    # print(env.action_space.n) # output: 2
    # print(observation) #  output: (array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), {})



    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha= alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
    n_games = 300
    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0 

    for  i in range(n_games):
        observation = env.reset()[0]
        truncation = False
        termination = False
        score = 0 
        while not (termination or truncation):
            action, prob, val =  agent.choose_action(observation)
            observation_, reward, termination, truncation, info = env.step(action)
            # env.render()
            # print('new observation after a step',observation_)

            # Reward shaping based on pole angle and angular velocity
            # pole_angle = abs(observation[2])  # Smaller angles are better
            # angular_velocity = abs(observation[3])  # Lower angular velocity is better

            # Adjust the reward to penalize large pole angles and high angular velocities
            # reward = reward - (0.1 * pole_angle) - (0.05 * angular_velocity)

            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, truncation, termination)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)


    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)


