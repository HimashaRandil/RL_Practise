import gymnasium as gym
from ppo import Agent

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')  # Enable rendering
    alpha = 0.0003
    
    # Initialize the agent with the same architecture and parameters used during training
    agent = Agent(n_actions=env.action_space.n, batch_size=5,
                  alpha=alpha, n_epochs=4, input_dims=env.observation_space.shape)
    
    # Load the trained models
    agent.load_models()

    # Test the agent
    n_games = 100  # Test for 10 episodes
    for i in range(n_games):
        observation = env.reset()[0]
        score = 0
        truncation = False
        termination = False

        while not (termination or truncation):
            action, _, _ = agent.choose_action(observation)  # Choose action based on the trained model
            observation_, reward, termination, truncation, info = env.step(action)
            score += reward
            env.render()  # Render the environment at each step
            observation = observation_

        print(f'Episode {i+1} score: {score}')
    
    env.close()  # Close the environment when done
