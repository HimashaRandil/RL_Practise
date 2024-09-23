import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_adversary_v3

def obs_list_to_state_vector(observation_list):
    return np.concatenate(observation_list)

if __name__ == '__main__':
    env = simple_adversary_v3.env()
    env.reset()

    n_agents = env.num_agents
    actor_dims = [env.observation_space(agent).shape[0] for agent in env.agents]
    critic_dims = sum(actor_dims)
    n_actions = env.action_space(env.agents[0]).n

    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario="simple_adversary",
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                                    n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 500
    N_GAMES = 50000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        env.reset()
        score = 0
        episode_step = 0
        done = {agent: False for agent in env.agents}
        obs = {agent: None for agent in env.agents}

        for agent in env.agent_iter():
            if episode_step >= MAX_STEPS:
                done = {agent: True for agent in env.agents}
                break

            if done[agent]:
                action = None  # Skip if done
            else:
                obs[agent], reward, done[agent], truncation, _ = env.last()

                if obs[agent] is not None:
                    actions = {}
                    for idx, agent_key in enumerate(env.agents):
                        agent_obs = obs[agent_key]
                        if agent_obs is not None:
                            action = maddpg_agents.agent[idx].choose_action(agent_obs)

                            # Ensure action is valid
                            if isinstance(action, np.ndarray):
                                action = action[0]  # Take the first element if it's an array
                            
                            action = np.clip(int(action), 0, n_actions - 1)  # Clip to valid action range

                            if 0 <= action < n_actions:
                                actions[agent_key] = action
                            else:
                                print(f"Invalid action for {agent_key}: {action}, using default action 0")
                                actions[agent_key] = 0  # Default action if invalid
                        else:
                            actions[agent_key] = 0  # Default action if observation is None

                    env.step(actions)

                    # Fetch the next state and rewards
                    obs_, reward_, done_, truncation, _ = env.last()

                    # Create state vectors
                    state = obs_list_to_state_vector([obs[agent_key] for agent_key in env.agents])
                    state_ = obs_list_to_state_vector([obs_[agent_key] for agent_key in env.agents])

                    # Store transitions in memory
                    memory.store_transition(obs, state, actions, reward, obs_, state_, done_)

                    if total_steps % 100 == 0 and not evaluate:
                        maddpg_agents.learn(memory)

                    score += reward
                    total_steps += 1
                    episode_step += 1
                else:
                    env.step(None)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score

        if i % PRINT_INTERVAL == 0 and i > 0:
            print(f'Episode {i}, Average Score: {avg_score:.1f}')
