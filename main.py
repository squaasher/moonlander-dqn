import os
import gym
import numpy as np
import torch
from gym.envs import box2d

from agent import DQNAgent

def fill_memory(env, agent, memory_fill_eps):
    for _ in range(memory_fill_eps):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.replay_memory.store(state, action, next_state, reward, done)
            state = next_state

def train(env, agent, train_eps, memory_fill_eps, batchsize, update_freq, model_filename):
    fill_memory(env, agent, memory_fill_eps)
    print("Samples in memory: ", len(agent.replay_memory))

    step_cnt = 0
    reward_history = []
    best_score = -np.inf

    for ep_cnt in range(train_eps):
        if ep_cnt % 500 == 0:
            env = gym.make("LunarLander-v2", render_mode="human")
        else:
            env = gym.make("LunarLander-v2")
        state = env.reset()
        done = False
        ep_reward = 0

        current_timestep_cntr = 0
        while not done:
            # EXPERIMENTAL_________________________________________________________________________
            if current_timestep_cntr >= 1000:
                reward = -100
                done = True
                agent.replay_memory.store(state, action, next_state, reward, done)
                agent.learn(batchsize)
                break
            #_____________________________________________________________________________________

            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            
            agent.replay_memory.store(state, action, next_state, reward, done)
            agent.learn(batchsize)

            if step_cnt % update_freq == 0:
                agent.update_target()

            state = next_state
            ep_reward += reward
            step_cnt += 1
            current_timestep_cntr += 1
            

        agent.update_epsilon()
        reward_history.append(ep_reward)

        current_avg_score = np.mean(reward_history[-100:])

        print("Ep: {}, Total Steps: {}, Ep Score: {}, Avg Score: {}; Updated Epsilon: {}".format(ep_cnt, step_cnt, ep_reward, current_avg_score, agent.epsilon))
        print()

        if current_avg_score >= best_score:
            agent.save("new_training_model")
            best_score = current_avg_score

def test(env, agent, test_eps):
    for ep_cnt in range(test_eps):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            ep_reward += reward

        print("Ep: {}, Ep Score: {}".format(ep_cnt, ep_reward))

def set_seed(env, seed_value):
    os.environ["PYTHONHASHSEED"]=str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    env.seed(seed_value)
    env.action_space.np_random.seed(seed_value)


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mode = False

    env = gym.make("LunarLander-v2", render_mode="human")
    model_filename = "ll_online_net"

    if train_mode:

        #set_seed(env, 0)

        agent = DQNAgent(observation_space=env.observation_space,
                         action_space=env.action_space,
                         device=device,
                         epsilon_max=1.0,
                         epsilon_min=0.01,
                         epsilon_decay=0.995,
                         memory_capacity=10000,
                         discount=0.99,
                         lr=1e-3)
        
        train(env=env,
              agent=agent,
              train_eps=2000,
              memory_fill_eps=20,
              batchsize=64,
              update_freq=1000,
              model_filename=model_filename)
    else:
        
        # set_seed(env, 10)

        agent = DQNAgent(observation_space=env.observation_space,
                         action_space=env.action_space,
                         device=device,
                         epsilon_max=0.0,
                         epsilon_min=0.0,
                         epsilon_decay=0.0,
                         memory_capacity=10000,
                         discount=0.99,
                         lr=1e-3)
        agent.load(model_filename)

        test(env=env, agent=agent, test_eps=100)