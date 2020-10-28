__author__ = "Jie Ren"
__email__ = "jieren9806@gmail.com"

import json
import os

import gym
import torch
import copy

from models import SAC
import utils
import argparse
import time
import numpy as np

if __name__ == '__main__':
    # Hyperparams  --------------------------------------------------
    config_name = "Walker2d-v2-SAC"
    params = json.load(open(os.path.join("./script/", f"{config_name}.json"), 'r'))

    cmd_args = argparse.ArgumentParser()
    cmd_args.add_argument("--env_name", default=None)
    cmd_args.add_argument("-k", default=10)
    args = cmd_args.parse_args()
    if args.env_name is not None:
        params["env_name"] = args.env_name
    if args.k is not None:
        params['k'] = int(args.k)

    print(params)
    #  Specific_param  ----------------------------------------------
    env_name = params["env_name"]
    k = params['k']
    baseline = params["baseline"]

    # General params  -----------------------------------------------
    super_params = json.load(open("./script/super.json", 'r'))
    print(super_params)

    device = torch.device(super_params["device"])
    display_delay = super_params["display_delay"]
    seed = super_params["seed"]

    resume_root = os.path.join("./ckp", env_name, baseline, f"k_{k}")

    env = gym.make(env_name)
    utils.set_seed(seed, env)
    env_info = utils.get_env_info(env)

    import pygame

    pygame.display.init()
    window = pygame.display.set_mode([400 + 250 * env_info["action_space"].shape[0], 100 + 50 * k])
    window.fill(pygame.Color(255, 255, 255))
    pygame.font.init()

    if baseline == "SAC":
        agent = SAC.SAC_Agent(env_info, params, device)

    ckp_name = os.listdir(resume_root)
    ckp_name.sort(reverse=True)
    # ckp_name = [f"k_{k}.pth"]
    print(f"load {ckp_name[0]}")
    ckp_path = os.path.join(resume_root, ckp_name[0])
    agent.load_model(ckp_path)

    noise_frequent = 10
    noise_length = 2
    epoch_counter = -1
    reward_list = []
    step_list = []

    while True:
        epoch_counter += 1
        reward_counter = 0
        step_counter = 0

        observation = env.reset()
        while True:
            with torch.no_grad():
                observation = torch.from_numpy(observation).float().to(super_params["device"])
                pi, agent_index, action, action_chosen = agent.take_visualization_action(observation)

                action_chosen = action_chosen.cpu().numpy()
                action = action.cpu().numpy()

            if step_counter % noise_frequent < noise_length:
                noise = np.random.randn(*action_chosen.shape) * 0
                # print(f'noise:{noise}')
            else:
                noise = 0

            action_chosen += noise
            scaled_action_chosen = action_chosen * env_info["action_scale"]
            scaled_action = action * env_info["action_scale"]
            observation_prime, reward, done, info = env.step(scaled_action_chosen)
            reward_counter += reward
            observation = observation_prime
            utils.visualize(window, env_info, agent_index, pi, scaled_action, agent.k)
            env.render()
            time.sleep(display_delay)
            step_counter += 1
            if done:
                step_list.append(step_counter)
                reward_list.append(reward_counter)
                print(f"================================================================================")
                print(epoch_counter)
                print(f"step = {step_counter}, reward = {reward_counter:.2f}")
                print(f"avg_step = {sum(step_list)//len(step_list)}, avg_reward = {sum(reward_list)/len(reward_list):.2f}")

                break
    # env.close()
