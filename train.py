__author__ = "Jie Ren"
__email__ = "jieren9806@gmail.com"

import json
import os
import time

import gym
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

try:
    from models import _SAC as SAC
except:
    from models import SAC
import tools
import train_funs
import utils
import argparse


if __name__ == '__main__':
    # Hyperparams  --------------------------------------------------
    cmd_args = argparse.ArgumentParser()
    cmd_args.add_argument("--env_name", default="Humanoid-v2")
    cmd_args.add_argument("--baseline", default="SAC")
    cmd_args.add_argument("-k", type=int, default=4)
    args = cmd_args.parse_args()

    config_name = args.env_name + '-' + args.baseline
    params = json.load(open(os.path.join("./script/", f"{config_name}.json"), 'r'))
    params["env_name"] = args.env_name
    params["k"] = int(args.k)
    params["baseline"] = args.baseline

    print(params)
    #  Specific_param  ----------------------------------------------
    env_name = params["env_name"]
    k = params["k"]
    baseline = params["baseline"]
    lr_mu = params["lr_actor"]
    lr_q = params["lr_critic"]
    buffer_maxlen = params["buffer_maxlen"]
    tau = params["tau"]
    save_interval = params["save_interval"]
    more_epoch = params["more_epoch"]
    inner_train_epoch = params["inner_train_epoch"]
    epoch_length = params["epoch_length"]

    # General params  -----------------------------------------------
    super_params = json.load(open("./script/super.json", 'r'))
    print(super_params)

    device = torch.device(super_params["device"])
    resume = super_params["resume"]
    debug = super_params["debug"]
    seed = super_params["seed"]
    allow_render = super_params["allow_render"]
    eval_interval = super_params["eval_interval"]
    eval_episodes = super_params["eval_episodes"]

    resume_root = os.path.join("./ckp", env_name, baseline, f"k_{k}")
    utils.recurrent_mkdir(resume_root)

    if not debug:
        writer = SummaryWriter(os.path.join("./logs", env_name, baseline, f'k_{k}'))
        render_interval = params["render_interval"]
    elif debug:
        render_interval = 999999

    # ----------------------------------------------------------
    env = gym.make(env_name)
    utils.set_seed(seed, env)
    env_info = utils.get_env_info(env)

    if allow_render:
        import pygame

        pygame.display.init()
        window = pygame.display.set_mode([800, 100 + 50 * k])
        window.fill(pygame.Color(255, 255, 255))
        pygame.font.init()

    memory = tools.ReplayBuffer(buffer_maxlen, device)

    if baseline == "SAC":
        agent = SAC.SAC_Agent(env_info, params, device)

    # Loading  ------------------------------------------------------
    if resume:
        ckp_name = os.listdir(resume_root)
        ckp_name.sort(reverse=True)
        # ckp_name = [f"k_{k}.pth"]
        ckp_path = os.path.join(resume_root, ckp_name[0])
        start_epoch, total_step = agent.load_model(ckp_path)
        print(f"start from {start_epoch}")
    else:
        start_epoch = 0

    # Saving data ---------------------------------------------------
    total_step = 0
    for epoch in tqdm.tqdm(range(start_epoch, start_epoch + more_epoch)):
        observation = env.reset()
        train_reward = 0
        step_counter = 0
        step_list = []
        while True:
            with torch.no_grad():
                observation = torch.from_numpy(observation).float().to(device)
                pi, agent_index, action = agent.take_action(observation)

            action = action.squeeze(0)

            observation = observation.cpu().numpy()
            action = action.cpu().numpy()

            scaled_action = action * env_info["action_scale"]
            observation_prime, reward, done, info = env.step(scaled_action)
            memory.put((observation, action, reward, observation_prime, done))
            train_reward += reward
            observation = observation_prime
            if (epoch + 1) % render_interval == 0 and allow_render:
                utils.visualize(window, env_info, agent_index, pi, scaled_action, k)
                env.render()
            step_counter += 1
            if done:
                if step_counter < epoch_length:
                    observation = env.reset()
                    step_list.append(step_counter-sum(step_list))
                else:
                    step_list.append(step_counter-sum(step_list))
                    break
        # Train, Log and Save  --------------------------------------
        total_step += sum(step_list)
        logs = {
            "step/train_max_step": max(step_list),
            "step/train_avg_step": sum(step_list) / len(step_list),
            "step/total_step": total_step,
            "reward/train_reward": train_reward,
            "reward/train_avg_reward": train_reward / sum(step_list)
        }
        if baseline == "SAC":
            train_logs = train_funs.train_sac(params, agent, memory)
            logs.update(train_logs)

        if (epoch + 1) % eval_interval == 0:
            eval_logs = train_funs.eval(super_params, env, env_info, agent)
            logs.update(eval_logs)
        if not debug:
            utils.write_logs(writer, logs, epoch)
            if (epoch + 1) % save_interval == 0:
                agent.save_model(os.path.join(resume_root, f"epoch_{epoch:04d}.pth"), epoch, total_step)
    env.close()
