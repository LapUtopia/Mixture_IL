__author__ = "Jie Ren"
__email__ = "jieren9806@gmail.com"

import torch
from torch.distributions import Categorical
from torch.nn import functional as F
import utils


def eval(super_params, env, env_info, agent):
    total_reward = 0
    step_list = []
    for _ in range(super_params["eval_episodes"]):
        step_counter = 0
        observation = env.reset()
        while True:
            with torch.no_grad():
                observation = torch.from_numpy(observation).float().to(super_params["device"])
                pi, agent_index, action = agent.take_eval_action(observation)

                action = action.squeeze(0).cpu().numpy()

            scaled_action = action * env_info["action_scale"]
            observation_prime, reward, done, info = env.step(scaled_action)
            observation = observation_prime
            total_reward += reward
            step_counter += 1
            if done:
                step_list.append(step_counter)
                break

    eval_logs = {
        "reward/eval_reward": total_reward / super_params["eval_episodes"],
        "reward/eval_avg_reward": total_reward / sum(step_list),
        "step/eval_max_step": max(step_list),
        "step/eval_avg_step": sum(step_list) / len(step_list)
    }
    return eval_logs


def train_sac(params, sac_agent, memory):
    q1_loss_total = 0
    q2_loss_total = 0
    actor_loss_total = 0
    pi_loss_total = 0
    alpha_loss_total = 0
    for i in range(params["inner_train_epoch"]):
        q1_loss, q2_loss, actor_loss, pi_loss, alpha_loss = sac_agent.train_epoch(memory)
        q1_loss_total += q1_loss
        q2_loss_total += q2_loss
        actor_loss_total += actor_loss
        pi_loss_total += pi_loss
        alpha_loss_total += alpha_loss
        if i % params["soft_update_interval"] == 0:
            sac_agent.soft_update()
    q1_loss_total /= params["inner_train_epoch"]
    q2_loss_total /= params["inner_train_epoch"]
    actor_loss_total /= params["inner_train_epoch"]
    pi_loss_total /= params["inner_train_epoch"]
    alpha_loss_total /= params["inner_train_epoch"]
    train_logs = {
        "loss/q1_loss": q1_loss_total,
        "loss/q2_loss": q2_loss_total,
        "loss/alpha_loss": alpha_loss_total,
        "loss/actor_loss": actor_loss_total,
        "loss/pi_loss": pi_loss_total,
    }
    return train_logs


