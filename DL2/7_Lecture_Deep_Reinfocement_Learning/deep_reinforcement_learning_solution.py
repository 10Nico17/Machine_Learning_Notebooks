from typing import List, Tuple, Union

import lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer

import torch
from torch.utils.data import DataLoader
import torch.optim as optim 
from torch.optim.optimizer import Optimizer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def create_mlp(input_shape: Tuple[int], n_actions: int, hidden_sizes: list = [128, 128]):

    net_layers = []
    net_layers.append(torch.nn.Linear(input_shape[0], hidden_sizes[0]))
    net_layers.append(torch.nn.ReLU())

    for i in range(len(hidden_sizes) - 1):
        net_layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        net_layers.append(torch.nn.ReLU())
    net_layers.append(torch.nn.Linear(hidden_sizes[-1], n_actions))

    return torch.nn.Sequential(*net_layers)


def discrete_actor_forward(actor_net, states):
    logits = actor_net(states)
    pi = torch.distributions.Categorical(logits=logits)
    action = pi.sample()
    return pi, action

def actor_critic_call(actor_net, critic_net, state):
    pi, action = actor_net(state)
    log_p = pi.log_prob(action)
    value = critic_net(state)
        
    return pi, action, log_p, value
    
def get_actor_loss(actor, state, action, logp_old, advantage, clip_ratio):
    pi, _ = actor(state)
    logp = pi.log_prob(action)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
    loss_actor = -(torch.min(ratio * advantage, clip_adv)).mean()
    return loss_actor

def get_critic_loss(critic, state, qvalue):
    value = critic(state)
    loss_critic = (qvalue - value).pow(2).mean()
    return loss_critic

def get_step_images(agent, state, env, max_episode_len):
    imgs = []
    for step in range(max_episode_len):
        pi, action, log_prob, value = agent(state)
        next_state, reward, done, *_ = env.step(action.cpu().numpy())
        state = torch.FloatTensor(next_state)
        img = env.render()
        imgs.append(img)
        if done:
            break
    return imgs