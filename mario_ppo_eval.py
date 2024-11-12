import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
import time
import matplotlib.pyplot as plt
# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import time
import random, math
from gym import Env
from gym import Wrapper
# NES Emulator for OpenAI Gym

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
import torch.nn.functional as F
import torch.multiprocessing as _mp
import argparse
import os
import shutil
from torch.distributions import Categorical
from collections import OrderedDict,deque
from datetime import datetime
import imageio
from torch.utils.tensorboard import SummaryWriter

LEFTRIGHT_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
]
class JoypadSpace(Wrapper):
    """An environment wrapper to convert binary to discrete action space."""

    # a mapping of buttons to binary values
    _button_map = {
        'right':  0b10000000,
        'left':   0b01000000,
        'down':   0b00100000,
        'up':     0b00010000,
        'start':  0b00001000,
        'select': 0b00000100,
        'B':      0b00000010,
        'A':      0b00000001,
        'NOOP':   0b00000000,
    }

    @classmethod
    def buttons(cls) -> list:
        """Return the buttons that can be used as actions."""
        return list(cls._button_map.keys())

    def __init__(self, env: Env, actions: list):
        """
        Initialize a new binary to discrete action space wrapper.
        Args:
            env: the environment to wrap
            actions: an ordered list of actions (as lists of buttons).
                The index of each button list is its discrete coded value
        Returns:
            None
        """
        super().__init__(env)
        # create the new action space
        self.action_space = gym.spaces.Discrete(len(actions))
        # create the action map from the list of discrete actions
        self._action_map = {}
        self._action_meanings = {}
        # iterate over all the actions (as button lists)
        for action, button_list in enumerate(actions):
            # the value of this action's bitmap
            byte_action = 0
            # iterate over the buttons in this button list
            for button in button_list:
                byte_action |= self._button_map[button]
            # set this action maps value to the byte action value
            self._action_map[action] = byte_action
            self._action_meanings[action] = ' '.join(button_list)

    def step(self, action):
        """
        Take a step using the given action.
        Args:
            action (int): the discrete action to perform
        Returns:
            a tuple of:
            - (numpy.ndarray) the state as a result of the action
            - (float) the reward achieved by taking the action
            - (bool) a flag denoting whether the episode has ended
            - (dict) a dictionary of extra information
        """
        # take the step and record the output
        return self.env.step(self._action_map[action])

    def reset(self,seed,options):
        """Reset the environment and return the initial observation."""
        return self.env.reset(seed=seed,options=options)

    def get_keys_to_action(self):
        """Return the dictionary of keyboard keys to actions."""
        # get the old mapping of keys to actions
        old_keys_to_action = self.env.unwrapped.get_keys_to_action()
        # invert the keys to action mapping to lookup key combos by action
        action_to_keys = {v: k for k, v in old_keys_to_action.items()}
        # create a new mapping of keys to actions
        keys_to_action = {}
        # iterate over the actions and their byte values in this mapper
        for action, byte in self._action_map.items():
            # get the keys to press for the action
            keys = action_to_keys[byte]
            # set the keys value in the dictionary to the current discrete act
            keys_to_action[keys] = action

        return keys_to_action

    def get_action_meanings(self):
        """Return a list of actions meanings."""
        actions = sorted(self._action_meanings.keys())
        return [self._action_meanings[action] for action in actions]

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape,antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation
# class CustomReward(gym.RewardWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.curr_score = 0
#         self.curr_coin=0
#         self.rs=[]
#         self.ret=0
#         self.far_x=40
#         self.x_prev=40
#         self.y_prev=79
#         self.no_progress=0
#         self.prev_status="small"
#         self.regret=False
#     def step(self, action):
#         obs, reward, done, trunk, info = self.env.step(action)
#         reward=-0.5 
#         self.no_progress+=1 
#         if info["coins"]>self.curr_coin:
#             reward += (info["coins"]-self.curr_coin)*400*(1+info["x_pos"]/2400)*(1+info["y_pos"]/150)
#             if info["score"] > (info["coins"]-self.curr_coin)*200+self.curr_score:
#                 reward += ((info["score"] - self.curr_score)-(info["coins"]-self.curr_coin)*200)*2*(1+info["x_pos"]/4800)
#             self.no_progress=0            
#         elif (info["score"] - self.curr_score)>0:
#             reward += (info["score"] - self.curr_score)*2*(1+info["x_pos"]/4800)
#             self.no_progress=0
    
            

#         if info["status"]=="tall" and self.prev_status=="small":
#             reward+=4800
#             self.no_progress=0
#         if info["status"]=="fireball" and self.prev_status=="tall":
#             reward+=4800
#             self.no_progress=0       
    
#         if done:
#             if info["flag_get"]:
#                 reward += 4000
#                 self.no_progress=0
#             else:
#                 reward -= 0
#                 self.no_progress=0
#         # if info["status"]=="tall" and info["coins"]==4:
#         #     if self.x_prev<=446:
#         #         reward+=(info["x_pos"]-self.x_prev)*40
#         # if info["coins"]<4 and info["x_pos"]>260: 
#         #     self.regret=True
#         # if info["status"]=="small" and info["x_pos"]>260: 
#         #     self.regret=True
#         # if info["coins"]<13 and info["x_pos"]>580:
#         #     self.regret=True
#         # if info["status"]!="fireball" and info["x_pos"]>830:
#         #     self.regret=True
#         # if info["coins"]<19 and info["x_pos"]>760:
#         #     self.regret=True 
#         # if info["status"]=="tall" and info["coins"]==13:
#         #     if  self.x_prev>=480 and self.x_prev<615:
#         #         reward+=(info["x_pos"]-self.x_prev)*40
#         if info["x_pos"]>self.x_prev and self.no_progress<32:
#             reward+=(info["x_pos"]-self.x_prev)*0.0025      
#         if info["x_pos"]>self.far_x:
#             reward+=(info["x_pos"]-self.far_x)*4
#             # reward+=abs(info["y_pos"]-self.y_prev)*1.5
#             self.far_x = info["x_pos"]
#             self.no_progress=0
#         if self.no_progress>32:
#             if info["x_pos"]<self.x_prev: 
#                 reward-=abs(info["x_pos"]-self.x_prev)*1
                         
#         self.x_prev = info["x_pos"]
#         self.y_prev = info["y_pos"]
#         self.curr_score = info["score"]
#         self.curr_coin = info["coins"] 
#         self.prev_status = info["status"]   
#         true_reward = reward/200
#         # self.ret = 0.99*self.ret+true_reward
#         # self.rs.append(self.ret)
#         # if len(self.rs)>1:
#         #     std_rolling = np.array(self.rs).std()
#         #     scale_reward = true_reward/(std_rolling+1e-8)
#         # else:
#         #     scale_reward = true_reward
#         return obs, true_reward, done, (self.far_x,self.no_progress,self.regret), info
#     def reset(self,seed,options):
#         self.curr_score = 0
#         self.x_prev=40
#         self.y_prev=79
#         self.curr_coin=0
#         self.rs=[]
#         self.ret=0
#         self.far_x=40
#         self.no_progress=0
#         self.prev_status="small"
#         self.regret=False
#         return self.env.reset(seed=seed,options=options)
    
    
# class Resetstate(gym.Wrapper):
#     def __init__(self, env):
#         """Return only every `skip`-th frame"""
#         super().__init__(env)
#         self.stuck=0
#         self.x_pos = 40
#         self.x_pre_pos=40
#         # self.y_pos = 79
#         # self.y_pre_pos=79
#     def step(self, action):
#         """Repeat action, and sum reward"""
#         # Accumulate reward and repeat the same action
#         obs, reward, done, info_1, info = self.env.step(action)
#         self.x_pos=info["x_pos"]
#         # self.y_pos=info["y_pos"]
#         if abs(self.x_pos-self.x_pre_pos)==2:
#             self.stuck+=1
#             reward=reward-0.1/200
#         elif abs(self.x_pos-self.x_pre_pos)==1:
#             self.stuck+=1
#             reward=reward-0.2/200
#         elif abs(self.x_pos-self.x_pre_pos)==0:
#             self.stuck+=1
#             reward=reward-0.5/200   
#         else:      
#             self.stuck=0
#         if info_1[1]>16:
#             reward-=info_1[1]/32/200
#         self.x_pre_pos=self.x_pos
#         # self.y_pre_pos=self.y_pos
#         if done or self.stuck>64 or info_1[1]>128 or info_1[2]:            
#             # obs, _ = self.env.reset(seed=114514,options={})
#             if self.stuck>64:
#                 self.stuck=0
#                 self.x_pos = 40
#                 self.x_pre_pos=40
#                 # self.y_pos = 79
#                 # self.y_pre_pos=79
#                 return obs, reward-120/200, True, (info_1[0],info_1[1],65,False), info
#             elif info_1[1]>128:
#                 self.stuck=0
#                 self.x_pos = 40
#                 self.x_pre_pos=40
#                 # self.y_pos = 79
#                 # self.y_pre_pos=79
#                 return obs, reward-120/200, True, (info_1[0],info_1[1],self.stuck,False), info
#             elif info_1[2]:
#                 self.stuck=0
#                 self.x_pos = 40
#                 self.x_pre_pos=40
#                 # self.y_pos = 79
#                 # self.y_pre_pos=79
#                 return obs, reward-120/200, True, (info_1[0],info_1[1],self.stuck,True), info
#             else:
#                 self.stuck=0
#                 self.x_pos = 40
#                 self.x_pre_pos=40
#                 # self.y_pos = 79
#                 # self.y_pre_pos=79
#                 return obs, reward-60/200, done, (info_1[0],info_1[1],self.stuck,False), info
       
#         return obs,reward, done, (info_1[0],info_1[1],self.stuck,False), info
#     def reset(self,seed,options):
#         self.stuck=0
#         self.x_pos = 40
#         self.x_pre_pos=40
#         # self.y_pos = 79
#         # self.y_pre_pos=79
#         return self.env.reset(seed=seed,options=options)

class CustomReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.curr_score = 0
        self.curr_coin=0
        self.rs=[]
        self.ret=0
        self.far_x=40
        self.x_prev=40
        self.y_prev=79
        self.no_progress=0
        self.prev_status="small"
        self.regret=False
    def step(self, action):
        obs, reward, done, trunk, info = self.env.step(action)
        reward-=1
        # self.no_progress+=1 
        if info["coins"]>self.curr_coin:
            reward += (info["coins"]-self.curr_coin)*100*(1+info["x_pos"]/2400)*(1+info["y_pos"]/150)
            if info["score"] > (info["coins"]-self.curr_coin)*200+self.curr_score:
                reward += ((info["score"] - self.curr_score)-(info["coins"]-self.curr_coin)*200)*(1+info["x_pos"]/4800)/10                      
        elif (info["score"] - self.curr_score)>0:
            reward += ((info["score"] - self.curr_score)-(info["coins"]-self.curr_coin)*200)*(1+info["x_pos"]/4800)/10
           
        
            

        # if info["status"]=="tall" and self.prev_status=="small":
        #     reward+=4800
        #     self.no_progress=0
        # if info["status"]=="fireball" and self.prev_status=="tall":
        #     reward+=4800
        #     self.no_progress=0       
    
        if done:
            if info["flag_get"]:
                reward += 40
                self.no_progress=0
            else:
                reward -= 25
                self.no_progress=0
        # # if info["status"]=="tall" and info["coins"]==4:
        # #     if self.x_prev<=446:
        # #         reward+=(info["x_pos"]-self.x_prev)*40
        # # if info["coins"]<4 and info["x_pos"]>260: 
        # #     self.regret=True
        # # if info["status"]=="small" and info["x_pos"]>260: 
        # #     self.regret=True
        # # if info["coins"]<13 and info["x_pos"]>580:
        # #     self.regret=True
        # # if info["status"]!="fireball" and info["x_pos"]>830:
        # #     self.regret=True
        # # if info["coins"]<19 and info["x_pos"]>760:
        # #     self.regret=True 
        # # if info["status"]=="tall" and info["coins"]==13:
        # #     if  self.x_prev>=480 and self.x_prev<615:
        # #         reward+=(info["x_pos"]-self.x_prev)*40
        # if info["x_pos"]>self.x_prev and self.no_progress<32:
        #     reward+=(info["x_pos"]-self.x_prev)*0.0025      
        if info["x_pos"]>self.far_x:
            # reward+=(info["x_pos"]-self.far_x)*4
            # reward+=abs(info["y_pos"]-self.y_prev)*1.5
            self.far_x = info["x_pos"]
            # self.no_progress=0
        # if self.no_progress>32:
        #     if info["x_pos"]<self.x_prev: 
        #         reward-=abs(info["x_pos"]-self.x_prev)*1
                         
        # self.x_prev = info["x_pos"]
        # self.y_prev = info["y_pos"]
        self.curr_score = info["score"]
        self.curr_coin = info["coins"] 
        # self.prev_status = info["status"]   
        true_reward = reward/10
        # self.ret = 0.99*self.ret+true_reward
        # self.rs.append(self.ret)
        # if len(self.rs)>1:
        #     std_rolling = np.array(self.rs).std()
        #     scale_reward = true_reward/(std_rolling+1e-8)
        # else:
        #     scale_reward = true_reward
        return obs, true_reward, done, (self.far_x,self.no_progress,self.regret), info
    def reset(self,seed,options):
        self.curr_score = 0
        self.x_prev=40
        self.y_prev=79
        self.curr_coin=0
        self.rs=[]
        self.ret=0
        self.far_x=40
        self.no_progress=0
        self.prev_status="small"
        self.regret=False
        return self.env.reset(seed=seed,options=options)
    
    
class Resetstate(gym.Wrapper):
    def __init__(self, env):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self.stuck=0
        self.x_pos = 40
        self.x_pre_pos=40
        # self.y_pos = 79
        # self.y_pre_pos=79
    def step(self, action):
        """Repeat action, and sum reward"""
        # Accumulate reward and repeat the same action
        obs, reward, done, info_1, info = self.env.step(action)
        self.x_pos=info["x_pos"]
        # self.y_pos=info["y_pos"]
        if abs(self.x_pos-self.x_pre_pos)<=2:
            self.stuck+=1
       
        self.x_pre_pos=self.x_pos
        # self.y_pre_pos=self.y_pos
        if done or self.stuck>64 or info_1[1]>128 or info_1[2]:            
            obs, _ = self.env.reset(seed=114514,options={})
            if self.stuck>64:
                self.stuck=0
                self.x_pos = 40
                self.x_pre_pos=40
                # self.y_pos = 79
                # self.y_pre_pos=79
                return obs, reward-40/10, True, (info_1[0],info_1[1],65,False), info
            elif info_1[1]>128:
                self.stuck=0
                self.x_pos = 40
                self.x_pre_pos=40
                # self.y_pos = 79
                # self.y_pre_pos=79
                return obs, reward, True, (info_1[0],info_1[1],self.stuck,False), info
            elif info_1[2]:
                self.stuck=0
                self.x_pos = 40
                self.x_pre_pos=40
                # self.y_pos = 79
                # self.y_pre_pos=79
                return obs, reward, True, (info_1[0],info_1[1],self.stuck,True), info
            else:
                self.stuck=0
                self.x_pos = 40
                self.x_pre_pos=40
                # self.y_pos = 79
                # self.y_pre_pos=79
                return obs, reward, done, (info_1[0],info_1[1],self.stuck,False), info
       
        return obs,reward, done, (info_1[0],info_1[1],self.stuck,False), info
    def reset(self,seed,options):
        self.stuck=0
        self.x_pos = 40
        self.x_pre_pos=40
        # self.y_pos = 79
        # self.y_pre_pos=79
        return self.env.reset(seed=seed,options=options)





class PPO(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c,h,w = input_dim[0][0],input_dim[0][1],input_dim[0][2]
        z = input_dim[1]
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.conv1 = nn.Conv2d(c, 32, 8, stride=4, padding=0)     
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)     
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)        
        self.linear1 = nn.Linear(3136+z,512)
        self.linear2 = nn.Linear(512, 512)
        self.conv1_v = nn.Conv2d(c, 32, 8, stride=4, padding=0)     
        self.conv2_v = nn.Conv2d(32, 64, 4, stride=2, padding=0)     
        self.conv3_v = nn.Conv2d(64, 64, 3, stride=1, padding=0)        
        self.linear1_v = nn.Linear(3136+z,512)
        self.linear2_v = nn.Linear(512, 512)
        self.critic_linear = nn.Linear(512, 1)      
        self.actor_linear = nn.Linear(512, output_dim)
        self._initialize_weights()
    def _initialize_weights(self):
        torch.manual_seed(114514)
        random.seed(114514)
        np.random.seed(114514)
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)
    def forward(self, x):
        x1 = x[0]
        x2 = x[0]
        x3 = x[1]
        x1 = F.relu(self.conv1(x1))
      
        x1 = F.relu(self.conv2(x1))
      
        x1 = F.relu(self.conv3(x1))
       
        x1= x1.view(x1.size(0), -1)

        x1=torch.cat((x1,x3),1)
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))

        x2 = F.relu(self.conv1_v(x2))
      
        x2 = F.relu(self.conv2_v(x2))
      
        x2 = F.relu(self.conv3_v(x2))
       
        x2= x2.view(x2.size(0), -1)

        x2 = torch.cat((x2,x3),1)
        x2 = F.relu(self.linear1_v(x2))
        x2 = F.relu(self.linear2_v(x2))


        return self.actor_linear(x1), self.critic_linear(x2)

class RND(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim):
        super().__init__()
        c,h,w = input_dim[0],input_dim[1],input_dim[2]
    
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
        self.predictor= nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6400,512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.target= nn.Sequential(
           nn.Conv2d(c, 32, 8, stride=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6400,512),
         )
        self._initialize_weights()
    def _initialize_weights(self):
        torch.manual_seed(114514)
        random.seed(114514)
        np.random.seed(114514)
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)
        for param in self.target.parameters():
            param.requires_grad = False
    def forward(self, x):
        target_feature = self.target(x)
        predict_feature = self.predictor(x)

        return predict_feature, target_feature


def create_train_env(world, stage):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage),apply_api_compatibility=True)
    env = JoypadSpace(env, LEFTRIGHT_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = CustomReward(env)
    env = FrameStack(env, num_stack=4)
    env = Resetstate(env)
    env.action_space.seed(114514)
    return env
class MultipleEnvironments:
    def __init__(self, world, stage, num_envs):
        self.agent_conns, self.env_conns = zip(*[_mp.Pipe() for _ in range(num_envs)])
        self.envs = [create_train_env(world, stage) for _ in range(num_envs)]
        self.num_states = self.envs[0].observation_space.shape
        self.num_actions=len(LEFTRIGHT_MOVEMENT)
        for index in range(num_envs):
            process = _mp.Process(target=self.run, args=(index,))
            process.start()
            self.env_conns[index].close()

    def run(self, index):
        random.seed(114514)
        np.random.seed(114514)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(114514)
        else:
            torch.manual_seed(114514)
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(self.envs[index].step(action.item()))
            elif request == "reset":                                
                self.env_conns[index].send(self.envs[index].reset(seed=114514,options={}))
            else:
                raise NotImplementedError
def eval(opt):
    
    random.seed(114514)
    np.random.seed(114514)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(114514)
    else:
        torch.manual_seed(114514)
    # env_eval = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(opt.world, opt.stage),apply_api_compatibility=True,render_mode="rgb_array")
    # env_eval = create_train_env(1, 2)
    env_eval = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(opt.world, opt.stage),apply_api_compatibility=True,render_mode="human")
    
    env_eval = JoypadSpace(env_eval, LEFTRIGHT_MOVEMENT)
    env_eval = SkipFrame(env_eval, skip=4)
    env_eval = GrayScaleObservation(env_eval)
    env_eval = ResizeObservation(env_eval, shape=84)
    env_eval = CustomReward(env_eval)
    env_eval = FrameStack(env_eval, num_stack=4)
    env_eval = Resetstate(env_eval)
    
    env_eval.action_space.seed(114514)
    # env_eval.action_space.seed(114514)
   
    # model = torch.compile(PPO(env_eval.observation_space.shape,len(LEFTRIGHT_MOVEMENT)))
    model = PPO((env_eval.observation_space.shape,10),len(LEFTRIGHT_MOVEMENT))
    # RND_model = RND(env_eval.observation_space.shape)
    state_dict1 = torch.load("best_ppo_PPO.pt")
    # state_dict2 = torch.load("best_ppo_RND.pt")
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #         if k[:10] == '_orig_mod.':
    #             name = k[10:]  # remove `module.`
    #         else:
    #             name = k
    #         new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(state_dict1)
    # RND_model.load_state_dict(state_dict2)
    if torch.cuda.is_available():
        model=model.cuda()
        # RND_model.cuda()

    model.eval()  
    # RND_model.eval()
   
    temp_state = torch.from_numpy(np.array(env_eval.reset(seed=114514,options={})[0]).reshape(1,4,84,84)).cuda()
    temp_reward_sum =0
    temp_step=0
    x_curr=0
    stuck=0
    frames = []
    success_=False
    state_info = torch.zeros(1,10)
    state_info[0][0]=40/1280
    state_info[0][1]=79/1024
    state_info[0][2]=40/1280             
    state_info[0][3]=79/1024
    state_info[0][4]=40/1280
    state_info[0][5]=0/400
    state_info[0][6]=0/200
    state_info[0][7]=0
    state_info[0][8]=0/10
    state_info[0][9]=0/100000     
    if torch.cuda.is_available():
        state_info = state_info.cuda()
    while True:
        
        
        
        # Run agent on the state
       
        start = time.time()   
        with torch.no_grad():   
            logits, value = model((temp_state,state_info))
        end = time.time()
        print(f"model elapsed time: {end - start}")
        policy = F.softmax(logits, dim=1)
        # print(policy)
        action = torch.argmax(policy).item()
        # old_m = Categorical(policy)
        # action = old_m.sample().item()
        # if temp_step%2==0:
        #     action=4
        # else:
        #     action=1
        # actions.append(action)
        # print(action)
        # Agent performs action
        start = time.time()  
        next_state, reward, done, info_1, info = env_eval.step(action)
        end = time.time()
        print(f"environment elapsed time: {end - start}")
        # print(info["status"])
        # print(f"{x_curr},{x_pre_curr},{stuck}")
        # target,predict=RND_model(torch.FloatTensor(np.array(next_state).reshape(1,8,126,126)).cuda())
        # reward_int = ((target - predict).pow(2).sum(1)/80000).item()
        start = time.time() 
        x_curr=info["x_pos"]
        state_info[0][0]=state_info[0][2]
        state_info[0][1]=state_info[0][3]
        state_info[0][2]=info["x_pos"]/1280               
        state_info[0][3]=info["y_pos"]/1024
        state_info[0][4]=info_1[0]/1280
        state_info[0][5]=info_1[1]/400
        state_info[0][6]=info_1[2]/200
        if info["status"]=="small":
            state_info[0][7]=0        
        elif info["status"]=="tall":
            state_info[0][7]=0.5 
        else:
            state_info[0][7]=1
        state_info[0][8]=info["coins"]/10
        state_info[0][9]=(999-info["time"])/100000      
        end = time.time()
        print("------------")
        print(state_info)
        print(f"At {temp_step}: x: {info['x_pos']} y {info['y_pos']} ")
        print(f"At {temp_step}: coins: {info['coins']} score: {info['score']}")
        print(f"At {temp_step}: reward: {reward} action: {action}")
        print(f"At {temp_step}: stuck: {info_1[2]} no-progress: {info_1[1]}")
        print(f"At {temp_step}: value: {value}")
        print(f"At {temp_step}: polciy: {policy}")
        # time.sleep(0.5)
        # print(f"At {info['t1']} {info['t2']} {info['t3']}")
        # x_pre_curr=x_curr
        # time.sleep(0.25)
        temp_reward_sum+=reward
        # print(f'{temp_step}, {info["x_pos"]}, {x_max}, {reward}')
        # print(info["y_pos"])
        # print(x_max)
        # print(reward)
                    # Update state
        
        # print(state_info[0][0:64])
        temp_step+=1
        frame = np.copy(env_eval.render())
        # print(frame==old_frame)
        # old_frame=frame
        
   
        frames.append(frame)
        # plt.imshow(temp_state[0][7].cpu())
        # plt.savefig(f"{temp_step}_1.png")
        # plt.close()
    
        # Check if end of game
        # if stuck>25:
        #         print(x_curr)
        #         print("Stuck. Killed")
        #         break
        # print(state_info)
        if done:
            if info["flag_get"]:
                success_=True
                print(f"Success at step {temp_step} with reward {temp_reward_sum}") 
                
            else:
                if info_1[2]>64 or info_1[1]>128 or info_1[3]:
                    print(f"Terminated at step {temp_step} position {x_curr} with action {action} and prob {policy[0]}")  
                    stuck=-1    
                else:
                    print(f"Died at step {temp_step} position {x_curr} with action {action} and prob {policy[0]}")
            break
        temp_state = torch.from_numpy(np.array(next_state).reshape(1,4,84,84)).cuda()
    if stuck < 0:
        print(f"Killed: Reward: {temp_reward_sum} Steps: {temp_step} X_reached: {x_curr}")
    elif success_==True:
        print(f"Success: Reward: {temp_reward_sum} Steps: {temp_step} X_reached: {x_curr}")
    else:
        print(f"Died: Reward: {temp_reward_sum} Steps: {temp_step} X_reached: {x_curr}")
    
  
    imageio.mimsave('mario_replay.gif', frames, fps=30)

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--gamma_int', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=0.95, help='parameter for GAE')
    parser.add_argument('--beta2', type=float, default=0.005, help='entropy coefficient')
    parser.add_argument('--beta1', type=float, default=0.5, help='value coefficient')
    parser.add_argument('--beta3', type=float, default=0.1, help='forward coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=50000)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=20, help="Number of steps between savings")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--adv_normalization", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    opt = get_args()
    use_cuda = torch.cuda.is_available()

    print(f"Using CUDA: {use_cuda}")
    # torch.set_float32_matmul_precision("high")
    eval(opt)




