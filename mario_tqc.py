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
        reward=-1
        self.no_progress+=1 
        if info["coins"]>self.curr_coin:
            reward += (info["coins"]-self.curr_coin)*100*(1+info["x_pos"]/2400)*(1+info["y_pos"]/150)
            if info["score"] > (info["coins"]-self.curr_coin)*200+self.curr_score:
                reward += ((info["score"] - self.curr_score)-(info["coins"]-self.curr_coin)*200)*0.5*(1+info["x_pos"]/4800)
            self.no_progress=0            
        elif (info["score"] - self.curr_score)>0:
            reward += (info["score"] - self.curr_score)*0.5*(1+info["x_pos"]/4800)
            self.no_progress=0
    
            

        # if info["status"]=="tall" and self.prev_status=="small":
        #     reward+=4800
        #     self.no_progress=0
        # if info["status"]=="fireball" and self.prev_status=="tall":
        #     reward+=4800
        #     self.no_progress=0       
    
        if done:
            if info["flag_get"]:
                reward += 300
                self.no_progress=0
            else:
                reward -= 0
                self.no_progress=0
        # if info["status"]=="tall" and info["coins"]==4:
        #     if self.x_prev<=446:
        #         reward+=(info["x_pos"]-self.x_prev)*40
        # if info["coins"]<4 and info["x_pos"]>260: 
        #     self.regret=True
        # if info["status"]=="small" and info["x_pos"]>260: 
        #     self.regret=True
        # if info["coins"]<13 and info["x_pos"]>580:
        #     self.regret=True
        # if info["status"]!="fireball" and info["x_pos"]>830:
        #     self.regret=True
        # if info["coins"]<19 and info["x_pos"]>760:
        #     self.regret=True 
        # if info["status"]=="tall" and info["coins"]==13:
        #     if  self.x_prev>=480 and self.x_prev<615:
        #         reward+=(info["x_pos"]-self.x_prev)*40
        if info["x_pos"]>self.x_prev and self.no_progress<32:
            reward+=(info["x_pos"]-self.x_prev)*0.1     
        if info["x_pos"]>self.far_x:
            reward+=(info["x_pos"]-self.far_x)*1
            # reward+=abs(info["y_pos"]-self.y_prev)*1.5
            self.far_x = info["x_pos"]
            self.no_progress=0
        if self.no_progress>32:
            if info["x_pos"]<self.x_prev: 
                reward-=abs(info["x_pos"]-self.x_prev)*0.25
                         
        self.x_prev = info["x_pos"]
        self.y_prev = info["y_pos"]
        self.curr_score = info["score"]
        self.curr_coin = info["coins"] 
        self.prev_status = info["status"]   
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
        if abs(self.x_pos-self.x_pre_pos)==2:
            self.stuck+=1
            reward=reward-0.1/10
        elif abs(self.x_pos-self.x_pre_pos)==1:
            self.stuck+=1
            reward=reward-0.2/10
        elif abs(self.x_pos-self.x_pre_pos)==0:
            self.stuck+=1
            reward=reward-0.5/10   
        else:      
            self.stuck=0
        if info_1[1]>16:
            reward-=info_1[1]/32/10
        self.x_pre_pos=self.x_pos
        # self.y_pre_pos=self.y_pos
        if done or self.stuck>64 or info_1[1]>128 or info_1[2]:            
            # obs, _ = self.env.reset(seed=114514,options={})
            if self.stuck>64:
                self.stuck=0
                self.x_pos = 40
                self.x_pre_pos=40
                # self.y_pos = 79
                # self.y_pre_pos=79
                return obs, reward-60/10, True, (info_1[0],info_1[1],65,False), info
            elif info_1[1]>128:
                self.stuck=0
                self.x_pos = 40
                self.x_pre_pos=40
                # self.y_pos = 79
                # self.y_pre_pos=79
                return obs, reward-60/10, True, (info_1[0],info_1[1],self.stuck,False), info
            elif info_1[2]:
                self.stuck=0
                self.x_pos = 40
                self.x_pre_pos=40
                # self.y_pos = 79
                # self.y_pre_pos=79
                return obs, reward-60/10, True, (info_1[0],info_1[1],self.stuck,True), info
            else:
                self.stuck=0
                self.x_pos = 40
                self.x_pre_pos=40
                # self.y_pos = 79
                # self.y_pre_pos=79
                return obs, reward-30/10, done, (info_1[0],info_1[1],self.stuck,False), info
       
        return obs,reward, done, (info_1[0],info_1[1],self.stuck,False), info
    def reset(self,seed,options):
        self.stuck=0
        self.x_pos = 40
        self.x_pre_pos=40
        # self.y_pos = 79
        # self.y_pre_pos=79
        return self.env.reset(seed=seed,options=options)





class SAC_actor(nn.Module):
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
      
        self.actor_linear = nn.Linear(512, output_dim)
        self._initialize_weights()
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                
    def forward(self, x):
        x1 = x[0]
        x3 = x[1]
        x1 = F.relu(self.conv1(x1))
      
        x1 = F.relu(self.conv2(x1))
      
        x1 = F.relu(self.conv3(x1))
       
        x1= x1.view(x1.size(0), -1)

        x1=torch.cat((x1,x3),1)
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))

        return self.actor_linear(x1)
class SAC_softQ(nn.Module):
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
      
        self.critic_linear = nn.Linear(512, output_dim*25)
        self._initialize_weights()
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                
    def forward(self, x):
        x1 = x[0]
        x3 = x[1]
        x1 = F.relu(self.conv1(x1))
      
        x1 = F.relu(self.conv2(x1))
      
        x1 = F.relu(self.conv3(x1))
       
        x1= x1.view(x1.size(0), -1)

        x1=torch.cat((x1,x3),1)
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))

        return self.critic_linear(x1)



def create_train_env(world, stage):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage),apply_api_compatibility=True)
    env = JoypadSpace(env, LEFTRIGHT_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = CustomReward(env)
    env = FrameStack(env, num_stack=4)
    env = Resetstate(env)
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
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(self.envs[index].step(action.item()))
            elif request == "reset":                                
                self.env_conns[index].send(self.envs[index].reset(seed=114514,options={}))
            else:
                raise NotImplementedError
def train(opt,writer):
    far_x=-100
    max_reward=-100
    step_max=0
    coins_max=0
    score_max=0
    _success_=False
    average_reward=[]
    average_x=[]
    average_coins=[]
    average_score=[]
    opt.saved_path =  opt.saved_path + "/"+datetime.now().strftime("%m_%d_%Y_%H:%M:%S")    
    env_eval = create_train_env(opt.world, opt.stage)
    env = create_train_env(opt.world, opt.stage)
    random.seed(114514)
    np.random.seed(114514)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(114514)
    else:
        torch.manual_seed(114514)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    model_actor = SAC_actor((env.observation_space.shape,10),len(LEFTRIGHT_MOVEMENT))
    model_q1 = SAC_softQ((env.observation_space.shape,10),len(LEFTRIGHT_MOVEMENT))
    model_q1_target = SAC_softQ((env.observation_space.shape,10),len(LEFTRIGHT_MOVEMENT))
    model_q2 = SAC_softQ((env.observation_space.shape,10),len(LEFTRIGHT_MOVEMENT))
    model_q2_target = SAC_softQ((env.observation_space.shape,10),len(LEFTRIGHT_MOVEMENT))
    model_q3 = SAC_softQ((env.observation_space.shape,10),len(LEFTRIGHT_MOVEMENT))
    model_q3_target = SAC_softQ((env.observation_space.shape,10),len(LEFTRIGHT_MOVEMENT))
    model_q4 = SAC_softQ((env.observation_space.shape,10),len(LEFTRIGHT_MOVEMENT))
    model_q4_target = SAC_softQ((env.observation_space.shape,10),len(LEFTRIGHT_MOVEMENT))
    model_q5 = SAC_softQ((env.observation_space.shape,10),len(LEFTRIGHT_MOVEMENT))
    model_q5_target = SAC_softQ((env.observation_space.shape,10),len(LEFTRIGHT_MOVEMENT))
    memory_s = torch.zeros(opt.memory_size, env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2],dtype=torch.float32)
    memory_s_info = torch.zeros(opt.memory_size, 10,dtype=torch.float32)
    memory_a = torch.zeros(opt.memory_size, 1, dtype=torch.int64)
    memory_r = torch.zeros(opt.memory_size, 1,dtype=torch.float32)
    memory_d = torch.zeros(opt.memory_size, 1, dtype=torch.uint8)
    alpha=torch.tensor(opt.alpha, dtype=float, requires_grad=False)
    
    target_entropy = torch.tensor(opt.alpha_min * (-np.log(1 / len(LEFTRIGHT_MOVEMENT))),requires_grad=False)

    
    if torch.cuda.is_available():
        model_actor=model_actor.cuda()
        model_q1=model_q1.cuda()
        model_q1_target=model_q1_target.cuda()
        model_q2=model_q2.cuda()
        model_q2_target=model_q2_target.cuda()
        model_q3=model_q3.cuda()
        model_q3_target=model_q3_target.cuda()
        model_q4=model_q4.cuda()
        model_q4_target=model_q4_target.cuda()
        model_q5=model_q5.cuda()
        model_q5_target=model_q5_target.cuda()
        alpha=alpha.cuda()
        target_entropy=target_entropy.cuda()
        alpha_log=torch.tensor(np.log(opt.alpha), dtype=float, requires_grad=True,device="cuda")
        memory_s = memory_s.cuda()
        memory_s_info = memory_s_info.cuda()
        memory_a =memory_a.cuda()
        memory_r = memory_r.cuda()
        memory_d = memory_d.cuda()
    else:
        alpha_log=torch.tensor(np.log(opt.alpha), dtype=float, requires_grad=True)

    optimizer_actor = torch.optim.Adam(model_actor.parameters(), lr=opt.lr)
    optimizer_q1 = torch.optim.Adam(model_q1.parameters(), lr=opt.lr)
    optimizer_q2 = torch.optim.Adam(model_q2.parameters(), lr=opt.lr)
    optimizer_q3 = torch.optim.Adam(model_q3.parameters(), lr=opt.lr)
    optimizer_q4 = torch.optim.Adam(model_q4.parameters(), lr=opt.lr)
    optimizer_q5 = torch.optim.Adam(model_q5.parameters(), lr=opt.lr)
    optimizer_alpha = torch.optim.Adam([alpha_log], lr=opt.lr) 

    steps=0

       

    curr_episode = 0
    # f = open("output.txt", "w+")
    for iter in range(opt.num_global_steps):
        
        curr_episode+=1
        if curr_episode%opt.update_freq==0:
            start = time.time()
        if curr_episode % opt.save_interval == 0 and curr_episode > 0:
            torch.save(model_actor.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}_TQC_actor".format(opt.saved_path, opt.world, opt.stage, curr_episode))
            torch.save(model_q1.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}_TQC_q1".format(opt.saved_path, opt.world, opt.stage, curr_episode))
            torch.save(model_q1_target.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}_TQC_q1_target".format(opt.saved_path, opt.world, opt.stage, curr_episode))
            torch.save(model_q2.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}_TQC_q2".format(opt.saved_path, opt.world, opt.stage, curr_episode))
            torch.save(model_q2_target.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}_TQC_q2_target".format(opt.saved_path, opt.world, opt.stage, curr_episode))
            torch.save(model_q3.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}_TQC_q3".format(opt.saved_path, opt.world, opt.stage, curr_episode))
            torch.save(model_q3_target.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}_TQC_q3_target".format(opt.saved_path, opt.world, opt.stage, curr_episode))
            torch.save(model_q4.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}_TQC_q4".format(opt.saved_path, opt.world, opt.stage, curr_episode))
            torch.save(model_q4_target.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}_TQC_q4_target".format(opt.saved_path, opt.world, opt.stage, curr_episode))
            torch.save(model_q5.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}_TQC_q5".format(opt.saved_path, opt.world, opt.stage, curr_episode))
            torch.save(model_q5_target.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}_TQC_q5_target".format(opt.saved_path, opt.world, opt.stage, curr_episode))
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

        

        state = torch.from_numpy(np.array(env.reset(seed=114514,options={})[0]).reshape(1,4,84,84))
        # sample trajectory
        if torch.cuda.is_available():
            state_info=state_info.cuda()
            state=state.cuda()           
        while True:
            if steps>opt.burn_in:
                with torch.no_grad():
                    logits  = model_actor((state,state_info))
                    policy = F.softmax(logits, dim=1)
                    old_m = Categorical(policy)
                    action = old_m.sample()                       
            else:                
                action =torch.randint(0,len(LEFTRIGHT_MOVEMENT),(1,))
            next_state, reward, done, info_1, info = env.step(action.item())            
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
                state_info[0][7]=1.0
            state_info[0][8]=info["coins"]/10
            state_info[0][9]=(999-info["time"])/100000      
            next_state = torch.from_numpy(np.array(next_state).reshape(1,4,84,84))
            if torch.cuda.is_available():
                next_state=next_state.cuda().detach()
                memory_s[steps%opt.memory_size]=state
                memory_s_info[steps%opt.memory_size]=state_info.clone()
                memory_a[steps%opt.memory_size]=action
                memory_r[steps%opt.memory_size]=reward
                memory_d[steps%opt.memory_size]=done                       
                
            else:
                memory_s[steps%opt.memory_size]=state
                memory_s_info[steps%opt.memory_size]=state_info.clone()
                memory_a[steps%opt.memory_size]=action
                memory_r[steps%opt.memory_size]=reward
                memory_d[steps%opt.memory_size]=done 
                   
                  
            state=next_state
    
            steps+=1
            if done:
                break
        
        
        # updating
        model_actor.train()   
        model_q1.train()   
        model_q2.train()   

        if steps>opt.burn_in:
           
            # retrieve exprerience
            if curr_episode%opt.update_freq==0:
                for _ in range(opt.gradient_steps):
                    ind = np.random.randint(min(opt.memory_size,steps),size=opt.batch_size)          
                        
                    states= memory_s[ind]
                    next_states= memory_s[(ind+opt.n_step_returns)%min(opt.memory_size,steps)]
                    actions= memory_a[ind].clone() 
                    rewards= memory_r[ind].clone()
                    dones= memory_d[ind].float().clone()
                    p_state_infos= memory_s_info[ind]
                    n_state_infos= memory_s_info[(ind+opt.n_step_returns)%min(opt.memory_size,steps)]
                   
                    # print(memory_r[ind])
                    # print(memory_r[ind+1])
                    # print(memory_r[ind+2])
                    # print(memory_d[ind])
                    # print(memory_d[ind+1])
                    # print(memory_d[ind+2])
                    # compute n-step returns here
                    # print(rewards)
                    # print(dones)
                    with torch.no_grad():
                        for i in range(opt.n_step_returns-1):
                            rewards+=(1-dones)*memory_r[(ind+i+1)%min(opt.memory_size,steps)]*(opt.gamma**(i+1))  
                            dones = 1-(1-dones)*(1-memory_d[(ind+i+1)%min(opt.memory_size,steps)].float())
                            # print(rewards)
                            # print(dones)
                    # time.sleep(2)      
                    # update q-network   
                    with torch.no_grad():
                        next_probs = F.softmax(model_actor((next_states,n_state_infos)), dim=1)
                        next_log_probs = torch.log(next_probs+1e-8)
                        q1_next_all=model_q1_target((next_states,n_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)
                        q2_next_all=model_q2_target((next_states,n_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)
                        q3_next_all=model_q3_target((next_states,n_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)
                        q4_next_all=model_q4_target((next_states,n_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)
                        q5_next_all=model_q5_target((next_states,n_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)
                        q_next_all=torch.cat((q1_next_all,q2_next_all,q3_next_all,q4_next_all,q5_next_all),dim=2)
                        q_next_all,_=torch.sort(q_next_all,dim=2,stable=True)
                        q_next_all=q_next_all[:,:,:20*5]
                        next_probs = next_probs.unsqueeze_(-1).expand(opt.batch_size,len(LEFTRIGHT_MOVEMENT),20*5) 
                        next_log_probs = next_log_probs.unsqueeze_(-1).expand(opt.batch_size,len(LEFTRIGHT_MOVEMENT),20*5) 
                        v_next = torch.sum(next_probs*(q_next_all-alpha*next_log_probs),dim=1)     
                        target_q = rewards.reshape(-1,1).expand(opt.batch_size,20*5)+(1-dones).reshape(-1,1).expand(opt.batch_size,20*5)*opt.gamma*v_next # [batch_size,100]                    
                        

                    q1_all = model_q1((states,p_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)             
                    q2_all = model_q2((states,p_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)
                    q3_all = model_q3((states,p_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)              
                    q4_all = model_q4((states,p_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)
                    q5_all = model_q5((states,p_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)              
        
                    # print(f"q1 {q1_all}",file=f)
                    # print(f"q2 {q2_all}",file=f)
                    actions=actions.reshape(-1,1)[:,:,None].expand(opt.batch_size,1,25)
               
                    q1 = torch.squeeze(q1_all.gather(1,actions))  # [batch_size,25]                 
                    q2 = torch.squeeze(q2_all.gather(1,actions))    
                    q3 = torch.squeeze(q3_all.gather(1,actions))    
                    q4 = torch.squeeze(q4_all.gather(1,actions))    
                    q5 = torch.squeeze(q5_all.gather(1,actions))    

                    pairwise_delta_q1 = target_q[:,None,:]-q1[:,:,None] # [batch_size,25,100]
                    pairwise_delta_q2 = target_q[:,None,:]-q2[:,:,None] # [batch_size,25,100]     
                    pairwise_delta_q3 = target_q[:,None,:]-q3[:,:,None] # [batch_size,25,100]  
                    pairwise_delta_q4 = target_q[:,None,:]-q4[:,:,None] # [batch_size,25,100] 
                    pairwise_delta_q5 = target_q[:,None,:]-q5[:,:,None] # [batch_size,25,100] 

                    abs_pairwise_delta_q1 = torch.abs(pairwise_delta_q1)
                    abs_pairwise_delta_q2 = torch.abs(pairwise_delta_q2)
                    abs_pairwise_delta_q3 = torch.abs(pairwise_delta_q3)
                    abs_pairwise_delta_q4 = torch.abs(pairwise_delta_q4)
                    abs_pairwise_delta_q5 = torch.abs(pairwise_delta_q5)

                    huber_loss_1 = torch.where(abs_pairwise_delta_q1>1,abs_pairwise_delta_q1-0.5,pairwise_delta_q1**2*0.5)
                    huber_loss_2 = torch.where(abs_pairwise_delta_q2>1,abs_pairwise_delta_q2-0.5,pairwise_delta_q2**2*0.5)
                    huber_loss_3 = torch.where(abs_pairwise_delta_q3>1,abs_pairwise_delta_q3-0.5,pairwise_delta_q3**2*0.5)
                    huber_loss_4 = torch.where(abs_pairwise_delta_q4>1,abs_pairwise_delta_q4-0.5,pairwise_delta_q4**2*0.5)
                    huber_loss_5 = torch.where(abs_pairwise_delta_q5>1,abs_pairwise_delta_q5-0.5,pairwise_delta_q5**2*0.5)

                    tau = torch.arange(25,device="cuda").float()/25+1/2/25

                    loss_1 = (torch.abs(tau[None,:,None]-(pairwise_delta_q1<0).float())*huber_loss_1).mean()
                    loss_2 = (torch.abs(tau[None,:,None]-(pairwise_delta_q2<0).float())*huber_loss_2).mean()
                    loss_3 = (torch.abs(tau[None,:,None]-(pairwise_delta_q3<0).float())*huber_loss_3).mean()
                    loss_4 = (torch.abs(tau[None,:,None]-(pairwise_delta_q4<0).float())*huber_loss_4).mean()
                    loss_5 = (torch.abs(tau[None,:,None]-(pairwise_delta_q5<0).float())*huber_loss_5).mean()
                    


                    q_loss = loss_1+loss_2+loss_3+loss_4+loss_5
                 
                    optimizer_q1.zero_grad(set_to_none=True)
                    optimizer_q2.zero_grad(set_to_none=True)
                    optimizer_q3.zero_grad(set_to_none=True)
                    optimizer_q4.zero_grad(set_to_none=True)
                    optimizer_q5.zero_grad(set_to_none=True)                
                    q_loss.backward()
                    optimizer_q1.step()
                    optimizer_q2.step()    
                    optimizer_q3.step()
                    optimizer_q4.step()  
                    optimizer_q5.step()
                                         
                    # update action network
                    for params in model_q1.parameters():
                        params.requires_grad = 	False
                    for params in model_q2.parameters():
                        params.requires_grad = 	False
                    for params in model_q3.parameters():
                        params.requires_grad = 	False
                    for params in model_q4.parameters():
                        params.requires_grad = 	False
                    for params in model_q5.parameters():
                        params.requires_grad = 	False
                


                    
                    probs =  F.softmax(model_actor((states,p_state_infos)), dim=1)
                    log_probs = torch.log(probs+1e-8)
                                     
                    with torch.no_grad():
                        q1_all = model_q1((states,p_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)
                        q2_all = model_q2((states,p_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)
                        q3_all = model_q3((states,p_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)
                        q4_all = model_q4((states,p_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)
                        q5_all = model_q5((states,p_state_infos)).reshape(opt.batch_size,len(LEFTRIGHT_MOVEMENT),-1)                           
                        q_all= torch.cat((q1_all,q2_all,q3_all,q4_all,q5_all),dim=2)
                    
                    a_loss = torch.sum(probs * (alpha*log_probs - q_all.mean(dim=2)), dim=1, keepdim=True).mean()



                    # print(f"a loss: {a_loss}",file=f)
                    optimizer_actor.zero_grad(set_to_none=True)
                    a_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model_actor.parameters(), 0.5)
                    optimizer_actor.step()

                    for params in model_q1.parameters():
                        params.requires_grad = 	True
                    for params in model_q2.parameters():
                        params.requires_grad = 	True
                    for params in model_q3.parameters():
                        params.requires_grad = 	True
                    for params in model_q4.parameters():
                        params.requires_grad = 	True
                    for params in model_q5.parameters():
                        params.requires_grad = 	True
                    # update alpha
                    
                    with torch.no_grad():
                        probs =  F.softmax(model_actor((states,p_state_infos)), dim=1).detach()
                        log_probs = torch.log(probs+1e-8).detach()
                        H_mean = torch.sum(probs*log_probs,dim=1).mean().detach()
                    alpha_loss = -alpha_log*(H_mean+target_entropy)
                    # print(f"alpha loss: {alpha_loss}",file=f)
                    optimizer_alpha.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    # print(H_mean)
                    # print(target_entropy)
                    # print(-(H_mean+target_entropy))
                    # print(alpha_log.grad)
                    # torch.nn.utils.clip_grad_norm_(alpha_log, 0.5)
                    optimizer_alpha.step()
                    alpha = alpha_log.exp()
                    # print(f"H mean: {H_mean}",file=f)
                    # print(f"target entropy: {target_entropy}",file=f)
                    # print(f"alpha: {alpha}",file=f)
                    # update target network
                    for param, target_param in zip(model_q1.parameters(), model_q1_target.parameters()):
                        target_param.data.copy_(opt.tau * param.data + (1 - opt.tau) * target_param.data)
                    for param, target_param in zip(model_q2.parameters(), model_q2_target.parameters()):
                        target_param.data.copy_(opt.tau * param.data + (1 - opt.tau) * target_param.data)
                    for param, target_param in zip(model_q3.parameters(), model_q3_target.parameters()):
                        target_param.data.copy_(opt.tau * param.data + (1 - opt.tau) * target_param.data)
                    for param, target_param in zip(model_q4.parameters(), model_q4_target.parameters()):
                        target_param.data.copy_(opt.tau * param.data + (1 - opt.tau) * target_param.data)
                    for param, target_param in zip(model_q5.parameters(), model_q5_target.parameters()):
                        target_param.data.copy_(opt.tau * param.data + (1 - opt.tau) * target_param.data)
                

                print("Episode: {} steps: {} q1 loss: {} q2 loss: {} q3 loss: {} q4 loss: {} q5 loss: {} actor loss: {} alpha loss: {} alpha:{} entropy:{}".format(curr_episode,steps,loss_1,loss_2,loss_3,loss_4,loss_5,a_loss,alpha_loss,alpha,-H_mean))
                writer.add_scalar('q1_Loss',loss_1, curr_episode)
                writer.add_scalar('q2_Loss',loss_2, curr_episode)
                writer.add_scalar('q3_Loss',loss_3, curr_episode)
                writer.add_scalar('q4_Loss',loss_4, curr_episode)
                writer.add_scalar('q5_Loss',loss_5, curr_episode)
                writer.add_scalar('actor_Loss',a_loss, curr_episode)
                writer.add_scalar('alpha_Loss',alpha_loss, curr_episode)
                writer.add_scalar('alpha',alpha, curr_episode)
                writer.add_scalar('entropy',-H_mean, curr_episode)    
            


        # evaluating
        if curr_episode % opt.update_freq == 0: 
            model_actor.eval() 
            time_spent=0
            temp_reward_sum =0
            temp_step=0
            success_=False
            stuck=0
            state = torch.from_numpy(np.array(env_eval.reset(seed=114514,options={})[0]).reshape(1,4,84,84)).clone()
            r_state_info = torch.zeros(1,10)
            r_state_info[0][0]=40/1280
            r_state_info[0][1]=79/1024
            r_state_info[0][2]=40/1280                
            r_state_info[0][3]=79/1024
            r_state_info[0][4]=40/1280
            r_state_info[0][5]=0/400
            r_state_info[0][6]=0/200
            r_state_info[0][7]=0
            r_state_info[0][8]=0/10 
            r_state_info[0][9]=0/100000
            
            if torch.cuda.is_available():
                r_state_info=r_state_info.cuda()
                state=state.cuda()
               
            # if record==1:
            #     print(f"At 14330516 {curr_episode}",file=f)  
            # temp=[]
            while True:       
                with torch.no_grad():  
                    # if record==1:
                    #     print(f"state_info {r_state_info}",file=f)   
                    #     print(f"The sum {torch.sum(state.cuda())}")
                    # temp.append(state)
                    logits  = model_actor((state,r_state_info))
                    policy = F.softmax(logits, dim=1)
                    action = torch.argmax(policy)
                    # print(action)
                    # if record==1:
                    #     print(f"logits {torch.sum(logits)}")
                    #     print(f"policy {policy}")
                    #     print(f"action {action}")
                    #     print(f"policy {policy}",file=f)  
                    #     print(f"action {action}",file=f)         
                next_state, reward, done, info_1, info = env_eval.step(action.item())  
                # print(state==next_state)
                # if len(temp)>1:  
                    # print(torch.sum(temp[len(temp)-1])==torch.sum(temp[len(temp)-2]))
                # if record==1:
                #         print(f"reward {reward}",file=f)  
                #         print(f"done {done}",file=f)        
                r_state_info[0][0]=r_state_info[0][2]
                r_state_info[0][1]=r_state_info[0][3]
                r_state_info[0][2]=info["x_pos"]/1280              
                r_state_info[0][3]=info["y_pos"]/1024
                r_state_info[0][4]=info_1[0]/1280
                r_state_info[0][5]=info_1[1]/400
                r_state_info[0][6]=info_1[2]/200
                if info["status"]=="small":
                    r_state_info[0][7]=0        
                elif info["status"]=="tall":
                    r_state_info[0][7]=0.5
                else:
                    r_state_info[0][7]=1.0
                r_state_info[0][8]=info["coins"]/10
                r_state_info[0][9]=(999-info["time"])/100000  

                # print("------------")
                # print(r_state_info)
                # print(f"At {temp_step}: x: {info['x_pos']} y {info['y_pos']} ")
                # print(f"At {temp_step}: coins: {info['coins']} score: {info['score']}")
                # print(f"At {temp_step}: reward: {reward} action: {action}")
                # print(f"At {temp_step}: stuck: {info_1[2]} no-progress: {info_1[1]}")
                # print(f"At {temp_step}: polciy: {policy}")  
                
                next_state=torch.from_numpy(np.array(next_state).reshape(1,4,84,84)).clone()
                if torch.cuda.is_available():
                    next_state=next_state.cuda().detach()
                    # memory_s[steps%opt.memory_size]=state
                    # memory_s_info[steps%opt.memory_size]=state_info.clone()
                    # memory_a[steps%opt.memory_size]=action
                    # memory_r[steps%opt.memory_size]=reward
                    # memory_d[steps%opt.memory_size]=done  
                    # steps+=1        
                    
                # else:
                #     memory_s[steps%opt.memory_size]=state
                #     memory_s_info[steps%opt.memory_size]=state_info.clone()
                #     memory_a[steps%opt.memory_size]=action
                #     memory_r[steps%opt.memory_size]=reward
                #     memory_d[steps%opt.memory_size]=done 
                #     steps+=1                     
              
                
               
                state=next_state                
            
                time_spent = 999-info["time"]            
                temp_reward_sum+=reward
                temp_step+=1
                if done:
                    if info["flag_get"]:
                        success_=True
                        print(f"Success at step {temp_step} with reward {temp_reward_sum}")                    
                    else:
                        if info_1[2]>64:
                            print(f"Stuck killed at step {temp_step} position {info['x_pos']} with action {action} and prob {policy[0]}")    
                            stuck=-3  
                        elif info_1[1]>128:
                            print(f"No progress killed at step {temp_step} position {info['x_pos']} with action {action} and prob {policy[0]}")    
                            stuck=-2  
                        elif info_1[3]:
                            print(f"Not collected all killed at step {temp_step} position {info['x_pos']} with action {action} and prob {policy[0]}")    
                            stuck=-1
                        else:
                            print(f"Died at step {temp_step} position {info['x_pos']} with action {action} and prob {policy[0]}")
                    break
            if stuck ==-3 :
                print(f"Stuck Killed: Episode: {curr_episode} Reward: {temp_reward_sum} Steps: {temp_step} X_reached: {info['x_pos']} Score:{info['score']} Coins:{info['coins']}")
            elif stuck==-2:
                print(f"No Progress Killed: Episode: {curr_episode} Reward: {temp_reward_sum} Steps: {temp_step} X_reached: {info['x_pos']} Score:{info['score']} Coins:{info['coins']}")
            elif stuck==-1:
                print(f"Not all collected Killed: Episode: {curr_episode} Reward: {temp_reward_sum} Steps: {temp_step} X_reached: {info['x_pos']} Score:{info['score']} Coins:{info['coins']}") 
            elif success_==True:
                print(f"Success: Episode: {curr_episode} Reward: {temp_reward_sum} Steps: {temp_step} X_reached: {info['x_pos']} Time_spent: {time_spent} Score:{info['score']} Coins:{info['coins']}")
            else:
                print(f"Died: Episode: {curr_episode} Reward: {temp_reward_sum} Steps: {temp_step} X_reached: {info['x_pos']} Time_spent: {time_spent} Score:{info['score']} Coins:{info['coins']}")
            average_reward.append(temp_reward_sum)
            moving_average = np.mean(average_reward[-100:])
            average_x.append(info['x_pos'])
            moving_x = np.mean(average_x[-100:])
            average_score.append(info["score"])
            moving_score = np.mean(average_score[-100:])
            average_coins.append(info["coins"])
            moving_coins = np.mean(average_coins[-100:])
            print(f"Moving_reward: {moving_average} Moving_x: {moving_x} Moving_score: {moving_score} Moving_coins: {moving_coins}")
            writer.add_scalar('Reward',temp_reward_sum, curr_episode)
            writer.add_scalar('Steps',temp_step, curr_episode)
            writer.add_scalar('X_reached',info['x_pos'], curr_episode)
            writer.add_scalar('Time_spent',time_spent, curr_episode)
            # writer.add_scalar('Fail_prob',policy[0][action], curr_episode)
            writer.add_scalar('Moving_reward',moving_average, curr_episode)
            writer.add_scalar('Moving_x',moving_x, curr_episode)
            writer.add_scalar('Success',success_, curr_episode)
            writer.add_scalar('Moving Score',moving_score, curr_episode)
            writer.add_scalar('Moving Coins', moving_coins, curr_episode)
            if temp_reward_sum>=max_reward:
                print(f"Mario breaks his record! Reaching {info['x_pos']} with return {temp_reward_sum} coins {info['coins']} score {info['score']}. Saving the model")
                torch.save(model_actor.state_dict(),"best_tqc_actor.pt")  
                torch.save(model_q1.state_dict(),"best_tqc_q1.pt")  
                torch.save(model_q1_target.state_dict(),"best_tqc_q1_target.pt")  
                torch.save(model_q2.state_dict(),"best_tqc_q2.pt")  
                torch.save(model_q2_target.state_dict(),"best_tqc_q2_target.pt")  
                torch.save(model_q3.state_dict(),"best_tqc_q3.pt")  
                torch.save(model_q3_target.state_dict(),"best_tqc_q3_target.pt")  
                torch.save(model_q4.state_dict(),"best_tqc_q4.pt")  
                torch.save(model_q4_target.state_dict(),"best_tqc_q4_target.pt")  
                torch.save(model_q5.state_dict(),"best_tqc_q5.pt")  
                torch.save(model_q5_target.state_dict(),"best_tqc_q5_target.pt")  
                max_reward=temp_reward_sum
                step_max = temp_step
                far_x = info['x_pos']
                _success_=success_
                coins_max = info['coins']
                score_max= info['score']
            # print(f"Farest: {far_x} Step: {step_max} Success? {success_}")
            # writer.add_scalar('Farest_x',far_x, curr_episode)
            print(f"Largest return: {max_reward} coins: {coins_max} scores: {score_max} Steps: {step_max} Reaching {far_x} Success? {_success_}")
            writer.add_scalar('Largest Return',max_reward, curr_episode)
            end = time.time()
            print(f"Total steps: {steps}")
            print(f"elapsed time: {end - start}")          


def get_args():
    parser = argparse.ArgumentParser(
        """SAC Implementation""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument('--lr', type=float, default=3*1e-4)
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')


    parser.add_argument('--batch_size', type=int, default=2048)

    parser.add_argument("--num_global_steps", type=int, default=100000)
    
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--saved_path", type=str, default="trained_models/tqc")


    parser.add_argument("--alpha_min", type=float, default="0.8")
    parser.add_argument("--tau", type=float, default="0.005")
    parser.add_argument("--alpha", type=float, default="0.2")
    parser.add_argument("--memory_size", type=int, default="70000")
    parser.add_argument("--burn_in", type=int, default="10000")
    parser.add_argument("--update_freq", type=int, default="1")
    parser.add_argument("--gradient_steps", type=int, default="100")
    parser.add_argument("--n_step_returns", type=int, default="3")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    
    opt = get_args()
    
    print(f"Using CUDA: {torch.cuda.is_available()}")
    # torch.compile(train)
    exp_saved_path = "runs/tqc/"+"lr="+str(opt.lr)+"_gamma="+str(opt.gamma)+"_memory_size="+str(opt.memory_size)+"_batch_size="+str(opt.batch_size)+"_alpha_min="+str(opt.alpha_min)+"_alpha="+str(opt.alpha)+"_tau="+str(opt.tau)+"_update_freq="+str(opt.update_freq)+"_n_step_returns="+str(opt.n_step_returns)+"_gradient_steps="+str(opt.gradient_steps)+"_"+datetime.now().strftime("%m_%d_%Y_%H:%M:%S")
    writer = SummaryWriter(log_dir=exp_saved_path)
    # torch.set_float32_matmul_precision("high")
   
    train(opt,writer)