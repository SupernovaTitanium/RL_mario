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
import copy
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

from torch.utils.tensorboard import SummaryWriter
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
   
        # print(observation)
        observation.copy()
      
        observation = torch.tensor(np.copy(observation), dtype=torch.float)
 
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
class CustomReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.curr_score = 0
        self.curr_coin=0
        # self.x_pos=0
        self.rs=[]
        self.ret=0
        self.far_x=40
        self.x_prev=40
        self.no_progress=0
        self.prev_status="small"
    def step(self, action):
        obs, reward, done, trunk, info = self.env.step(action)
        reward=-0.25
        # if abs(info["x_pos"]-self.x_prev)<1:
        #     reward-=0.2
        # elif abs(info["x_pos"]-self.x_prev)<2:
        #     reward-=0.1
        # elif abs(info["x_pos"]-self.x_prev)<3:
        #     reward-=0.05
        if info["x_pos"]>self.x_prev:
            reward+=(info["x_pos"]-self.x_prev)*0.05
        self.x_prev = info["x_pos"]
        if info["x_pos"]>self.far_x:
            reward+=(info["x_pos"]-self.far_x)*2
            self.far_x = info["x_pos"]
            self.no_progress=0
        else:
            self.no_progress+=1
        
        if info["coins"]>self.curr_coin:
            reward += (info["coins"]-self.curr_coin)*800
            if info["score"] > (info["coins"]-self.curr_coin)*200+self.curr_score:
                reward += ((info["score"] - self.curr_score)-(info["coins"]-self.curr_coin)*200)*4
            self.no_progress=0
            self.curr_score = info["score"]
            self.curr_coin = info["coins"]    
        elif (info["score"] - self.curr_score)>0:
            reward += (info["score"] - self.curr_score)*4
            self.no_progress=0
            self.curr_score = info["score"]
        else:
            self.curr_score = info["score"]
        # if self.no_progress>64:
        #     reward-=1
        # reward += (info["x_pos"] - self.x_pos)*1.14514*2
        # if info["x_pos"]==self.x_pos:
        #     reward-=0.1
        if info["status"]=="tall" and self.prev_status=="small":
            reward+=4000
        if info["status"]=="fireball" and self.prev_status=="small":
            reward+=4000
        if info["status"]=="fireball" and self.prev_status=="tall":
            reward+=4000
        self.prev_status = info["status"]
        # self.x_pos = info["x_pos"]
        if done:
            if info["flag_get"]:
                reward += 2000
            else:
                reward -= 0
        true_reward = reward/800
        # self.ret = 0.99*self.ret+true_reward
        # self.rs.append(self.ret)
        # if len(self.rs)>1:
        #     std_rolling = np.array(self.rs).std()
        #     scale_reward = true_reward/(std_rolling+1e-8)
        # else:
        #     scale_reward = true_reward
        return obs, true_reward, done, (self.far_x,self.no_progress), info
    def reset(self,seed,options):
        self.curr_score = 0
        self.x_prev=40
        self.curr_coin=0
        self.rs=[]
        self.ret=0
        self.far_x=40
        self.no_progress=0
        self.prev_status="small"
        
        return self.env.reset(seed=seed,options=options)

class Resetstate(gym.Wrapper):
    def __init__(self, env):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self.stuck=0
        self.pos = 0
        self.pre_pos=-1
    def step(self, action):
        """Repeat action, and sum reward"""
        # Accumulate reward and repeat the same action
        obs, reward, done, info_1, info = self.env.step(action)
        self.pos=info["x_pos"]
        if abs(self.pos-self.pre_pos)==2:
            self.stuck+=1
            reward=reward-0.25/800
        elif abs(self.pos-self.pre_pos)==1:
            self.stuck+=1
            reward=reward-0.5/800
        elif abs(self.pos-self.pre_pos)==0:
            self.stuck+=1
            reward=reward-1/800       
        else:
            self.stuck=0
        if info_1[1]>64:
            reward-=0.5/800
        self.pre_pos=self.pos
        if done or self.stuck>64 or info_1[1]>128:            
            obs, _ = self.env.reset(seed=114514,options={})
            if self.stuck>64:
                self.stuck=0
                self.pos = 0
                self.pre_pos=-1
                return obs, reward-200/800, True, (info_1[0],info_1[1],65), info
            elif info_1[1]>128:
                self.stuck=0
                self.pos = 0
                self.pre_pos=-1
                return obs, reward-200/800, True, (info_1[0],info_1[1],self.stuck), info
            else:
                self.stuck=0
                self.pos = 0
                self.pre_pos=-1
                return obs, reward-100/800, done, (info_1[0],info_1[1],self.stuck), info
       
        return obs,reward, done, (info_1[0],info_1[1],self.stuck), info



class PPO(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c,h,w = input_dim[0][0],input_dim[0][1],input_dim[0][2]
        z = input_dim[1]
        if h != 126:
            raise ValueError(f"Expecting input height: 126, got: {h}")
        if w != 126:
            raise ValueError(f"Expecting input width: 126, got: {w}")

        self.conv1 = nn.Conv2d(c, 32, 8, stride=4, padding=0)
       
       
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
       
     
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        
        
        self.linear1 = nn.Linear(9216+z,512)
        self.linear2 = nn.Linear(512, 512)
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
        x3 = x[1]
        x1 = F.relu(self.conv1(x1))
      
        x1 = F.relu(self.conv2(x1))
      
        x1 = F.relu(self.conv3(x1))
       
        x1= x1.view(x1.size(0), -1)

        x1=torch.cat((x1,x3),1)
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))

        return self.actor_linear(x1), self.critic_linear(x1)

class RND(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim):
        super().__init__()
        c,h,w = input_dim[0],input_dim[1],input_dim[2]
    
        if h != 126:
            raise ValueError(f"Expecting input height: 126, got: {h}")
        if w != 126:
            raise ValueError(f"Expecting input width: 126, got: {w}")
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
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=126)
    env = CustomReward(env)
    env = FrameStack(env, num_stack=8)
    env = Resetstate(env)
    env.action_space.seed(114514)
    return env
class MultipleEnvironments:
    def __init__(self, world, stage, num_envs,num_steps,model):
        self.agent_conns, self.env_conns = zip(*[_mp.Pipe() for _ in range(num_envs)])
        
        
        self.num_actions=len(COMPLEX_MOVEMENT)
        for index in range(num_envs):
            
            process = _mp.Process(target=self.run, args=(world, stage,index,num_steps,model))
            process.start()
            # self.env_conns[index].close()

    def run(self, world, stage,index,num_steps,model):
        self.agent_conns[index].close()   
        env = create_train_env(world, stage)
      
       
        while True:
            request = self.env_conns[index].recv()
            
            if request == "start":
                old_log_policies = []
                actions = []
                values = []            
                states = []            
                rewards = []            
                dones = []
                states_info=[]                
                for i in range(num_steps):
                    states.append(state)
                    states_info.append(state_info)
                    
                    logits, value = model((state,state_info))
                    values.append(value.squeeze().detach())
                    policy = F.softmax(logits, dim=1)
                    old_m = Categorical(policy)
                    action = old_m.sample()            
                    actions.append(action.detach())
                    old_log_policy = old_m.log_prob(action)
                    old_log_policies.append(old_log_policy.detach())
                    next_state, reward, done, info_1, info = env.step(action.item())
                    state = torch.FloatTensor(np.array(next_state).reshape(1,8,126,126))

                    if torch.cuda.is_available():
                        state = state.cuda()
                        reward = torch.tensor(reward).cuda().detach()
                        done = torch.tensor(done).cuda().detach()
                    else:
                        reward = torch.tensor(reward)
                        done = torch.tensor(done)
                    rewards.append(reward)
                    dones.append(done)
                                 
                    if done == True:
                        action_queue = deque([-1/48]*64,maxlen=64) 
                        x_pos_queue = deque([40/12800]*65,maxlen=65) 
                        y_pos_queue = deque([163/1024]*65,maxlen=65)                           
                        left_x_pos_queue= deque([40/1024]*65,maxlen=65)  
                        state_info[0][0:64]=torch.FloatTensor(list(action_queue))
                        state_info[0][64:128]=torch.FloatTensor(list(x_pos_queue)[0:64])
                        state_info[0][128:192]=torch.FloatTensor(list(y_pos_queue)[0:64])
                        state_info[0][192:256]=torch.FloatTensor(list(left_x_pos_queue)[0:64])
                        state_info[0][256]=40/12800
                        state_info[0][257]=163/1024
                        state_info[0][258]=40/12800
                        state_info[0][259]=0/200
                        state_info[0][260]=0/200
                        state_info[0][261]=0
                        state_info[0][262]=0/50
                    else:                    
                        action_queue.append(action/48)                  
                        x_pos_queue.append(info["x_pos"]/12800)       
                        y_pos_queue.append(info["y_pos"]/1024)       
                        left_x_pos_queue.append(info["left_x_pos"]/1024) 
                        state_info[0][0:64]=torch.FloatTensor(list(action_queue))
                        state_info[0][64:128]=torch.FloatTensor(list(x_pos_queue)[0:64])
                        state_info[0][128:192]=torch.FloatTensor(list(y_pos_queue)[0:64])
                        state_info[0][192:256]=torch.FloatTensor(list(left_x_pos_queue)[0:64])
                        state_info[0][256]=info["x_pos"]/12800
                        state_info[0][257]=info["y_pos"]/1024
                        state_info[0][258]=info_1[0]/12800
                        state_info[0][259]=info_1[1]/200
                        state_info[0][260]=info_1[2]/200
                        if info["status"]=="small":
                            state_info[0][261]=0        
                        elif info["status"]=="tall":
                            state_info[0][261]=0.1 
                        else:
                            state_info[0][261]=0.2
                        state_info[0][262]=info["coins"]/50         
                self.env_conns[index].send([state,state_info,states,states_info,actions,values,old_log_policies,rewards,dones])
            elif request=="reset":
                
                state, _ = env.reset(seed=114514,options={})
                        
                if torch.cuda.is_available():
                    state = torch.FloatTensor(np.array(state).reshape(1,8,126,126)).cuda()
                else:
                    state = torch.FloatTensor(np.array(state).reshape(1,8,126,126))                 
                action_queue = deque([-1/48]*64,maxlen=64)
                x_pos_queue = deque([40/12800]*65,maxlen=65) 
                y_pos_queue = deque([163/1024]*65,maxlen=65)
                left_x_pos_queue = deque([40/1024]*65,maxlen=65)
                state_info = torch.zeros(1,263)    
                if torch.cuda.is_available(): 
                    state_info=state_info.cuda()          
                state_info[0][0:64]=torch.FloatTensor(list(action_queue))
                state_info[0][64:128]=torch.FloatTensor(list(x_pos_queue)[0:64])
                state_info[0][128:192]=torch.FloatTensor(list(y_pos_queue)[0:64])
                state_info[0][192:256]=torch.FloatTensor(list(left_x_pos_queue)[0:64])            
                state_info[0][256]=40/12800
                state_info[0][257]=163/1024
                state_info[0][258]=40/12800
                state_info[0][259]=0/200
                state_info[0][260]=0/200
                state_info[0][261]=0 
                state_info[0][262]=0/50     
                
                self.env_conns[index].send(state)              
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
    win_trajectory=0 
    env_eval = create_train_env(opt.world, opt.stage)
    random.seed(114514)
    np.random.seed(114514)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(114514)
    else:
        torch.manual_seed(114514)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    
    # forward_mse = nn.MSELoss(reduction='none')
    model = PPO(((8,126,126),263),len(COMPLEX_MOVEMENT))
    # RND_model = RND(envs.num_states)
    #  model = torch.compile(model)
    # model2 = PPO(envs.num_states,envs.num_actions)
    # for ((k1, v1), (k2, v2)) in zip(model.state_dict().items(), model2.state_dict().items()):
    #     assert k1 == k2, "Parameter name not match"
    #     if not torch.equal(v1, v2):
    #         print("Parameter value not match", k1)
    # print("same")
    # state_dict = torch.load("best_ppo.pt")
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #         if k[:10] == '_orig_mod.':
                
    #             name = k  # remove `module.`
    #         else:
    #             name = "_orig_mod."+k
    #         new_state_dict[name] = v
    # # model = PPO(envs.num_states, envs.num_actions)
    # model.load_state_dict(new_state_dict)
    if torch.cuda.is_available():
        model.cuda()
        # RND_model.cuda()
    model.share_memory()
    envs = MultipleEnvironments(opt.world, opt.stage,opt.num_processes,opt.num_local_steps,model)
    # optimizer = torch.optim.Adam(list(model.parameters()) + list(RND_model.predictor.parameters()), lr=opt.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # optimizer =dadaptation.DAdaptAdam(list(model.parameters()) + list(RND_model.predictor.parameters()), lr=opt.lr)
    # optimizer = dadaptation.DAdaptAdam(model.parameters(), lr=opt.lr)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch: max(0.998615**epoch,0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch: max(0.999653**epoch,0.01))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch: 1)
   
    [agent_conn.send("reset") for agent_conn in envs.agent_conns]
   
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
   

    

 
    if torch.cuda.is_available():        
        curr_states=torch.zeros(opt.num_processes,8,126,126).cuda()
        state_info=torch.zeros(opt.num_processes,263).cuda()
        states=torch.zeros(opt.num_processes,opt.num_local_steps,8,126,126).cuda()
        states_info=torch.zeros(opt.num_processes,opt.num_local_steps,263).cuda()
        actions=torch.zeros(opt.num_processes,opt.num_local_steps,1).cuda()
        values_ext=torch.zeros(opt.num_processes,opt.num_local_steps,1).cuda()
        old_log_policies=torch.zeros(opt.num_processes,opt.num_local_steps,1).cuda()
        rewards=torch.zeros(opt.num_processes,opt.num_local_steps,1).cuda()
        dones=torch.zeros(opt.num_processes,opt.num_local_steps,1).cuda()
    else:
        curr_states=torch.zeros(opt.num_processes,8,126,126)
        state_info=torch.zeros(opt.num_processes,263)
        states=torch.zeros(opt.num_processes,opt.num_local_steps,8,126,126)
        states_info=torch.zeros(opt.num_processes,opt.num_local_steps,263)
        actions=torch.zeros(opt.num_processes,opt.num_local_steps,1)
        values_ext=torch.zeros(opt.num_processes,opt.num_local_steps,1)
        old_log_policies=torch.zeros(opt.num_processes,opt.num_local_steps,1)
        rewards=torch.zeros(opt.num_processes,opt.num_local_steps,1)
        dones=torch.zeros(opt.num_processes,opt.num_local_steps,1)

    curr_episode = 0
    # explore_rate = 0.2
    
    for iter in range(opt.num_global_steps):
        start = time.time()
        # explore_rate = explore_rate*0.999
        # explore_rate = max(explore_rate,0.1)
        if curr_episode % opt.save_interval == 0 and curr_episode > 0:
            torch.save(model.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}_PPO".format(opt.saved_path, opt.world, opt.stage, curr_episode))
            # torch.save(RND_model.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}_RND".format(opt.saved_path, opt.world, opt.stage, curr_episode))
        curr_episode += 1
        # if torch.cuda.is_available():
        #     run_true = torch.cuda.FloatTensor(np.ones(opt.num_processes))
        # else:
        #     run_true = torch.FloatTensor(np.ones(opt.num_processes))

       
        model.eval()  
        [agent_conn.send("start")  for agent_conn in envs.agent_conns]
        j=0
        for agent_conn in envs.agent_conns:
            receive = agent_conn.recv() 
            if torch.cuda.is_available():    
                curr_states[j]=torch.tensor(receive[0]).cuda()
                state_info[j]=torch.tensor(receive[1]).cuda()
                states[j]=torch.tensor(receive[2]).cuda()
                states_info[j]=torch.tensor(receive[3]).cuda()
                actions[j]=torch.tensor(receive[4]).cuda()
                values_ext[j]=torch.tensor(receive[5]).cuda()
                old_log_policies[j]=torch.tensor(receive[6]).cuda()
                rewards[j]=torch.tensor(receive[7]).cuda()
                dones[j]=torch.tensor(receive[8]).cuda()
            else:
                curr_states[j]=torch.tensor(receive[0])
                state_info[j]=torch.tensor(receive[1])
                states[j]=torch.tensor(receive[2])
                states_info[j]=torch.tensor(receive[3])
                actions[j]=torch.tensor(receive[4])
                values_ext[j]=torch.tensor(receive[5])
                old_log_policies[j]=torch.tensor(receive[6])
                rewards[j]=torch.tensor(receive[7])
                dones[j]=torch.tensor(receive[8])
            j=j+1
         
          
        _, next_value_ext = model((curr_states,state_info))
        next_value_ext = next_value_ext.squeeze()
       
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values_ext = torch.cat(values_ext).detach()
      
        states = torch.cat(states)
        
        states_info = torch.cat(states_info)
 
        gae = 0
        R_ext = []
        for value, reward, done in list(zip(values_ext, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau*(1 - done)
            gae = gae + reward + opt.gamma * next_value_ext.detach() * (1 - done) - value.detach()
            next_value_ext = value
            R_ext.append(gae + value)
        R_ext = R_ext[::-1]
        R_ext = torch.cat(R_ext).detach()
        advantages_ext = R_ext - values_ext

        # gae = 0
        # R_int = []
        # for value, reward, done in list(zip(values_int, rewards_int, dones))[::-1]:
        #     gae = gae * opt.gamma_int * opt.tau
        #     gae = gae + reward + opt.gamma_int * next_value_int.detach()- value.detach()
        #     next_value_int = value
        #     R_int.append(gae + value)
        # R_int = R_int[::-1]
        # R_int = torch.cat(R_int).detach()
        # advantages_int = R_int - values_int

        # advantages_int = (advantages_int-advantages_int.mean())/(advantages_int.std()+1e-8)
        # advantages_ext = (advantages_ext-advantages_ext.mean())/(advantages_ext.std()+1e-8)
        # if iter>5:
        # advantages = advantages_int*0.25+advantages_ext
        advantages = advantages_ext
        # else:
        #     advantages = advantages_ext
        # for j in range(len(advantages)):
        #     print(f"At {j}")
        #     print(R_ext[j])
        #     print(R_int[j])
        #     print(values_ext[j])
        #     print(values_int[j])
        #     print(rewards[j//4][j-j//4*4])
        #     print(rewards_int[j//4][j-j//4*4])
        #     print(advantages_ext[j])
        #     print(advantages_int[j])

        # print(advantages_int)
        # print(advantages_ext)
        # print(rewards)
        # print(rewards_int)




        # print(len(states))
        # states= torch.transpose(states, 0, 1)
        
        # states = torch.cat(states)
        
        model.train()  
        for i in range(opt.num_epochs):
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)
            for j in range(opt.batch_size):
                batch_indices = indice[
                                int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
                                        opt.num_local_steps * opt.num_processes / opt.batch_size))]
                logits, value_ext = model((states[batch_indices],states_info[batch_indices]))  
                # predict_next_state_feature, target_next_state_feature = RND_model(next_states[batch_indices])  
                # forward_loss = opt.beta3*forward_mse(predict_next_state_feature,target_next_state_feature.detach()).mean()
                old_value = values_ext[batch_indices]         
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                if opt.adv_normalization==0:
                    nor_advantages=advantages[batch_indices]
                elif opt.adv_normalization==1:
                    nor_advantages= (advantages[batch_indices]-advantages[batch_indices].mean())/(advantages[batch_indices].std()+1e-8)
                else:
                    nor_advantages= (advantages[batch_indices])/(advantages[batch_indices].std()+1e-8)
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * nor_advantages,
                                                   torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) *
                                                   nor_advantages))
                if opt.v_clip==1:
                    critic_loss_2 = torch.mean((R_ext[batch_indices] - value_ext.squeeze()) ** 2) 
                    critic_loss_3 = torch.mean((torch.clamp(value_ext.squeeze(),(1-opt.epsilon_1)*old_value,(1+opt.epsilon_1)*old_value)-R_ext[batch_indices])**2)
                    critic_loss_1 = opt.beta1*torch.max(critic_loss_2,critic_loss_3)
                else:
                    critic_loss_1 = opt.beta1*torch.mean((R_ext[batch_indices] - value_ext.squeeze()) ** 2) / 2
                # critic_loss_2 = opt.beta1*torch.mean((R_int[batch_indices] - value_int) ** 2) / 2
                # print(value.shape)
                # print(value.squeeze().shape)
                # critic_loss_1 = torch.mean((R[batch_indices] - value) ** 2) / 2
                # critic_loss_2 = torch.mean((old_value+old_value*torch.clamp((value.squeeze()-old_value)/old_value,-opt.epsilon,opt.epsilon)-R[batch_indices])**2)/2
                # critic_loss = opt.beta1*torch.max(critic_loss_1,critic_loss_2)
                entropy_loss = -opt.beta2*torch.mean(new_m.entropy())             
                # total_loss = actor_loss + critic_loss_1 + critic_loss_2+entropy_loss+forward_loss
                total_loss = actor_loss + critic_loss_1 +entropy_loss
                # total_loss = actor_loss + critic_loss_1 + critic_loss_2+entropy_loss+forward_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
 
        
        # print("Episode: {} Total loss: {} actor loss: {} critic loss_1: {}  critic loss_2: {} entropy loss: {} forward loss: {}".format(curr_episode,total_loss,actor_loss,critic_loss_1,critic_loss_2,entropy_loss,forward_loss))
        print("Episode: {} Total loss: {} actor loss: {} critic loss_1: {}  entropy loss: {} ".format(curr_episode,total_loss,actor_loss,critic_loss_1,entropy_loss))
        print(f"Winning trajectories: {win_trajectory}, learning rate: {optimizer.param_groups[0]['lr']}")
        writer.add_scalar('Loss',total_loss, curr_episode)
        writer.add_scalar('critic_Loss_1',critic_loss_1, curr_episode)
        # writer.add_scalar('critic_Loss_2',critic_loss_2, curr_episode)
        writer.add_scalar('entropy_Loss',entropy_loss, curr_episode)
        writer.add_scalar('actor_Loss',actor_loss, curr_episode)
        # writer.add_scalar('forward_Loss',forward_loss, curr_episode)
        # writer.add_scalar('Winning trajectories',win_trajectory, curr_episode)
        writer.add_scalar('Learning rate',optimizer.param_groups[0]['lr'], curr_episode)
        scheduler.step()
        model.eval()        
       
        temp_state = torch.FloatTensor(np.array(env_eval.reset(seed=114514,options={})[0]).reshape(1,8,126,126)).cuda()
        temp_reward_sum =0
        temp_step=0
        success_=False
    
        state_info = torch.zeros(1,263)
        action_queue =  deque([-1/48]*64,maxlen=64)
        x_pos_queue = deque([40/12800]*65,maxlen=65)
        y_pos_queue = deque([163/1024]*65,maxlen=65)
        l_x_pos_queue=deque([40/1024]*65,maxlen=65)
        state_info[0][0:64]=torch.FloatTensor(list(action_queue))
        state_info[0][64:128]=torch.FloatTensor(list(x_pos_queue)[0:64])
        state_info[0][128:192]=torch.FloatTensor(list(y_pos_queue)[0:64])
        state_info[0][192:256]=torch.FloatTensor(list(l_x_pos_queue)[0:64])
        state_info[0][256]=40/12800
        state_info[0][257]=163/1024
        state_info[0][258]=40/12800
        state_info[0][259]=0/200
        state_info[0][260]=0/200
        state_info[0][261]=0
        state_info[0][262]=0/50
        time_spent=0
        stuck=0
        
        if torch.cuda.is_available():
            state_info = state_info.cuda()
        x_curr=0    


        while True:

            # Run agent on the state
            # state_info = torch.zeros(opt.num_processes,20).cuda()
            logits, value_ext  = model((temp_state,state_info))

            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()
            # actions.append(action)
            # print(action)
            # Agent performs action
            next_state, reward, done, info_1, info = env_eval.step(action)
            x_curr=info["x_pos"]
            action_queue.append(action/48)
            x_pos_queue.append(x_curr/12800)
            y_pos_queue.append(info["y_pos"]/1024)
            l_x_pos_queue.append(info["left_x_pos"]/1024)
            state_info[0][0:64]=torch.FloatTensor(list(action_queue))
            state_info[0][64:128]=torch.FloatTensor(list(x_pos_queue)[0:64])
            state_info[0][128:192]=torch.FloatTensor(list(y_pos_queue)[0:64])
            state_info[0][192:256]=torch.FloatTensor(list(l_x_pos_queue)[0:64])
            state_info[0][256]=x_curr/12800
            state_info[0][257]=info["y_pos"]/1024
            state_info[0][258]=info_1[0]/12800
            state_info[0][259]=info_1[1]/200
            state_info[0][260]=info_1[2]/200
            if info["status"]=="small":
                state_info[0][261]=0        
            elif info["status"]=="tall":
                state_info[0][261]=0.1
            else:
                state_info[0][261]=0.2
            state_info[0][262]=info["coins"]/50
            
            # print(f"{info['x_pos']} {x_curr} {x_pre_curr} {stuck}")

            time_spent = 999-info["time"]
            # print(reward)
            temp_reward_sum+=reward
                            # Update state
            temp_state = torch.FloatTensor(np.array(next_state).reshape(1,8,126,126)).cuda()
            if torch.cuda.is_available():
                temp_state=temp_state.cuda()
            temp_step+=1
            # time.sleep(0.5)
            # print(state_info)
            # print(temp_reward_sum)
            # Check if end of game
            if done:
                if info["flag_get"]:
                    success_=True
                    print(f"Success at step {temp_step} with reward {temp_reward_sum}") 
                    
                else:
                    if info_1[2]>64/200 or info_1[1]>128/200:
                        print(f"Terminated at step {temp_step} position {x_curr} with action {action} and prob {policy[0]}")    
                        stuck=-1  
                    else:
                        print(f"Died at step {temp_step} position {x_curr} with action {action} and prob {policy[0]}")
                break
            # if stuck>25:                
            #     print(f"Stuck at position {x_curr}. Killed ):")
            #     stuck=-1
            #     break

        if stuck < 0:
            print(f"Killed: Episode: {curr_episode} Reward: {temp_reward_sum} Steps: {temp_step} X_reached: {x_curr} Score:{info['score']} Coins:{info['coins']}")
        elif success_==True:
            print(f"Success: Episode: {curr_episode} Reward: {temp_reward_sum} Steps: {temp_step} X_reached: {x_curr} Time_spent: {time_spent} Score:{info['score']} Coins:{info['coins']}")
        else:
            print(f"Died: Episode: {curr_episode} Reward: {temp_reward_sum} Steps: {temp_step} X_reached: {x_curr} Time_spent: {time_spent} Score:{info['score']} Coins:{info['coins']}")
        average_reward.append(temp_reward_sum)
        moving_average = np.mean(average_reward[-100:])
        average_x.append(x_curr)
        moving_x = np.mean(average_x[-100:])
        average_score.append(info["score"])
        moving_score = np.mean(average_score[-100:])
        average_coins.append(info["coins"])
        moving_coins = np.mean(average_coins[-100:])
        print(f"Moving_reward: {moving_average} Moving_x: {moving_x} Moving_score: {moving_score} Moving_coins: {moving_coins}")
        writer.add_scalar('Reward',temp_reward_sum, curr_episode)
        writer.add_scalar('Steps',temp_step, curr_episode)
        writer.add_scalar('X_reached',x_curr, curr_episode)
        writer.add_scalar('Time_spent',time_spent, curr_episode)
        # writer.add_scalar('Fail_prob',policy[0][action], curr_episode)
        writer.add_scalar('Moving_reward',moving_average, curr_episode)
        writer.add_scalar('Moving_x',moving_x, curr_episode)
        writer.add_scalar('Success',success_, curr_episode)
        writer.add_scalar('Moving Score',moving_score, curr_episode)
        writer.add_scalar('Moving Coins', moving_coins, curr_episode)
        if temp_reward_sum>=max_reward:
            print(f"Mario breaks his record! Reaching {x_curr} with return {temp_reward_sum} coins {info['coins']} score {info['score']}. Saving the model")
            torch.save(model.state_dict(),"best_ppo_PPO.pt")  
            # torch.save(RND_model.state_dict(),"best_ppo_RND.pt")  
            # time.sleep(500)
            max_reward=temp_reward_sum
            step_max = temp_step
            far_x = x_curr
            _success_=success_
            coins_max = info['coins']
            score_max= info['score']
        # print(f"Farest: {far_x} Step: {step_max} Success? {success_}")
        # writer.add_scalar('Farest_x',far_x, curr_episode)
        print(f"Largest return: {max_reward} coins: {coins_max} scores: {score_max} Steps: {step_max} Reaching {far_x} Success? {_success_}")
        writer.add_scalar('Largest Return',max_reward, curr_episode)

        end = time.time()
        print(f"elapsed time: {end - start}")
        # model2 = PPO(env_eval.observation_space.shape,len(COMPLEX_MOVEMENT))
        # state_dict = torch.load("best_ppo.pt", map_location=lambda storage, loc: storage)
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #         if k[:10] == '_orig_mod.':
        #             name = k[10:]  # remove `module.`
        #         else:
        #             name = k
        #         new_state_dict[name] = v
        # model2.load_state_dict(new_state_dict)
        # # model.load_state_dict(state_dict)

        # if torch.cuda.is_available():
        #     model2.cuda()
        # for ((k1, v1), (k2, v2)) in zip(model.state_dict().items(), model2.state_dict().items()):
        #     assert k1 == k2, "Parameter name not match"
        #     if not torch.equal(v1, v2):
        #         print("Parameter value not match", k1)
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument('--lr', type=float, default=5*1e-4)
    parser.add_argument('--gamma', type=float, default=0.975, help='discount factor for rewards')
    parser.add_argument('--gamma_int', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=0.95, help='parameter for GAE')
    parser.add_argument('--beta2', type=float, default=0.004, help='entropy coefficient')
    parser.add_argument('--beta1', type=float, default=0.5, help='value coefficient')
    parser.add_argument('--beta3', type=float, default=0.1, help='forward coefficient')
    parser.add_argument('--epsilon', type=float, default=0.114514, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--epsilon_1', type=float, default=0.114514*4, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=50000)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=20, help="Number of steps between savings")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--adv_normalization", type=int, default=1)
    parser.add_argument("--v_clip", type=int, default=2)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    # _mp.set_start_method('spawn', force=True)
    # print("spawned")
    torch.multiprocessing.set_start_method('spawn', force=True)
    opt = get_args()
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    # torch.compile(train)
    exp_saved_path = "runs/"+"lr="+str(opt.lr)+"_gamma="+str(opt.gamma)+"_gamma_int="+str(opt.gamma_int)+"_noise="+str(opt.beta2)+"_GAE="+str(opt.tau)+"_clip="+str(opt.epsilon)+"_clip_1="+str(opt.epsilon_1)+"_adv_normalization="+str(opt.adv_normalization)+"_RND=False"+"_valueclip="+str(opt.v_clip)+"_"+datetime.now().strftime("%m_%d_%Y_%H:%M:%S")
    writer = SummaryWriter(log_dir=exp_saved_path)
    # torch.set_float32_matmul_precision("high")
    train(opt,writer)





