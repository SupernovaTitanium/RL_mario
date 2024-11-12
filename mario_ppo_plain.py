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
from parameterfree import COCOB
import time
import random, math
# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
import torch.nn.functional as F
import torch.multiprocessing as _mp
import argparse
import os
import shutil
from torch.distributions import Categorical
from collections import OrderedDict
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
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
class CustomReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.curr_score = 0
        # self.x_pos=0
        self.rs=[]
        self.ret=0
       
    def step(self, action):
        obs, reward, done, trunk, info = self.env.step(action)
        # reward += (info["score"] - self.curr_score)/10. 
        # reward += (info["x_pos"] - self.x_pos)*1.14514*2
        # if info["x_pos"]==self.x_pos:
        #     reward-=0.1
        # self.curr_score = info["score"]
        # self.x_pos = info["x_pos"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        true_reward = reward/50
        # self.ret = 0.99*self.ret+true_reward
        # self.rs.append(self.ret)
        # if len(self.rs)>1:
        #     std_rolling = np.array(self.rs).std()
        #     scale_reward = true_reward/(std_rolling+1e-8)
        # else:
        #     scale_reward = true_reward
        return obs, true_reward, done, trunk, info
    def reset(self):
        self.curr_score = 0
        # self.x_pos=0
        self.rs=[]
        self.ret=0
        return self.env.reset()

# Apply Wrappers to environment



class PPO(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.conv1 = nn.Conv2d(c, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.linear1 = nn.Linear(4608, 512)
        self.linear2 = nn.Linear(512, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, output_dim)
        self._initialize_weights()
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.linear1(x.view(x.size(0), -1)))
        x = self.linear2(x)

        return self.actor_linear(x), self.critic_linear(x)



class Resetstate(gym.Wrapper):
    def __init__(self, env):
        """Return only every `skip`-th frame"""
        super().__init__(env)


    def step(self, action):
        """Repeat action, and sum reward"""
        # Accumulate reward and repeat the same action
        obs, reward, done, trunk, info = self.env.step(action)
        if done:            
            obs, _ = self.env.reset()
            return obs, reward, done, trunk, info
        return obs,reward, done, trunk, info
    


def create_train_env(world, stage):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage),apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
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
        self.num_actions=len(RIGHT_ONLY)
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
                self.env_conns[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError
def train(opt,writer):
    far_x=0
    step_max=0
    success_=False
    average_reward=[]
    average_x=[]
    opt.saved_path =  opt.saved_path + "/"+datetime.now().strftime("%m_%d_%Y_%H:%M:%S")
    random.seed(114514)
    np.random.seed(114514)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(114514)
    else:
        torch.manual_seed(114514)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    envs = MultipleEnvironments(opt.world, opt.stage,opt.num_processes)
    model = PPO(envs.num_states, envs.num_actions)
    # model = torch.compile(model)
    # model.load_state_dict(torch.load("best_ppo.pt"))
    # state_dict = torch.load("best_ppo.pt")
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #         if k[:10] == '_orig_mod.':
    #             name = k[10:]  # remove `module.`
    #         else:
    #             name = k
    #         new_state_dict[name] = v
    # model = PPO(envs.num_states, envs.num_actions)
    # model.load_state_dict(new_state_dict)
    if torch.cuda.is_available():
        model.cuda()
   
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    optimizer = COCOB(model.parameters(),alpha=100)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch: max(0.998269**epoch,0.01))
    
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
   
    curr_states = [np.array(agent_conn.recv()[0]).reshape(1,4,84,84) for agent_conn in envs.agent_conns]


    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))	
    
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()
    curr_episode = 0
    # explore_rate = 0.2
    
    while True:
        # explore_rate = explore_rate*0.999
        # explore_rate = max(explore_rate,0.1)
        if curr_episode % opt.save_interval == 0 and curr_episode > 0:
            torch.save(model.state_dict(),"{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_episode))
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []

        # if torch.cuda.is_available():
        #     run_true = torch.cuda.FloatTensor(np.ones(opt.num_processes))
        # else:
        #     run_true = torch.FloatTensor(np.ones(opt.num_processes))

        for local_steps in range(opt.num_local_steps):  
            # print(local_steps)
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            # policy1 = torch.ones(opt.num_processes//2,envs.num_actions).cuda()*explore_rate
            # policy2 = torch.zeros(opt.num_processes-opt.num_processes//2,envs.num_actions).cuda()
            # policy3 = torch.cat((policy1,policy2),dim=0)
            # print(policy.shape)
            # print(policy)
            # print(policy3.shape)
            # print(policy3)
            # policy = (policy+policy3)/(policy+policy3).sum(dim=1,keepdim=True)
            # print(policy)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            if torch.cuda.is_available():
                [agent_conn.send(("step", act))  for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act))  for agent_conn, act  in zip(envs.agent_conns, action)]

            state, reward, done, trunc,info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
           
            state = torch.from_numpy(np.array(state).reshape(opt.num_processes,4,84,84))
           
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)


            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)

      
            rewards.append(reward)
            dones.append(done)
            curr_states = state            
            # print(done)
            
            # run_true = (1-done)*(1-trunc)*(1-flags)*run_true        
            # for w in range(opt.num_processes):
            #     if run_true[w]<1:   
            #         envs.agent_conns[w].send(("reset", None))              
            #         state = np.array(envs.agent_conns[w].recv()[0]).reshape(1,4,84,84)  
            #         state = torch.from_numpy(state)
            #         if torch.cuda.is_available():
            #             state = state.cuda()                                    
            #         run_true[w]=1
            #         curr_states[w] = state
         
          
        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        # print(actions)
        # print(values)
        
        # actions = torch.cat(actions)
        
        # gae = 0
        # R = []
        # advantages=[]
        # for value, reward, done in list(zip(values, rewards, dones))[::-1]:
        #     value= torch.FloatTensor(value).cpu()
        #     reward=torch.FloatTensor(reward).cpu()
        #     done=torch.FloatTensor(done).cpu()
        #     gae = gae * opt.gamma * opt.tau * done
        #     gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
        #     next_value = value
        #     R.append(gae + value)
        #     advantages.append(gae)
        
        # R = R[::-1]
        # R = torch.cat(R).detach()
        
        # advantages = advantages[::-1]
        # advantages = torch.cat(advantages).detach()

        
        # old_log_policies = [item for sublist in old_log_policies for item in sublist]
        # old_log_policies = torch.FloatTensor(old_log_policies)
        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau*(1 - done)
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values

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
                logits, value = model(states[batch_indices])
                # old_value = values[batch_indices]
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
                # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
                # print(value.shape)
                # print(value.squeeze().shape)
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                # critic_loss_2 = F.smooth_l1_loss(old_value+(value.squeeze()-old_value)*torch.clamp(1,1.0-opt.epsilon,1.0+opt.epsilon),R[batch_indices])
                # critic_loss = torch.max(critic_loss_1,critic_loss_2)
                entropy_loss = -torch.mean(new_m.entropy())
                total_loss = actor_loss + opt.beta1*critic_loss + opt.beta2 * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        # print(optimizer.param_groups[0]['lr'])
        # scheduler.step()
        print("Episode: {} Total loss: {} actor loss: {} critic loss: {} entropy loss: {}".format(curr_episode,total_loss,actor_loss,critic_loss,entropy_loss))
        writer.add_scalar('Loss',total_loss, curr_episode)
        writer.add_scalar('critic-Loss',critic_loss, curr_episode)
        writer.add_scalar('entropy-Loss',entropy_loss, curr_episode)
        writer.add_scalar('actor-Loss',actor_loss, curr_episode)
     
        model.eval()        
        env_eval = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(opt.world, opt.stage),apply_api_compatibility=True)
        env_eval = JoypadSpace(env_eval, RIGHT_ONLY)
        env_eval = SkipFrame(env_eval, skip=4)
        env_eval = GrayScaleObservation(env_eval)
        env_eval = ResizeObservation(env_eval, shape=84)
        env_eval = CustomReward(env_eval)
        env_eval = FrameStack(env_eval, num_stack=4)
        temp_state = torch.FloatTensor(np.array(env_eval.reset()[0]).reshape(1,4,84,84)).cuda()
        temp_reward_sum =0
        temp_step=0

  
        time_spent=0
        stuck=0

        x_curr=0        
        while True:

            # Run agent on the state
        
            logits, value = model(temp_state)

            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()
            # actions.append(action)
            # print(action)
            # Agent performs action
            next_state, reward, done, trunc, info = env_eval.step(action)
            if info["x_pos"]>x_curr:
                stuck=0
            if info["x_pos"]==x_curr:
                stuck+=1
            x_curr=info["x_pos"]
        
            time_spent = 999-info["time"]
            temp_reward_sum+=reward
                            # Update state
            temp_state = torch.FloatTensor(np.array(next_state).reshape(1,4,84,84)).cuda()
            temp_step+=1
         
            # Check if end of game
            if done:
                if info["flag_get"]:
                    success_=True
                else:
                    print(f"Died with action {action} at step {temp_step} position {x_curr} with prob {policy[0][action]}")
            # if stuck>25:                
            #     print(f"Stuck at position {x_curr}. Killed ):")
            #     stuck=-1
            #     break
            if done:
                break
        if stuck < 0:
            print(f"Episode: {curr_episode} Reward: {temp_reward_sum} Steps: Killed X_reached: {x_curr}")
        else:
            print(f"Episode: {curr_episode} Reward: {temp_reward_sum} Steps: {temp_step} X_reached: {x_curr} Time_spent: {time_spent}")

        average_reward.append(temp_reward_sum)
        moving_average = np.mean(average_reward[-50:])
        average_x.append(x_curr)
        moving_x = np.mean(average_x[-50:])
        print(f"Moving_reward: {moving_average}")
        print(f"Moving_x: {moving_x}")
        writer.add_scalar('Reward',temp_reward_sum, curr_episode)
        writer.add_scalar('Steps',temp_step-1, curr_episode)
        writer.add_scalar('X_reached',x_curr, curr_episode)
        writer.add_scalar('Time_spent',time_spent, curr_episode)
        writer.add_scalar('Fail_prob',policy[0][action], curr_episode)
        writer.add_scalar('Moving_reward',moving_average, curr_episode)
        writer.add_scalar('Moving_x',moving_x, curr_episode)
        if x_curr>far_x:
            print(f"Mario breaks his record! Reaching {x_curr}. Saving the model")
            torch.save(model.state_dict(),"best_ppo.pt")  
            # time.sleep(500)
            far_x=x_curr
            step_max = temp_step
        print(f"Farest: {far_x} Step: {step_max} Success? {success_}")
        writer.add_scalar('Farest_x',far_x, curr_episode)
        
        # model2 = PPO(env_eval.observation_space.shape,len(RIGHT_ONLY))
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
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=0.95, help='parameter for GAE')
    parser.add_argument('--beta2', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--beta1', type=float, default=0.5, help='value coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=25, help="Number of steps between savings")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--adv_normalization", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    opt = get_args()
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    # torch.compile(train)
    exp_saved_path = "runs/"+"lr="+str(opt.lr)+"_noise="+str(opt.beta2)+"_clip="+str(opt.epsilon)+"_adv_normalization="+str(opt.adv_normalization)+"_exponential_annealing"
    writer = SummaryWriter(log_dir=exp_saved_path)
    # torch.set_float32_matmul_precision("high")
    train(opt,writer)




