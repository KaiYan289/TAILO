import torch
import wandb
import numpy as np
import torch.nn as nn
import random
from NN import *
from torch.optim import Adam
from torch import autograd
import time
from tqdm import tqdm
from dataset import *
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class TanhNormalPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_sizes=(256,256), action_space=None,
                 mean_range=(-7.24, 7.24), logstd_range=(-5., 2.), eps=1e-6):
        
        # action space is a tuple (low, high), each of them is a np array
        
        super(TanhNormalPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_sizes[0])
        self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        self.mean_linear = nn.Linear(hidden_sizes[1], num_actions)
        self.log_std_linear = nn.Linear(hidden_sizes[1], num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space[1] - action_space[0]) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space[1] + action_space[0]) / 2.)
        
        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def forward(self, inputs):
        
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        mean = torch.clamp(mean, self.mean_min, self.mean_max)
        logstd = self.log_std_linear(x)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        std = torch.exp(logstd)
        pretanh_action_dist = Normal(mean, std)
        pretanh_action = pretanh_action_dist.rsample()
        action = torch.tanh(pretanh_action)
        log_prob, pretanh_log_prob = self.logprob(pretanh_action_dist, pretanh_action, is_pretanh_action=True)

        return action, pretanh_action, log_prob, pretanh_log_prob, pretanh_action_dist

    def sample(self, inputs):
        return self(inputs)[0]

    def logprob(self, pretanh_action_dist, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = torch.tanh(pretanh_action)
        else:
            pretanh_action = atanh(torch.clamp(action, -1 + self.eps, 1 - self.eps))

        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)
        log_prob = pretanh_log_prob - torch.log(1 - action ** 2 + self.eps)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob, pretanh_log_prob

    def deterministic_action(self, inputs):
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        mean = torch.clamp(mean, self.mean_min, self.mean_max)
        action = torch.tanh(mean)
        return action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(TanhNormalPolicy, self).to(device)


class Discriminator_twostate(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        middle_size = 256
        self.input_size = input_size * 2
        self.net = nn.Sequential(
            nn.Linear(self.input_size, middle_size),
            nn.Tanh(), # tanh? relu?
            nn.Linear(middle_size, middle_size),
            nn.Tanh(),
            nn.Linear(middle_size, 1))
            
    def forward(self, s1, s2):
        return self.net(torch.cat([s1, s2], dim=-1))
    
    def predict_reward(self, state, state2):
        with torch.no_grad():
            self.eval() 
            d = self(state, state2)
            s = torch.sigmoid(d)
            # log(d^E/d^O)
            # reward  = - (1/s-1).log()
            reward = s.log() - (1 - s).log()
            self.train()
            return reward 
 
def choice(st, ed, size):
    perm = torch.randperm(int(ed - st))
    return perm[:size] + st

def choice2(st, ed, size):
    return torch.randint(low=st, high=ed-size+1, size=(1,)) + torch.arange(size)

def estimate_traj_val(states, N_states):
    assert states.shape[0] >= N_states, "Error!"
    states = []
    for j in range(100):
        idx_TA_now = torch.sort(choice(0, np.arange(states.shape[0]), N_states))[0]
        states.append(states[idx_TA_now].view(-1).unsqueeze(0))
    states = torch.cat(states, dim=0)
    return disc.predict_reward(states).mean()

def train_wandering_discriminator(states_TA, states_TS, belong_TA, belong_TS, steps_TA, steps_TS, train_hyperparams, initials, N_states=2, no_log=False):
    device = torch.device('cuda:0')
    Disc = Discriminator(states_TA.shape[-1] * N_states).to(device).double()
    optimizer = torch.optim.Adam(Disc.net.parameters(), lr=train_hyperparams['lr'])
    #dataset_TA = RepeatedDataset([belong_TA, states_TA], train_hyperparams['batch_size'])
    #dataset_TS = RepeatedDataset([belong_TS, states_TS], train_hyperparams['batch_size'])
    N_TA, N_TS = int(belong_TA[-1].item()) + 1, int(belong_TS[-1].item()) + 1 # num of trajs
    print("N_TA:", N_TA, "N_TS:", N_TS) 
    def compute_grad_pen(expert_state, offline_state, lambda_):
        if lambda_ == 0: return torch.zeros(1).double().to(device)
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = expert_state 
        offline_data = offline_state

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * offline_data
        mixup_data.requires_grad = True

        disc = Disc(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
    
    flag = train_hyperparams["suffix"].find("antmaze") == -1 and train_hyperparams["suffix"].find("kitchen") == - 1
    
    # WARNING: expert_idx IS ONLY VALID IN OPENAI GYM ENVIRONMENTS! and walker2d is 100
    if flag:
        if train_hyperparams["suffix"].find("expert40") != -1:
            expert_idx = 40
        elif train_hyperparams["suffix"].find("walker2d") != -1: 
            expert_idx = 100
        else: expert_idx = 200 
        # expert_vs, policy_vs = [], []
    BS = train_hyperparams['batch_size']
    # print("NTA:", N_TA, "BS:", BS)
    for _ in tqdm(range(train_hyperparams['N'])): # epoches of trajs in TA
        random_idx = torch.randperm(N_TA)
        for i in range(N_TA // BS + 1):
            t0 = time.time()
            idx_TA = np.arange(i * BS, min((i + 1) * BS, N_TA)) # sample from traj
            assert N_TS == 1, "Error!"
            s_TA, s_TS = [], []
            banned = np.zeros(idx_TA.shape[0])
            # sample
            t1 = time.time()
            for k, j in enumerate(idx_TA):
                #print(j)
                #print(random_idx[j], N_TA - 1, random_idx.shape[0], initials.shape[0])
                st = initials[random_idx[j]].item()
                if random_idx[j] < N_TA - 1: ed = initials[random_idx[j]+1].item() 
                else: ed = N_TA - 1
                
                if ed - st < N_states: 
                    banned[k] = 1
                    # print("continue1!")
                    continue # exclude this trajectory 
                
                idx_TA_now = torch.sort(choice(st, ed, size=N_states))[0]
                # print("idx_TA_now:", idx_TA_now)
                s_TA.append(states_TA[idx_TA_now].view(-1).unsqueeze(0))
                idx_TS_now = torch.sort(choice(0, states_TS.shape[0], size=N_states))[0]
                
                s_TS.append(states_TS[idx_TS_now].view(-1).unsqueeze(0))
            t2 = time.time()
            if len(s_TA) == 0: 
                print("continue2!")
                continue # exclude this batch
            s_TA = torch.cat(s_TA, dim=0)
            s_TS = torch.cat(s_TS, dim=0)
            
            # training loss
            policy_d = Disc(s_TA)
            expert_d = Disc(s_TS)
            t3 = time.time()
            if flag: 
                expert_id, random_id = [], []
                # print(banned.shape[0], idx_TA.shape[0])
                cnt = 0
                for k, j in enumerate(idx_TA):
                    # print(k, j)
                    if banned[k] == 1:
                        cnt += 1
                        continue
                    if random_idx[j] < expert_idx: expert_id.append(k - cnt)
                    else: random_id.append(k - cnt)
                # print("random-id:", len(random_id), "idx:", idx_TA.shape[0])
                #expert_vs.append(policy_d[expert_id].cpu().detach().numpy())
                #policy_vs.append(policy_d[random_id].cpu().detach().numpy())
            if "PU" in train_hyperparams and train_hyperparams["PU"] == "rebalance": # positive-unlabeled learning, "mixed distribution" alpha = 0.7: the negative label consists of 70% expert 
                expert_loss = F.binary_cross_entropy_with_logits(
                    expert_d,
                    torch.ones(expert_d.size()).to(device))
                policy_nonexpert = F.binary_cross_entropy_with_logits(
                        policy_d,
                        torch.zeros(policy_d.size()).to(device))
                expert_nonexpert = F.binary_cross_entropy_with_logits(
                        expert_d,
                        torch.zeros(policy_d.size()).to(device))             
                # https://arxiv.org/pdf/1703.00593.pdf
                # gail_loss = train_hyperparams["PU_alpha"] * expert_loss + policy_nonexpert + (1 - train_hyperparams["PU_alpha"]) * policy_actually_nonexpert
                # max(0, policy_nonexpert - PU2_alpha * expert_nonexpert)
                if train_hyperparams["no_max"] == 1:
                    gail_loss = train_hyperparams["PU_alpha"] * expert_loss + torch.maximum(policy_nonexpert - train_hyperparams["PU_alpha"] * expert_nonexpert, torch.zeros(1).double().to('cuda:0')) 
                else:     
                    if policy_nonexpert - train_hyperparams["PU_alpha"] * expert_nonexpert >= 0 or train_hyperparams["no_max"] == 0:
                        gail_loss = train_hyperparams["PU_alpha"] * expert_loss + policy_nonexpert - train_hyperparams["PU_alpha"] * expert_nonexpert 
                    else:
                        gail_loss = train_hyperparams["PU_alpha"] * expert_nonexpert - policy_nonexpert
            else:
                expert_loss = F.binary_cross_entropy_with_logits(
                    expert_d,
                    torch.ones(expert_d.size()).to(device))
                policy_loss = F.binary_cross_entropy_with_logits(
                        policy_d,
                        torch.zeros(policy_d.size()).to(device))
                
                gail_loss = expert_loss + policy_loss
            grad_pen = compute_grad_pen(s_TS, s_TA, train_hyperparams['lipschitz'])
            t4 = time.time()     
            loss = gail_loss + grad_pen
            
            if not no_log:
                if "PU" in train_hyperparams and train_hyperparams["PU"] == "rebalance":
                    wandb.log({'expert_d': expert_d.mean(), 'policy_nonexpert': policy_nonexpert, "expert_nonexpert": expert_nonexpert, 'difference': policy_nonexpert - train_hyperparams["PU_alpha"] * expert_nonexpert, 'expert_output': torch.sigmoid(expert_d).mean(), 'offline_output': torch.sigmoid(policy_d).mean(), "expert_loss": expert_loss, "grad_pen": grad_pen, "loss": loss})
                else:
                    wandb.log({'expert_d': expert_d.mean(), 'offline_d': policy_d.mean(), 'expert_output': torch.sigmoid(expert_d).mean(), 'offline_output': torch.sigmoid(policy_d).mean(), "expert_loss": expert_loss, "policy_loss": policy_loss, "grad_pen": grad_pen, "loss": loss})
            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t5 = time.time()
            # print("sample state:", t2-t1, "forward disc:", t3-t2, "compute loss:", t4-t3, "backward+step:", t5-t4)
        """
        if not no_log:
            wandb.log({"expert_in_TA_d": np.concatenate(expert_vs).mean(), "offline_in_TA_d": np.concatenate(policy_vs).mean(), "diff_in_TA": np.concatenate(expert_vs).mean() - np.concatenate(policy_vs).mean()})
            expert_vs, policy_vs = [], []
        """
        if _ % 100 == 99 and "save_name" in train_hyperparams:
            torch.save(Disc, train_hyperparams["save_name"]+"_ep"+str(_)+".pt")
     
    return Disc
    
    
def train_twostate_discriminator(states_TA, states_TS, belong_TS, steps_TS, train_hyperparams, no_log=False):
    device = torch.device('cuda:0')
    Disc = Discriminator_twostate(states_TA.shape[-1]).to(device).double()
    # we assume that |states_TA| >> states_TS, so we can sample states_TS while iterating states_TA.
    optimizer = torch.optim.Adam(Disc.net.parameters(), lr=train_hyperparams['lr'])
    dataset_TA = RepeatedDataset([states_TA], train_hyperparams['batch_size'])
    dataset_TS = RepeatedDataset([belong_TS, states_TS, steps_TS], train_hyperparams['batch_size'])   
    dataset_TS2 = RepeatedDataset([belong_TS, states_TS, steps_TS], train_hyperparams['batch_size'])  
    
    def compute_grad_pen(expert_state, offline_state, expert_state2, lambda_):
        if lambda_ == 0: return torch.zeros(1).double().to(device)
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = expert_state 
        offline_data = offline_state

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * offline_data
        mixup_data.requires_grad = True

        disc = Disc(mixup_data, expert_state2)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
    
    # initials_TS and the indices that a TS state belongs to are given!
    
    for i in tqdm(range(train_hyperparams['N'])):
        # sample
        states_TA, (belong_TS, states_TS, steps_TS), (belong_TS2, states_TS2, steps_TS2) = dataset_TA.getitem(), dataset_TS.getitem(), dataset_TS2.getitem()
        # training loss
        policy_d = Disc(states_TA, states_TS2)
        expert_d = Disc(states_TS, states_TS2)
        
        # calculate expert label!################
        disc_decay = 0.999
        
        tag = (belong_TS == belong_TS2)
        
        expert_label = tag * disc_decay ** (steps_TS - steps_TS2).abs() 
        # print("expert label:", expert_label)
        # print(expert_label.shape) # BS
        #########################################
        
        expert_loss = -(expert_label * torch.log(torch.sigmoid(expert_d)) + (1 - expert_label) * torch.log(torch.sigmoid(1 - expert_d))).mean() # definition of cross entropy for soft labels
        
        policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(device))

        gail_loss = expert_loss + policy_loss
        grad_pen = compute_grad_pen(states_TS, states_TA, states_TS2, train_hyperparams['lipschitz'])
 
        loss = gail_loss + grad_pen
        if not no_log: wandb.log({'expert_output': torch.sigmoid(expert_d).mean(), 'offline_output': torch.sigmoid(policy_d).mean(), "expert_loss": expert_loss, "policy_loss": policy_loss, "grad_pen": grad_pen, "loss": loss})
        # optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return Disc