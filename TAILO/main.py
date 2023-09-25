from NN import *
from discriminator import train_discriminator
from advance_NN import *
from dataset import Testdataset, RepeatedDataset, add_terminals, construct_dataset, concatlist, list2gen, get_dataset
import argparse, subprocess
import wandb
import gym
from tqdm import tqdm
import d4rl
import time
from datetime import datetime
import torch
from torch.optim import Adam
import numpy as np
import random
from maze_model import *
from utils import get_git_diff, git_commit

device = torch.device('cuda:0')

def get_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="")
    parser.add_argument("--skip_suffix_TA", type=str, default="")
    parser.add_argument("--skip_suffix_TS", type=str, default="")
    parser.add_argument("--normalize_obs", type=int, default=1)
    parser.add_argument("--wbc_lr", type=float, default=0.0001)
    parser.add_argument("--lr_disc", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--N", type=int, default=20000)
    parser.add_argument("--R_N", type=int, default=40000)
    parser.add_argument("--R_N_pretrain", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1234567)
    parser.add_argument("--eval_deter", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--coeff_scale", type=float, default=1.25)
    parser.add_argument("--gamma", type=float, default=0.998)
    parser.add_argument("--N_states", type=int, default=2)
    parser.add_argument("--mode", type=str, default="expdecay")
    parser.add_argument('--lipschitz', type=float, default=10)
    parser.add_argument('--lipschitz_pretrain', type=float, default=10)
    parser.add_argument('--PU', type=int, default=5)
    parser.add_argument('--PU1_alpha', type=float, default=0.8)
    parser.add_argument('--PU2_alpha', type=float, default=0.2)
    parser.add_argument("--PU1_morepositive", type=float, default=0) 
    parser.add_argument('--no_max', type=int, default=1) # same as beforew  
    parser.add_argument('--auto', type=int, default=0)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--EMA", type=int, default=0)
    args = parser.parse_args()
    return args

def evaluation(policy, env, eval_use_argmax, mean_state=None, std_state=None, is_kitchen=False, is_kitchen_goal=False, goal_name=""):
        T = 10
        policy.eval()
        points_x, points_y = [], []
        tot_r, tot_l, tot_cnt = 0, 0, 0
        Rs = []
        tot_bottomburner, tot_kettle, tot_microwave, tot_switch, tot_slider = 0, 0, 0, 0, 0
        # print(env.action_space.shape)
        for i in range(T):
            # TODO: an evaluation of a whole episode in the environment.
            tag_bottomburner, tag_kettle, tag_microwave, tag_switch, tag_slider = 0, 0, 0, 0, 0
            state = env.reset()
            print("state - start of episode:", state)
            while True:           
                if mean_state is not None and std_state is not None: state = (state - mean_state.cpu().numpy()) / std_state.cpu().numpy()
                if not eval_use_argmax:
                    action = policy.sample(torch.from_numpy(state).to(device).double())
                else:
                    action = policy.deterministic_action(torch.from_numpy(state).to(device).double())
                if isinstance(policy, ActorDiscrete): 
                    action = action.item()
                else: action = action.detach().cpu().numpy().reshape(-1)
                state, reward, done, _ = env.step(action) 
                # print("state:", state, "action:", action, "reward:", reward)
                
                if is_kitchen and not is_kitchen_goal:
                    if "kettle" not in env.tasks_to_complete:
                        tag_kettle = 1
                    if "light switch" not in env.tasks_to_complete:
                        tag_switch = 1
                    if "bottom burner" not in env.tasks_to_complete:
                        tag_bottomburner = 1
                    if "microwave" not in env.tasks_to_complete:
                        tag_microwave = 1
                    if "slide cabinet" not in env.tasks_to_complete:
                        tag_slider = 1
                if is_kitchen_goal: 
                    if reward > 0 and goal_name.split("-")[1] not in env.tasks_to_complete:
                        reward = 1.0
                        done = True
                    else:
                        reward = 0.0
                        # done = False # if all the tasks are complete, then surely the target task is completed
                
                #print("new_state:", state)
                tot_r += reward
                #print("step!")
                if done: # do not use normalization with visualization together on Navigation!
                    Rs.append(tot_r)
                    tot_r = 0
                    break
            
            tot_bottomburner += tag_bottomburner
            tot_kettle += tag_kettle
            tot_microwave += tag_microwave
            tot_switch += tag_switch
            tot_slider += tag_slider
            
        policy.train()
        Rs = np.array(Rs)
        
        if is_kitchen and not is_kitchen_goal:
            wandb.log({"suc_bottomburner": tot_bottomburner / T, "suc_kettle": tot_kettle / T, "suc_microwave": tot_microwave / T, "suc_switch": tot_switch / T, "suc_slider": tot_slider / T}, commit=False)
        wandb.log({"average_reward": Rs.mean(), "max_reward": Rs.max(), 'min_reward': Rs.min(), "std_reward": Rs.std()})
        return


def get_v(initials, N_states, states_TA, disc, args):
    new_R = []
    for i in tqdm(range(len(initials))):
        if i < len(initials) - 1: 
            st, ed = initials[i].item(), initials[i+1].item()
        else: 
            st, ed = initials[i].item(), states_TA.shape[0]
        if ed - st < N_states: v = 0 # too short!
        else:
            states = []
            for j in range(100):
                idx_TA_now = torch.sort(choice(st, ed, N_states))[0]
                if args.env_name == "antmaze" and args.skip_suffix_TS == "mismatch":
                    states.append(states_TA[idx_TA_now, :2].view(-1).unsqueeze(0)) 
                else: states.append(states_TA[idx_TA_now].view(-1).unsqueeze(0))
            states = torch.cat(states, dim=0)
            
            v = torch.exp(disc.predict_reward(states).mean() * args.coeff_scale)
        new_R.append(v * torch.ones(ed-st).double().to(device))
        #f.write(str(v.item())+"\n")
    new_R = torch.cat(new_R, dim=0).detach()
    return new_R

def get_v_single(states_TA, disc, args):
    new_R = []
    BS = 4096
    for i in tqdm(range(states_TA.shape[0] // BS + 1)):
        states = states_TA[i*BS:(i+1)*BS]
        
        if args.env_name == "antmaze" and args.skip_suffix_TS == "mismatch":
            states = states[:, :2]
        
        v = disc.predict_reward(states).view(-1)
        new_R.append(v)
        #f.write(str(v.item())+"\n")
    new_R = torch.cat(new_R, dim=0).detach()
    return new_R

def train(TA_dataset, TS_dataset, args):
    
    states_TA, actions_TA, next_states_TA, terminals_TA, steps_TA, rewards_TA = get_dataset(TA_dataset)
    states_TS, actions_TS, next_states_TS, terminals_TS, steps_TS, rewards_TS = get_dataset(TS_dataset)
    the_terminal = torch.cat([torch.zeros_like(torch.from_numpy(TA_dataset[0]["state"])).to(device).double(), torch.ones(1).to(device).double()]).view(1, -1)
    states_TA, actions_TA, next_states_TA, terminals_TA, steps_TA, rewards_TA = add_terminals(states_TA, actions_TA, next_states_TA, terminals_TA, steps_TA, rewards_TA, 0, the_terminal)
    states_TS, actions_TS, next_states_TS, terminals_TS, steps_TS, rewards_TS = add_terminals(states_TS, actions_TS, next_states_TS, terminals_TS, steps_TS, rewards_TS, 0, the_terminal)
    
    #print(states_TA.mean(dim=0), states_TA.std(dim=0))
    #exit(0)
    
    initials = torch.nonzero(steps_TA.view(-1) == 0)
    
    TA_mean, TA_std = None, None

    if args.normalize_obs == 1:
        TA_mean, TA_std = states_TA.mean(dim=0), states_TA.std(dim=0) + (1e-4 if args.env_name != "kitchen" else 1e-3) # 1e-10
        print("mean:", TA_mean, "std:", TA_std)
        states_TA = (states_TA - TA_mean.view(1, -1)) / TA_std.view(1, -1)
        next_states_TA = (next_states_TA - TA_mean.view(1, -1)) / TA_std.view(1, -1)
        
        if args.env_name == "antmaze" and args.skip_suffix_TS == "mismatch": 
            states_TS = (states_TS - TA_mean[:4].view(1, -1)) / TA_std[:4].view(1, -1)
            next_states_TS = (next_states_TS - TA_mean[:4].view(1, -1)) / TA_std[:4].view(1, -1)
        else:
            states_TS = (states_TS - TA_mean.view(1, -1)) / TA_std.view(1, -1)
            next_states_TS = (next_states_TS - TA_mean.view(1, -1)) / TA_std.view(1, -1)
    
    disc_hyperparam = {"batch_size": min(512, states_TS.shape[0]), "lr": args.lr_disc, "N": args.R_N, "lipschitz": args.lipschitz, "no_max": args.no_max}
    
    if args.mode == "whole-multistep": # needs a separate reward function
        N_states = args.N_states 
        disc_hyperparam["suffix"] = str(N_states) + "_" + args.env_name +"_"+args.skip_suffix_TA+"_"+args.skip_suffix_TS
        belong_TA = torch.zeros(steps_TA.shape[0]) 
        belong_TS = torch.zeros(steps_TS.shape[0])
        for i in range(len(initials)):
            if i < len(initials) - 1:
                belong_TA[initials[i]:initials[i+1]] = i
            else: 
                belong_TA[initials[i]:] = i
        
        disc_hyperparam["save_name"] =  "model/disc-"+str(args.R_N)+"_"+args.skip_suffix_TA+"_"+args.skip_suffix_TS+"_"+str(args.coeff_scale)+"_"+args.mode+"_"+args.env_name
        
        if args.PU in [1, 3, 4, 5]: # positive-unlabeled learning, "reliable negative sample"  alpha = 0.7: the least unlikely 70% will become the negative sample
            disc_hyperparam['N'] = args.R_N_pretrain
            disc_hyperparam['lipschitz'] = args.lipschitz_pretrain
            print("pseudolabeling in progress...")
            disc_hyperparam["PU"] = "pretrain"
            if args.PU in [3, 5]: disc_hyperparam["PU"] += "_rebalance"
            pre_disc = train_wandering_discriminator(states_TA, states_TS, belong_TA, belong_TS, steps_TA, steps_TS, disc_hyperparam, initials, N_states=N_states)
            print("pseudolabeling disc trained!")
            disc_hyperparam['N'] = args.R_N
            disc_hyperparam["PU"] = "formal"
            disc_hyperparam['lipschitz'] = args.lipschitz
            pre_R = get_v(initials, N_states, states_TA, pre_disc, args)
            pre_R_by_traj = []
            print("count by traj...")
            for i in tqdm(range(initials.shape[0])):
                if i < len(initials) - 1: 
                    st, ed = initials[i].item(), initials[i+1].item()
                else: 
                    st, ed = initials[i].item(), states_TA.shape[0]
                pre_R_by_traj.append(pre_R[st:ed].mean().item())
            pre_R_by_traj = torch.tensor(pre_R_by_traj)
            quantile_alpha_by_traj = torch.quantile(pre_R_by_traj, args.PU1_alpha)
            print("quantile_alpha:", quantile_alpha_by_traj)
            idx_by_traj = torch.nonzero(pre_R_by_traj < quantile_alpha_by_traj)
            idx = []
            for i in idx_by_traj:
                if i < len(initials) - 1: 
                    st, ed = initials[i].item(), initials[i+1].item()
                else: 
                    st, ed = initials[i].item(), states_TA.shape[0]
                idx.append(torch.tensor(np.arange(st, ed)))
            idx = torch.cat(idx).to('cuda:0') 
            print("numtraj:", len(idx_by_traj), len(initials) * args.PU1_alpha) 
            states_TA_final = states_TA[idx]
            
            belong_TA_final = belong_TA[idx]
            steps_TA_final  = steps_TA[idx] 
            initials_final  = initials[idx_by_traj]
            
            if args.PU1_morepositive > 1e-10:
                morepositive_by_traj = torch.nonzero(pre_R_by_traj > torch,quantile(pre_R_by_traj, 1 - args.PU1_morepositive))
                safe_idx = []
                for i in morepositive_by_traj:
                    if i < len(initials) - 1: 
                        st, ed = initials[i].item(), initials[i+1].item()
                    else: 
                        st, ed = initials[i].item(), states_TA.shape[0]
                    safe_idx.append(torch.tensor(np.arange(st, ed)))
                safe_idx = torch.cat(safe_idx).to('cuda:0') 
                states_TS_final = torch.cat([states_TS, states_TA[safe_idx]], dim=0)
                wandb.log({"total-positive": states_TS_final.shape[0]})
                # exit(0)
            else: states_TS_final = states_TS
            if disc_hyperparam["suffix"].find("expert40") != -1:
                    expert_num = 40
            elif suffix.find("walker2d") != -1: 
                expert_num = 100
            else: expert_num = 200  
            wandb.log({"expert_num": expert_num, "ratio_expert_as_negative": torch.count_nonzero(idx < initials[expert_num]) / initials[expert_num], "ratio_random_as_negative": torch.nonzero_count(idx >= initials[expert_num]) / (states_TA.shape[0] - initials[expert_num])})
            
            print('states_TA_final_shape:', states_TA_final.shape[0]) 
        else: 
            states_TA_final, belong_TA_final, steps_TA_final, initials_final, states_TS_final = states_TA, belong_TA, steps_TA, initials, states_TS
        if args.PU in [2, 3, 4]: # positive-unlabeled learning, alpha is "positive class prior" 
            disc_hyperparam["PU"], disc_hyperparam["PU_alpha"] = "rebalance", args.PU2_alpha 
        else: disc_hyperparam["PU"] = ""
        
        if args.env_name == "antmaze" and args.skip_suffix_TS == "mismatch":
            states_TA_final = states_TA_final[:, :2]
            states_TS_final = states_TS_final[:, :2]
        
        disc = train_wandering_discriminator(states_TA_final, states_TS_final, belong_TA_final, belong_TS, steps_TA_final, steps_TS, disc_hyperparam, initials_final, N_states=N_states)
        
        new_R = get_v(initials, N_states, states_TA, disc, args)

    else:
        disc_hyperparam = {"batch_size": min(512, states_TS.shape[0]), "lr": args.lr_disc, "N": args.R_N, "lipschitz": args.lipschitz, "no_max": args.no_max, "suffix": args.env_name +"_"+args.skip_suffix_TA+"_"+args.skip_suffix_TS}
            
        if disc_hyperparam["suffix"].find("expert40") != -1:
            expert_num = 40
        elif disc_hyperparam["suffix"].find("recoilRFE") != -1:
            expert_num = 30 
        elif disc_hyperparam["suffix"].find("walker2d") != -1: 
            expert_num = 100
        else: expert_num = 200
        
        is_expert_traj = torch.cat([torch.ones(initials[expert_num]), torch.zeros(states_TA.shape[0] - initials[expert_num])]).to(device)
        
        print(is_expert_traj.shape)
        
        if args.mode not in ["random", "ideal", "load", "uniform"]:
            if args.PU in [1, 3, 4, 5]: # positive-unlabeled learning, "reliable negative sample"  alpha = 0.7: the least unlikely 70% will become the negative sample
                
                print("pseudolabeling in progress...")
                disc_hyperparam["PU"] = "pretrain"
                disc_hyperparam["EMA"] = (args.EMA in [1, 3])
                if args.PU in [3, 5]: 
                    disc_hyperparam["PU"] += "_rebalance"  
                    disc_hyperparam["PU_alpha"] = args.PU2_alpha
                disc_hyperparam['N'] = args.R_N_pretrain
                disc_hyperparam["lipschitz"] = args.lipschitz_pretrain
                if args.env_name == "antmaze" and args.skip_suffix_TS == "mismatch": pre_disc, ema = train_discriminator(states_TA[:, :2], states_TS[:, :2], disc_hyperparam, is_expert_traj=is_expert_traj)
                else: pre_disc, ema = train_discriminator(states_TA, states_TS, disc_hyperparam, is_expert_traj=is_expert_traj)
                print("pseudolabeling disc trained!")
                
                if args.EMA in [1, 3]: 
                    ema.apply_shadow()
                
                disc_hyperparam["PU"] = "formal" 
                disc_hyperparam['N'] = args.R_N
                disc_hyperparam['lipschitz'] = args.lipschitz
                pre_R = get_v_single(states_TA, pre_disc, args)
                
                pre_R_by_traj = []
                print("count by traj...")
                for i in tqdm(range(initials.shape[0])):
                    if i < len(initials) - 1: 
                        st, ed = initials[i].item(), initials[i+1].item()
                    else: 
                        st, ed = initials[i].item(), states_TA.shape[0]
                    pre_R_by_traj.append(pre_R[st:ed].mean().item())
                pre_R_by_traj = torch.tensor(pre_R_by_traj)
                quantile_alpha_by_traj = torch.quantile(pre_R_by_traj, args.PU1_alpha)
                print("quantile_alpha:", quantile_alpha_by_traj)
                idx_by_traj = torch.nonzero(pre_R_by_traj < quantile_alpha_by_traj)
                idx = []
                for i in idx_by_traj:
                    if i < len(initials) - 1: 
                        st, ed = initials[i].item(), initials[i+1].item()
                    else: 
                        st, ed = initials[i].item(), states_TA.shape[0]
                    idx.append(torch.tensor(np.arange(st, ed)))
                idx = torch.cat(idx).to('cuda:0') 
                print("numtraj:", len(idx_by_traj), len(initials) * args.PU1_alpha) 
                
                states_TA_final = states_TA[idx]
                 
                is_expert_traj_final = is_expert_traj[idx]
                wandb.log({"expert_in_new_TA": is_expert_traj_final.sum(), "expert_in_old_TA": is_expert_traj.sum()})
                
                if args.PU1_morepositive > 1e-10:
                    morepositive_by_traj = torch.nonzero(pre_R_by_traj > torch.quantile(pre_R_by_traj, 1 - args.PU1_morepositive))
                    safe_idx = []
                    for i in morepositive_by_traj:
                        if i < len(initials) - 1: 
                            st, ed = initials[i].item(), initials[i+1].item()
                        else: 
                            st, ed = initials[i].item(), states_TA.shape[0]
                        safe_idx.append(torch.tensor(np.arange(st, ed)))
                    safe_idx = torch.cat(safe_idx).to('cuda:0')
                    
                    states_TS_final = torch.cat([states_TS, states_TA[safe_idx]], dim=0)
                    print(states_TS_final.shape)
                    print("selected traj:", morepositive_by_traj)
                    wandb.log({"total-positive": states_TS_final.shape[0]})
                    
                else: states_TS_final = states_TS
                wandb.log({"expert_num": expert_num, "ratio_expert_as_negative": torch.count_nonzero(idx < initials[expert_num]) / initials[expert_num], "ratio_random_as_negative": torch.count_nonzero(idx >= initials[expert_num]) / (states_TA.shape[0] - initials[expert_num])})
                # print('states_TA_final_shape:', states_TA_final.shape, states_TA.shape * args.PU1_alpha) 
                # exit(0)
            else: 
                states_TA_final, states_TS_final, is_expert_traj_final = states_TA, states_TS, is_expert_traj
                
            if args.env_name == "antmaze" and args.skip_suffix_TS == "mismatch":
                    states_TA_final = states_TA_final[:, :2]
                    states_TS_final = states_TS_final[:, :2]
                
            if args.PU in [2, 3, 4]: # positive-unlabeled learning, alpha is "positive class prior"
                disc_hyperparam["PU"], disc_hyperparam["PU_alpha"] = "rebalance", args.PU2_alpha
            else: disc_hyperparam["PU"] = ""
        else: 
            states_TA_final, states_TS_final = states_TA, states_TS
            #if args.env_name == "antmaze" and args.skip_suffix_TS == "mismatch":
            #    states_TA_final = states_TA_final[:, :2]
            #    states_TS_final = states_TS_final[:, :2]
        print(states_TA_final.shape, states_TS_final.shape)
        
        if args.mode not in ["random", "ideal", "load", "uniform"]:
            disc_hyperparam["EMA"] = (args.EMA in [2, 3])
                
            disc, ema = train_discriminator(states_TA_final, states_TS_final, disc_hyperparam, is_expert_traj=is_expert_traj_final)
        
            if args.EMA in [2, 3]: 
                ema.apply_shadow()
            
            # exit(0)
            R = []
            for i in range(states_TA.shape[0] // 4096 + 1):
                if args.env_name == "antmaze" and args.skip_suffix_TS == "mismatch":  R.append(disc.predict_reward(states_TA[i*4096:(i+1)*4096, :2]))
                else: R.append(disc.predict_reward(states_TA[i*4096:(i+1)*4096]))
            R = torch.cat(R, dim=0)
        
        if args.mode in ["whole", "expdecay"]:
            R = torch.exp(R * args.coeff_scale)
        
        elif args.mode == "linear_decay":
            R = args.coeff_scale * torch.sigmoid(R)
        
        elif args.mode == "expratio_decay":
            R = torch.exp(torch.sigmoid(R) * args.coeff_scale)
            import matplotlib.pyplot as plt
            def draw_bins(data, bins, true_bins=None):
                hist, bin_edges = np.histogram(data, bins)
                ax.bar(range(len(hist)), hist, width=1)
                ax.set_xticks([i - 0.5 for i in range(len(bins))])
                # Set the xticklabels to a string that tells us what the bin edges were
                if true_bins is None: ax.set_xticklabels([i for i in bins])
                else: ax.set_xticklabels([true_bins[i] for i in bins])
            plt.clf()
            fig, ax = plt.subplots()
            bins = [0, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10, 100, 1000, 3000, 5000, 7000, 10000, 20000, 30000, 50000] #[0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 
            ax.set_yscale("log")
            draw_bins(R.detach().cpu().numpy(), bins)
            plt.savefig("debug-pic/expratio-decay-"+args.env_name+"_"+args.skip_suffix_TA+"_"+args.skip_suffix_TS+".jpg")
            plt.clf()
            # exit(0)
        #f = open(args.env_name + "_v"+args.skip_suffix_TA+"_"+args.skip_suffix_TS+".txt", "w")
        if args.mode == "whole":
            for i in range(len(initials)):
                if i < len(initials) - 1: 
                    v, s = R[initials[i]:initials[i+1]].mean(), initials[i+1]-initials[i]
                else: 
                    v, s = R[initials[i]:].mean(), R.shape[0]-initials[i] 
                new_R.append(v * torch.ones(s).double().to(device))
                #f.write(str(v.item())+"\n")
            new_R = torch.cat(new_R, dim=0)
        elif args.mode in ["expdecay", "expratio_decay", "linear_decay"]: # exponential-decay
            new_R = []
            gamma = args.gamma
            for i in tqdm(range(len(initials))):
                t0 = time.time()
                if i < len(initials) - 1:
                    st, ed = initials[i].item(), initials[i+1].item()
                else:
                    st, ed = initials[i].item(), R.shape[0]
                traj_R, now_R = [R[ed-1] / (1 - gamma)], R[ed-1] / (1 - gamma)
                t1 = time.time()
                #print("st:", st, "ed:", ed, "traj-R:", traj_R)
                for j in reversed(range(st, ed-1)):
                    now_R = gamma * now_R + R[j]
                    traj_R.append(now_R.view(-1))
                #print("new-R:", new_R, "traj-R:", traj_R)
                t2 = time.time()
                traj_R.reverse()
                new_R.extend(traj_R)
                #for j in range(len(traj_R)):
                #    f.write(str(traj_R[j].item())+"\n")
                #    f.flush()
                #f.write("--------------------------------\n")
                t3 = time.time()
                print(t1-t0, t2-t1, t3-t2)
            new_R = torch.cat(new_R, dim=0)
            
            
        elif args.mode == "random": # random!
            new_R = torch.cat([torch.rand(initials[40].item()) * 0.9 + 0.1, torch.rand(R.shape[0] - initials[40]) * 0.045 + 0.005]).double().to(device) 
        elif args.mode == "uniform":
            new_R = torch.ones(states_TA.shape[0]).double().to(device)
            print("new_R:", new_R, new_R.std(), new_R.mean())
            # exit(0)
        elif args.mode == "ideal":
            new_R = torch.cat([torch.ones(initials[40].item()), torch.zeros(R.shape[0] - initials[40])]).double().to(device)
        elif args.mode == "load":
            new_R = torch.load("coeff_TA_299_threeparts.pt").detach()
            newer_R = torch.zeros_like(new_R)
            for i in range(len(initials)):
                if i < len(initials) - 1:
                    st, ed = initials[i].item(), initials[i+1].item()
                else:
                    st, ed = initials[i].item(), R.shape[0]
                    
                def laplacian_smoothing(v):
                    v2 = v / 2
                    v2[0] += v[1] / 2
                    v2[-1] += v[-1] / 2
                    v2[1:] += v[:-1] / 4
                    v2[:-1] += v[1:] / 4
                    return v2
                
                def exponential_smoothing(array, factor): # adopted from wandb https://docs.wandb.ai/v/zh-hans/dashboard/features/standard-panels/line-plot/smoothing
                    res = torch.zeros_like(array)
                    last = 0
                    for i in range(array.shape[0]):
                        last = last * factor + (1 - factor) * array[i]
                        debias_weight = 1 - factor ** (i + 1)
                        res[i] = last / debias_weight
                    return res
                
                # newer_R[st:ed] = exponential_smoothing(new_R[st:ed], 0.9)
                #newer_R[st:ed] = laplacian_smoothing(new_R[st:ed]) 
                newer_R[st:ed] = new_R[st:ed].mean()
                # newer_R[st:ed] = new_R[st:ed]
            new_R = newer_R 
            assert new_R.shape[0] == states_TA.shape[0], "Error data!"
    
    print(new_R, new_R.max(), new_R.min())
    new_R /= new_R.max()
    
    if args.env_name != "pointmaze":
        env = gym.make(args.env_name+"-"+(("random" if args.env_name != "antmaze" else "umaze") if args.env_name != "kitchen" else "mixed")+"-"+("v2" if args.env_name != "kitchen" else "v0"))
    else:
        GOAL_MAPPING = {'right':1, 'left':2, 'down':3, 'up':4}
        GOAL_LOC_MAPPING = {'right':[7,4], 'left':[1,4], 'down': [4,1], 'up':[4,7]}
        init_target = GOAL_LOC_MAPPING['left']
        goal_id = GOAL_MAPPING['left']
        maze = EXAMPLE_MAZE
        env = MazeEnv(maze, reset_target=False, init_target=init_target)
    
    action_low, action_high = env.action_space.low, env.action_space.high
    n_state, n_action = env.observation_space.shape[0], env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n
    policy = ActorTanh(n_state, n_action, scale=(action_high - action_low) / 2, bias=(action_low + action_high) / 2).to(device).double()
    
    policy_optim = Adam(policy.parameters(), lr=args.wbc_lr, weight_decay=args.weight_decay)

    optimizer = policy_optim 
    coeff_lst = new_R
    WBC_step = 0
    
    quant = torch.quantile(coeff_lst, torch.tensor([0.5, 0.9, 0.99, 0.999]).double().to(device), interpolation='midpoint')
    quant_50 = quant[0]
    quant_90 = quant[1]
    quant_99 = quant[2]
    quant_999 = quant[3]
    
    wandb.log({"coeff_top_0.1percent_mean": quant_999, "coeff_top_1percent_mean": quant_99, "coeff_top_10percent_mean": quant_90, "coeff_top_50percent_mean": quant_50})
    
    train_loader = RepeatedDataset([states_TA, actions_TA, terminals_TA, next_states_TA, coeff_lst], batch_size=args.batch_size)
    
    
    policy.train()
    
    N = args.N 
    for _ in tqdm(range(N)):
        # for batch_idx, sample_batched in enumerate(train_loader): 
        
        if  _ % args.eval_interval == args.eval_interval - 1:
            evaluation(policy, env, eval_use_argmax=(args.eval_deter == 1), mean_state=TA_mean, std_state=TA_std, is_kitchen=(args.env_name == "kitchen"), is_kitchen_goal=(args.env_name == "kitchen" and args.skip_suffix_TS.find("goal") != -1), goal_name=args.skip_suffix_TS)  
        
        debug_actions, debug_coeffs = [], []
        # print("size:", states_TA.shape, len(train_loader))
        for __ in tqdm(range(len(train_loader))):

            state, action, terminal, next_state, coeff = train_loader.getitem()
            # print(state.shape, action.shape, terminal.shape, next_state.shape, coeff.shape)
            log_prob, entropy, var = policy.logprob(state, action)
            loss = -(log_prob.view(-1, 1) * coeff.view(-1, 1)).mean()
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 1)
            g = 0
            for param in policy.parameters():
                g += torch.norm(param.grad, 2)
            if WBC_step % args.log_interval == 0:
                wandb.log({"WBC gradient norm": g}, commit=False) 
                wandb.log({"entropy_train":entropy.mean(), "logvar_mean": var.mean(), "weighted_entropy_train": (entropy.view(-1) * coeff.view(-1)).mean(), "weighted_logvar_train": (var.mean(dim=-1) * coeff.view(-1)).mean()}, commit=False)
                wandb.log({"WBC train loss": loss, 'WBC_steps': WBC_step})    
            WBC_step += 1
            if WBC_step >= 1050000: exit(0)
            optimizer.step()  

if __name__ == "__main__":
    args = get_args()
    
    seed = args.seed
    
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # when using multiple GPUs torch.cuda.manual_seed(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
    runtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if len(get_git_diff()) > 0:
        git_commit(runtime) 
    print("please input description:")
    
    suffix = args.skip_suffix_TS + args.skip_suffix_TA
    
    if args.auto == 0: a = input()
    else: 
        if args.mode == "expdecay": a = suffix + "-" + "PU" + str(args.PU) + "-nomax" + str(args.no_max) + "-safeneg" + str(args.PU1_alpha) + "-unbias" + str(args.PU2_alpha) + "-" + str(args.coeff_scale) + "-" + str(args.gamma) + "BS" + str(args.batch_size) +  "-auto" + "-EMA" + str(args.EMA) + "pretrain" + str(args.R_N_pretrain) + "RN" + str(args.R_N) # "-sensitivity-analysis"
        elif args.mode == "linear_decay": a = suffix + "-" + "PU" + str(args.PU) + "-nomax" + str(args.no_max) + "-safeneg" + str(args.PU1_alpha) + "-unbias" + str(args.PU2_alpha) + "-" + str(args.coeff_scale) + "-" + str(args.gamma) + "BS" + str(args.batch_size) +  "-auto" + "-EMA" + str(args.EMA) + "pretrain" + str(args.R_N_pretrain) + "RN" + str(args.R_N) + ("-bugnotfixed" if args.env_name == "kitchen" else "") + "-orilablation-R"
        elif args.mode == "uniform": 
            a = "plainBC-"+ "BS" + str(args.batch_size) + "-auto-" + suffix
        else: 
            print("Error!")
            exit(0)
    wandb.login(key="XXXXXXX") 
    wandb.init(entity="XXXXXXX",project="XXXXXXX", name=str(runtime)+"_"+str(args.seed)+"_"+args.env_name+"_main_"+a)
    
    TA_dataset, TS_dataset = torch.load("data/"+args.env_name+"/TA-read-again-unnormalized"+args.skip_suffix_TA+".pt"), torch.load("data/"+args.env_name+"/TS-read-again-unnormalized"+args.skip_suffix_TS+".pt") 
    train(TA_dataset, TS_dataset, args)
