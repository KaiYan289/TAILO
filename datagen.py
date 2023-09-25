import d4rl
import gym
import numpy as np
from dataset import get_dataset
import torch
from tqdm import tqdm
import h5py
from envs.kitchen_envs import *
env_names = ['halfcheetah', 'hopper', 'walker2d', 'ant']
np.random.seed(888)

def get_data_pre(env, num_traj=1e100, dataset=None, is_TS=False, missing_num=0): # SMODICE format
        
        initial_obs_, obs_, next_obs_, action_, reward_, done_, step_ = [], [], [], [], [], [], []
        
        if dataset is None:
            dataset = env.get_dataset() 
        N = dataset['rewards'].shape[0]

        use_timeouts = ('timeouts' in dataset)
        traj_count = 0
        episode_step = 0
        for i in range(N-1):
            # only use this condition when num_traj < 2000
            if traj_count == num_traj:
                break
            obs = dataset['observations'][i].astype(np.float32)
            new_obs = dataset['observations'][i+1].astype(np.float32)
            action = dataset['actions'][i].astype(np.float32)
            reward = dataset['rewards'][i].astype(np.float32)
            done_bool = bool(dataset['terminals'][i])
            # if not expert_data: print(dataset['terminals'][i])
            is_final_timestep = dataset['timeouts'][i] if use_timeouts else (episode_step == env._max_episode_steps - 1)
            if is_final_timestep and not is_TS:
                # Skip this transition and don't apply terminals on the last step of an episode
                traj_count += 1
                episode_step = 0
                continue
            if missing_num == 0 or i % missing_num != missing_num // 2:
                # if traj_count > 0 or not expert_data:     
                obs_.append(obs)
                next_obs_.append(new_obs)
                action_.append(action)
                reward_.append(reward)
                done_.append(done_bool) 
                step_.append(episode_step)
            episode_step += 1

            if done_bool or is_final_timestep:
                traj_count += 1
                episode_step = 0
        
        dataset = {
            'observations': np.array(obs_, dtype=np.float32),
            'actions': np.array(action_, dtype=np.float32),
            'next_observations': np.array(next_obs_, dtype=np.float32),
            'rewards': np.array(reward_, dtype=np.float32),
            'terminals': np.array(done_, dtype=np.float32),
            'steps': np.array(step_, dtype=np.float32)
        }
        
        return dataset


def get_data(env, lim=1e100, missing_num=0, is_TS=False, dataset=None):
    data = []
    tot_traj = 0
    
    if dataset is None: dataset = get_data_pre(env, lim, missing_num=missing_num, is_TS=is_TS)

    use_timeouts = ('timeouts' in dataset)
    
    FLAG = int(dataset is None or 'next_observations' in dataset) 
    
    for i in range(dataset['actions'].shape[0] - (1 - FLAG)):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['next_observations'][i].astype(np.float32) if FLAG else dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        episode_step = dataset['steps'][i]
        if not is_TS: data.append({"state": obs, "action": action, "next_state": new_obs, "terminal": done_bool, "step": episode_step, 'reward': reward})
        else: data.append({"state": obs, "action": action, "next_state": new_obs, "terminal": done_bool, "step": episode_step}) 
        is_final_timestep = dataset['timeouts'][i] if use_timeouts else (episode_step == env._max_episode_steps - 1)
        if is_final_timestep:
            tot_traj += 1
            if tot_traj >= lim: break
    return data
    


env_names = ['hopper', 'halfcheetah', 'walker2d', 'ant']

for env_name in env_names: 
    
    data = get_data(gym.make(env_name + "-medium-v2")) + get_data(gym.make(env_name + "-expert-v2"), 200 if env_name != 'walker2d' else 100)
    
    print(env_name)
        
    print("totlen:", len(data))
    
    torch.save(data, "data/"+env_name+"/TA-read-again-unnormalizedrecoilME.pt")
    
    data = get_data(gym.make(env_name + "-random-v2")) + get_data(gym.make(env_name + "-expert-v2"), 30)
    
    print("totlen:", len(data))
    
    torch.save(data, "data/"+env_name+"/TA-read-again-unnormalizedrecoilRFE.pt")


##################################################################### NORMAL #######################################################

env_names = ['hopper', 'halfcheetah', 'walker2d', 'ant'] 

for env_name in env_names: 
    
    data = get_data(gym.make(env_name + "-expert-v2"), 200 if env_name != 'walker2d' else 100) + get_data(gym.make(env_name + "-random-v2"))
    
    print(env_name)
        
    print("totlen:", len(data))
    
    torch.save(data, "data/"+env_name+"/TA-read-again-unnormalized.pt")
    
    data = get_data(gym.make(env_name + "-expert-v2"), 40) + get_data(gym.make(env_name + "-random-v2"))
    torch.save(data, "data/"+env_name+"/TA-read-again-unnormalizedexpert40.pt")
    
    for x in [2, 3, 5, 10, 20]:
        data = get_data(gym.make(env_name + "-expert-v2"), 200 if env_name != 'walker2d' else 100, missing_num=x) + get_data(gym.make(env_name + "-random-v2"), missing_num=x)
        torch.save(data, "data/"+env_name+"/TA-read-again-unnormalizedmissing"+str(x)+".pt")
    
        
    data = get_data(gym.make(env_name + "-expert-v2"), 1, is_TS=True)

    torch.save(data, "data/"+env_name+"/TS-read-again-unnormalized.pt")

    x, y = [1, 10, 50, 90, 100], [100, 90, 50, 10, 1] 

    for i in range(len(x)):
       xx, yy = x[i], y[i] 
       d = data[:xx] + data[-yy:]
       for _ in range(len(d)):
           d[_]["step"] = _
       torch.save(d, "data/"+env_name+"/TS-read-again-unnormalizedheadtail"+str(xx/1000)+"_"+str(yy/1000)+".pt")
    
#################################################################### KITCHEN #########################################################


env = gym.make("kitchen-mixed-v0")

set = env.get_dataset()

data = get_data(env)

torch.save(data, "data/kitchen/TA-read-again-unnormalized.pt")

env = gym.make("kitchen-complete-v0") # kitchen-normal

data = get_data(env, 1)          


"""
# We generate our data from SMODICE with seed 0 using the following code:
# But due to randomness in SMODICE's code, the data generated here might be different. We provide our generated data directly in the repo.
########################

e = []
for i in range(expert_obs.shape[0] - 1):
    e.append({"state": expert_obs[i], "next_state": expert_obs[i+1], 'action': np.zeros(1), 'terminal': np.ones(1), 'step': np.zeros(1)}) 
print(expert_obs.shape) 
torch.save(e, ".../data/kitchen/TS-read-again-unnormalizedgoal-"+config['dataset']+".pt")
exit(0) 

########################

torch.save(data, "data/kitchen/TS-read-again-unnormalized.pt")  

env = KitchenMicrowaveV0() 

expert_obs = env.get_example(set, num_expert_obs=500)
expert_traj = {'observations': expert_obs}

e = []
for i in range(expert_obs.shape[0] - 1):
    e.append({"state": expert_obs[i], "next_state": expert_obs[i+1], 'action': np.zeros(1), 'terminal': np.ones(1), 'step': np.zeros(1)}) 
torch.save(e, "data/kitchen/TS-read-again-unnormalizedgoal-microwave.pt")    



env = KitchenKettleV0()

expert_obs = env.get_example(set, num_expert_obs=500)
expert_traj = {'observations': expert_obs}      

e = []
for i in range(expert_obs.shape[0] - 1):
    e.append({"state": expert_obs[i], "next_state": expert_obs[i+1], 'action': np.zeros(1), 'terminal': np.ones(1), 'step': np.zeros(1)}) 
print(expert_obs.shape) 
torch.save(e, "data/kitchen/TS-read-again-unnormalizedgoal-kettle.pt")                   
"""

################################################################### ANTMAZE (task-agnostic only) ##########################################################

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def get_dataset(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    return data_dict


env = gym.make("antmaze-umaze-v2")

dataset1 = get_data_pre(env, 3000)

dataset2 = get_dataset("envs/demos/antmaze-umaze-v2-randomstart-noiserandomaction.hdf5") # run envs/generate_anatmaze_random.py first!



print(dataset1['observations'].shape, dataset2['observations'].shape)
data = {}
for key in dataset1.keys():
     data[key] = np.concatenate([dataset1[key], dataset2[key]], axis=0)


data = data1 + data2
print(len(data1), len(data2))
torch.save(data, "data/antmaze/TA-read-again-unnormalized.pt")

# See SMODICE for details of generation of TS-mismatch and TS-goal. We provide our generated TS data directly in the repo. There is some randomness in generation (we use seed=0 in SMODICE) of TS-goal.
