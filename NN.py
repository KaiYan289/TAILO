import torch
import torch.nn as nn
import math
import torch.nn.functional as F
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)    

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        middle_size = 256
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.Tanh(), 
            nn.Linear(middle_size, middle_size),
            nn.Tanh(),
            nn.Linear(middle_size, 1))
            
    def forward(self, s):
        return self.net(s)
    
    def embedding(self, s):
        return self.net[:-1](s)
    
    def predict_reward(self, state):
        with torch.no_grad():
            self.eval()
            d = self(state)
            s = torch.sigmoid(d)
            # log(d^E/d^O)
            # reward  = - (1/s-1).log()
            reward = s.log() - (1 - s).log()
            return reward 

class Normal_Predictor(nn.Module):
    def __init__(self, input_size, debug_scaling=1, use_bn=False):
        super().__init__()
        middle_size = 256
        self.debug_scaling = debug_scaling
        if not use_bn:
            self.net = nn.Sequential(
                nn.Linear(input_size, middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.ReLU(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_size, middle_size),
                nn.BatchNorm1d(middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.BatchNorm1d(middle_size),
                nn.ReLU(),
            )
        self.net_a = nn.Linear(middle_size, 1)
        self.net_b = nn.Linear(middle_size, 1)
        self.net_c = nn.Linear(middle_size, 1)
        # self.net_a, self.net_b, self.net_c = nn.Linear(input_size, 1, bias=False),  nn.Linear(input_size, 1, bias=False), nn.Linear(input_size, 1, bias=False)
        self.net_d = nn.parameter.Parameter(torch.rand(1, requires_grad=True), requires_grad=True)
        
    def forward(self, w, tag):
        if w is not None: x = w * self.debug_scaling
        else: x = w
        if tag == 3: return self.net_d
        v = self.net(x)
        if tag == -1: return self.net_a(v), self.net_b(v), self.net_c(v), self.net_d
        elif tag == 0: return self.net_a(v)
        elif tag == 1: return self.net_b(v)
        elif tag == 2: return self.net_c(v)

class ActorDiscrete(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        middle_size = 256
        self.input_size, self.output_size = n, m
        self.net = nn.Sequential(
                nn.Linear(self.input_size, middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, self.output_size),
                nn.Softmax()
            )
        # self.apply(weights_init_)
    
    def logprob(self, state, x):
        # p, mean, logvar = self.prob(state)
        # return p.log_prob(x)
        if len(state.shape) < 2: state = state.unsqueeze(0) # this is faster
        output = self.net(state) # output: BS * action size
        res = torch.gather(output, 1, x.long().reshape(-1, 1))
        return torch.log(res), -res * torch.log(res + 1e-10), torch.zeros_like(res).double().to('cuda:0') # entropy and logvar
            
    def sample(self, state, size=None):
        # input batchsize * (state + action); return batchsize * size * state
        output = self.net(state)
        if size is None: return torch.multinomial(output, 1, replacement=True) # batch size * 1
        else: return torch.multinomial(output, size, replacement=True).T # batch size * number of samples
            
    def deterministic_action(self, state, size=None):
        if len(state.shape) < 2: state = state.unsqueeze(0) # this is faster
        output = self.net(state)
        return output.argmax(dim=1)
        
def atanh(z):
    return 0.5 * (torch.log(1 + z) - torch.log(1 - z))  

class Actor_MultiDiscrete(nn.Module): # still receive continuous action and give action in continuous space.
    def __init__(self, n, dim_len, action_low, action_high, partition=200, device='cuda:0'): # 100 action per dimension
        super().__init__()
        middle_size = 256
        self.input_size = n
        self.device = device
        self.action_dim = dim_len
        self.partition = partition
        self.action_low, self.action_high = torch.from_numpy(action_low).to(self.device).double(), torch.from_numpy(action_high).to(self.device).double()
        self.net = nn.Sequential(
                nn.Linear(self.input_size, middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.ReLU(),
            )
        self.output = nn.ModuleList([nn.Sequential(nn.Linear(middle_size, partition), nn.Softmax()) for _ in range(self.action_dim)])
        
    def logprob(self, state, x):
        discrete_x = torch.zeros_like(x).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        for i in range(self.action_dim):
            discrete_x[:, i] = torch.div((x[:, i] - self.action_low[i]) * self.partition, (self.action_high[i] - self.action_low[i]), rounding_mode='floor')
        discrete_x = torch.clamp(discrete_x, max=self.partition - 1)
        discrete_x = discrete_x.long()
        v = self.net(state)
        tot_p = torch.zeros(state.shape[0], 1).to(self.device)
        for i in range(self.action_dim):
            p = self.output[i](v)
            tot_p += torch.log(torch.gather(p, 1, discrete_x[:, i].reshape(-1, 1)))
        return tot_p, torch.zeros_like(tot_p).double().to(self.device), torch.zeros_like(tot_p).double().to(self.device)
    
    def sample(self, state, size=None):
        assert len(state.shape) == 1, "Error: more than one state!"
        v = self.net(state)
        if size is None:
           sample = torch.zeros(self.action_dim).to(self.device).double()
           for i in range(self.action_dim):
               p = self.output[i](v)
               sample[i] = self.action_low[i].double() + (self.action_high[i] - self.action_low[i]).double() * torch.multinomial(p, 1, replacement=True) / self.partition
               #print("p:", p)
           #print("sample:", sample)
           return sample
        else: raise NotImplementedError("Error: more than one sample!")
         
    def deterministic_action(self, state, size=None):
        assert len(state.shape) == 1, "Error: more than one state!"
        output = self.net(state)
        if size is None:
           sample = torch.zeros(self.action_dim).double().to(self.device)
           for i in range(self.action_dim):
               p = self.output[i](v)
               sample[i] = self.action_low[i].double() + (self.action_high[i] - self.action_low[i]).double() * p.argmax(dim=1)
           return sample
        else: raise NotImplementedError("Error: more than one sample!")

class Actor_Twinhead(nn.Module):
    def __init__(self, n, m, clip_var):
        super().__init__()
        middle_size = 256
        self.clip_var = clip_var
        self.input_size, self.output_size = n, m
        self.net = nn.Sequential(
                nn.Linear(self.input_size, middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.ReLU(),
            )
        self.mean, self.logvar = nn.Linear(middle_size, self.output_size), nn.Linear(middle_size, self.output_size)
        # self.apply(weights_init_)
        
    def prob(self, state):
        if len(state.shape) < 2: state = state.unsqueeze(0)
        output = self.net(state)
        mean, logvar = self.mean(output), self.logvar(output) # output[:, :self.output_size], output[:, self.output_size:]
        if self.clip_var == 1: logvar = 7 * F.tanh(logvar) - 3 # (var is -5 to 2, middle point -1.5)
        p = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=torch.cat([torch.diag(torch.exp(logvar[i])).unsqueeze(0) for i in range(state.shape[0])], dim=0))
        return p, mean, logvar
    
    def logprob(self, state, x):
        # p, mean, logvar = self.prob(state)
        # return p.log_prob(x)
        if len(state.shape) < 2: state = state.unsqueeze(0) # this is faster
        output = self.net(state)
        mean, logvar = self.mean(output), self.logvar(output) # output[:, :self.output_size], output[:, self.output_size:]
        if self.clip_var == 1: logvar = 7 * F.tanh(logvar) - 3 # (var is -5 to 2, middle point -1.5)
        v = -0.5 * (math.log(2 * math.pi) * mean.shape[-1] + logvar.sum(dim=-1) + ((x - mean) ** 2 / torch.exp(logvar)).sum(dim=-1))
        # print("mean:", mean, "logvar:", logvar, "x:", x)
        #p, mean, logvar = self.prob(state)
        #return p.log_prob(x)
        # print(p.log_prob(x), v)
        # assert (p.log_prob(x) - v).abs().max() < 1e-4, "Error!"
        """
        p = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=torch.cat([torch.diag(torch.exp(logvar[i])).unsqueeze(0) for i in range(state.shape[0])], dim=0))
        print("state:", state.shape[1], "output:", self.output_size)
        print("entropy:", p.entropy() - (self.output_size / 2 * (1 + math.log(2 * math.pi)) + 0.5 * torch.sum(logvar, dim=1)))
        exit(0)
        """
        return v, self.output_size / 2 * (1 + math.log(2 * math.pi)) + 0.5 * torch.sum(logvar, dim=1), logvar
            
    def sample(self, state, size=None):
        # input batchsize * (state + action); return batchsize * size * state
        p, mean, logvar = self.prob(state)
        if self.clip_var == 1: logvar = 7 * F.tanh(logvar) - 3 # (var is -5 to 2, middle point -1.5)
        if size is None: return p.sample()
        else: 
            samples = []
            for i in range(size):
                samples.append(p.sample().unsqueeze(1))
            return torch.cat(samples, dim=1)
            
    def deterministic_action(self, state, size=None):
        if len(state.shape) < 2: state = state.unsqueeze(0) # this is faster
        output = self.net(state)
        mean, logvar = self.mean(output), self.logvar(output)
        if state.shape[0] == 1: mean = mean.squeeze() 
        return mean

class Actor(nn.Module):
    def __init__(self, n, m, clip_var):
        super().__init__()
        middle_size = 256
        self.clip_var = clip_var
        self.input_size, self.output_size = n, m
        self.net = nn.Sequential(
                nn.Linear(self.input_size, middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, 2 * self.output_size)
            )
        # self.apply(weights_init_)
        
    def prob(self, state):
        if len(state.shape) < 2: state = state.unsqueeze(0)
        output = self.net(state)
        mean, logvar = output[:, :self.output_size], output[:, self.output_size:]
        if self.clip_var == 1: logvar = 7 * F.tanh(logvar) - 3 # (var is -5 to 2, middle point -1.5)
        p = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=torch.cat([torch.diag(torch.exp(logvar[i])).unsqueeze(0) for i in range(state.shape[0])], dim=0))
        return p, mean, logvar
    
    def logprob(self, state, x):
        # p, mean, logvar = self.prob(state)
        # return p.log_prob(x)
        if len(state.shape) < 2: state = state.unsqueeze(0) # this is faster
        output = self.net(state)
        mean, logvar = output[:, :self.output_size], output[:, self.output_size:]
        if self.clip_var == 1: logvar = 7 * F.tanh(logvar) - 3 # (var is -5 to 2, middle point -1.5)
        v = -0.5 * (math.log(2 * math.pi) * mean.shape[-1] + logvar.sum(dim=-1) + ((x - mean) ** 2 / torch.exp(logvar)).sum(dim=-1))
        # print("mean:", mean, "logvar:", logvar, "x:", x)
        #p, mean, logvar = self.prob(state)
        #return p.log_prob(x)
        # print(p.log_prob(x), v)
        # assert (p.log_prob(x) - v).abs().max() < 1e-4, "Error!"
        """
        p = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=torch.cat([torch.diag(torch.exp(logvar[i])).unsqueeze(0) for i in range(state.shape[0])], dim=0))
        print("state:", state.shape[1], "output:", self.output_size)
        print("entropy:", p.entropy() - (self.output_size / 2 * (1 + math.log(2 * math.pi)) + 0.5 * torch.sum(logvar, dim=1)))
        exit(0)
        """
        return v, self.output_size / 2 * (1 + math.log(2 * math.pi)) + 0.5 * torch.sum(logvar, dim=1), logvar
            
    def sample(self, state, size=None):
        # input batchsize * (state + action); return batchsize * size * state
        p, mean, logvar = self.prob(state)
        if self.clip_var == 1: logvar = 7 * F.tanh(logvar) - 3 # (var is -5 to 2, middle point -1.5)
        if size is None: return p.sample()
        else: 
            samples = []
            for i in range(size):
                samples.append(p.sample().unsqueeze(1))
            return torch.cat(samples, dim=1)
            
    def deterministic_action(self, state, size=None):
        if len(state.shape) < 2: state = state.unsqueeze(0) # this is faster
        output = self.net(state)
        mean, logvar = output[:, :self.output_size], output[:, self.output_size:]
        if state.shape[0] == 1: mean = mean.squeeze() 
        return mean

    
class ActorTanh(nn.Module):
    def __init__(self, n, m, scale, bias):
        super().__init__()
        middle_size = 256
        self.scale, self.bias = torch.from_numpy(scale).to('cuda:0'), torch.from_numpy(bias).to('cuda:0')
        self.input_size, self.output_size = n, m
        self.net = nn.Sequential(
                nn.Linear(self.input_size, middle_size),
                nn.ReLU(),
                nn.Linear(middle_size, middle_size),
                nn.ReLU()
            )
        self.mean, self.logvar = nn.Linear(middle_size, self.output_size), nn.Linear(middle_size, self.output_size)
        # self.apply(weights_init_)

    def forward(self, state):
        output = self.net(state)
        mean, logvar = self.mean(output), 7 * F.tanh(self.logvar(output)) - 3
        return mean, logvar

    def logprob(self, state, action):
        # p, mean, logvar = self.prob(state)
        # return p.log_prob(x)
        
        # pretanh distribution
        if len(state.shape) < 2: state = state.unsqueeze(0) # this is faster
        mean, logvar = self(state)
        normalized_action = (action - self.bias) / self.scale
        pretanh_action = atanh(torch.clamp(normalized_action, -1 + 1e-6, 1-1e-6))
        #print("normalized_action:", normalized_action, "pretanh_action:", pretanh_action)
        v = -0.5 * (math.log(2 * math.pi) * mean.shape[-1] + logvar.sum(dim=-1) + ((pretanh_action - mean) ** 2 / torch.exp(logvar)).sum(dim=-1))
        # the derivative of atanh(x) is 1/(1-x^2)
        v -= torch.log(1 - normalized_action ** 2 + 1e-10).sum(dim=-1)
        
        # print("mean:", mean, "logvar:", logvar, "x:", x)
        #p, mean, logvar = self.prob(state)
        #return p.log_prob(x)
        # print(p.log_prob(x), v)
        # assert (p.log_prob(x) - v).abs().max() < 1e-4, "Error!"
        return v, -v, logvar
            
    def sample(self, state, size=None):
        # input batchsize * (state + action); return batchsize * size * state
        if len(state.shape) < 2: state = state.unsqueeze(0) # this is faster
        mean, logvar = self(state)
        p = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=torch.cat([torch.diag(torch.exp(logvar[i])).unsqueeze(0) for i in range(state.shape[0])], dim=0))
        if size is None: return torch.tanh(p.sample()) * self.scale + self.bias
        else: 
            samples = []
            for i in range(size):
                samples.append(p.sample().unsqueeze(1))
            return torch.tanh(torch.cat(samples, dim=1)) * self.scale + self.bias
    
    def deterministic_action(self, state, size=None):
        if len(state.shape) < 2: state = state.unsqueeze(0) # this is faster
        mean, logvar = self(state)
        if state.shape[0] == 1: mean = mean.squeeze() 
        return torch.tanh(mean) * self.scale + self.bias
