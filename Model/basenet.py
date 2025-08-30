import torch.nn as nn
import torch.nn.functional as f
import torch


def init_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier 初始化（适用于 ReLU 激活函数）
        torch.nn.init.xavier_uniform_(m.weight)
        # 偏置初始化为零
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class MLP(nn.Module):
    #RNN多智能体复用
    def __init__(self, input_shape, args):
        super(MLP, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.n_actions)

        self.apply(init_weights)

    def forward(self, obs):
        x = f.relu(self.fc1(obs))
        x = f.relu(self.fc2(x))
        return self.fc3(x)



class Actor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.net = nn.Sequential(
            nn.Linear(self.args.obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.args.n_actions)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.net = nn.Sequential(
            nn.Linear(self.args.state_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        value = self.net(x)
        return value