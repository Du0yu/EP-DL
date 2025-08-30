import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

# 权重初始化函数
def init_weights(m):
    # 只对 Linear 层进行初始化
    if isinstance(m, nn.Linear):
        # 使用 He 初始化（适用于 ReLU 激活函数）
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

        # 偏置初始化为零
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args

        # 使用 Hyper Networks 生成网络参数
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(
                nn.Linear(args.state_shape, args.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim)
            )
            self.hyper_w2 = nn.Sequential(
                nn.Linear(args.state_shape, args.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim)
            )
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(args.state_shape, args.qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.qmix_hidden_dim, 1)
        )

        self.apply(init_weights)

    def forward(self, q_values, states):
        # q_values 的形状是 (batch_size, n_agents)，表示每个智能体的 Q 值
        # states 的形状是 (batch_size, state_shape)，表示每个样本的状态

        batch_size = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)
        states = states.view(batch_size, self.args.state_shape)  # (batch_size, state_shape)

        # 使用 Hyper Networks 生成 w1 和 b1
        w1 = torch.abs(self.hyper_w1(states))  # (batch_size, n_agents * qmix_hidden_dim)
        b1 = self.hyper_b1(states)  # (batch_size, qmix_hidden_dim)

        # 重塑 w1 和 b1
        w1 = w1.view(batch_size, self.args.n_agents,
                     self.args.qmix_hidden_dim)  # (batch_size, n_agents, qmix_hidden_dim)
        b1 = b1.view(batch_size, 1, self.args.qmix_hidden_dim)  # (batch_size, 1, qmix_hidden_dim)
        # 计算混合网络的隐藏状态
        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # (batch_size, 1, qmix_hidden_dim)

        # 使用 Hyper Networks 生成 w2 和 b2
        w2 = torch.abs(self.hyper_w2(states))  # (batch_size, qmix_hidden_dim)
        b2 = self.hyper_b2(states)  # (batch_size, 1)

        # 重塑 w2 和 b2
        w2 = w2.view(batch_size, self.args.qmix_hidden_dim, 1)  # (batch_size, qmix_hidden_dim, 1)
        b2 = b2.view(batch_size, 1, 1)  # (batch_size, 1, 1)

        # 计算全局 Q 值
        q_total = torch.bmm(hidden, w2) + b2  # (batch_size, 1, 1)
        q_total = q_total.view(batch_size, 1)  # (batch_size, 1)

        return q_total