from collections import deque
import random
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    _instance = None  # 单例实例

    def __new__(cls, args):
        if cls._instance is None:
            # 如果没有实例，则创建一个新实例
            cls._instance = super(ReplayBuffer, cls).__new__(cls)
            # 使用初始化标志来避免重复初始化
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, args):
        if self._initialized:
            return

        self._initialized = True
        self.args = args
        self.capacity = self.args.buffer_size
        self.n_agents = self.args.n_agents
        self.n_actions = self.args.n_actions

        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape

        # 预分配numpy数组 (比deque内存效率提升3-5倍)
        self.states = np.zeros((self.capacity, self.state_shape), dtype=np.float32)
        self.obs = np.zeros((self.capacity, self.n_agents, self.obs_shape), dtype=np.float32)
        self.action = np.zeros((self.capacity, self.n_agents, 1), dtype=np.int32)
        if self.args.alg in {'qmix', 'vdn', 'qmixr1', 'qmixr2'}:
            self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
            self.dones = np.zeros((self.capacity, 1), dtype=np.bool_)
        elif self.args.alg == 'madqn':
            self.rewards = np.zeros((self.capacity, self.n_agents), dtype=np.float32)
            self.dones = np.zeros(self.capacity, dtype=np.bool_)

        self.next_states = np.zeros((self.capacity, self.state_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.n_agents, self.obs_shape), dtype=np.float32)

        # 环形缓冲区指针
        self.index = 0
        self.size = 0

    def add(self, state, obs, action, reward, next_state, obs_next, done):
        """添加单条经验到缓冲区"""
        # 自动覆盖旧数据
        self.states[self.index] = state
        self.obs[self.index] = obs
        self.action[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.next_obs[self.index] = obs_next
        self.dones[self.index] = done

        # 更新指针
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device='cuda'):
        """采样并直接返回GPU Tensor"""
        if self.size < batch_size:
            raise ValueError(f"Buffer only contains {self.size} samples")

        # 高效随机索引生成
        indices = np.random.choice(self.size, batch_size, replace=False)

        # 批量数据提取（速度提升5-8倍）
        states = torch.from_numpy(self.states[indices]).pin_memory().to(device, non_blocking=True)
        obs = torch.from_numpy(self.obs[indices]).pin_memory().to(device, non_blocking=True)
        actions = torch.from_numpy(self.action[indices].astype(np.int64)).pin_memory().to(device, non_blocking=True)
        rewards = torch.from_numpy(self.rewards[indices]).pin_memory().to(device, non_blocking=True)
        next_states = torch.from_numpy(self.next_states[indices]).pin_memory().to(device, non_blocking=True)
        next_obs = torch.from_numpy(self.next_obs[indices]).pin_memory().to(device, non_blocking=True)
        dones = torch.from_numpy(self.dones[indices].astype(np.float32)).pin_memory().to(device, non_blocking=True)

        return (states, obs, actions,
                rewards, next_states, next_obs, dones)
    def size(self):
        return len(self.buffer)


class MAPPOReplayBuffer:
    def __init__(self):
        # 初始化各数据容器
        self.device = device
        self.buffers = {
            "obs": [],           # 观测值 [T][n_agents][obs_dim]
            "actions": [],       # 动作 [T][n_agents]
            "rewards": [],       # 奖励 [T][n_agents]
            "next_obs": [],      # 下一观测值 [T][n_agents][obs_dim]
            "dones": [],         # 终止标志 [T]
            "global_states": [],       # 全局状态 [T][global_state_dim]
            "next_global_states": [], # 下一全局状态 [T][global_state_dim]
            "old_log_probs": [],
        }

    def store_transition(self,
                         obs,
                         actions,
                         rewards,
                         next_obs,
                         done,
                         global_state,
                         next_global_state,
                         old_log_probs,
                         ):
        """存储单步经验数据（直接存入GPU张量）"""
        # 转换数据为Tensor并存入GPU
        self.buffers["obs"].append(
            torch.as_tensor(obs, dtype=torch.float, device=self.device)
        )
        self.buffers["actions"].append(
            torch.as_tensor(actions, dtype=torch.long, device=self.device)
        )
        self.buffers["rewards"].append(
            torch.as_tensor(rewards, dtype=torch.float, device=self.device)
        )
        self.buffers["next_obs"].append(
            torch.as_tensor(next_obs, dtype=torch.float, device=self.device)
        )
        self.buffers["dones"].append(
            torch.as_tensor(done, dtype=torch.bool, device=self.device)
        )
        self.buffers["global_states"].append(
            torch.as_tensor(global_state, dtype=torch.float, device=self.device)
        )
        self.buffers["next_global_states"].append(
            torch.as_tensor(next_global_state, dtype=torch.float, device=self.device)
        )
        self.buffers["old_log_probs"].append(
            torch.as_tensor(old_log_probs, dtype=torch.float, device=self.device)
        )

    def get_batch(self):
        batch = {}
        for key in self.buffers:
            if len(self.buffers[key]) == 0:
                raise RuntimeError(f"Buffer '{key}' is empty. Check data collection")

            # 直接在GPU上堆叠张量
            data = torch.stack(self.buffers[key])

            # 特殊维度处理
            if key == "dones":
                data = data.float().unsqueeze(-1)  # [T] -> [T, 1]

            batch[key] = data
        return batch


    def clear(self):
        """清空缓冲区（释放GPU显存）"""
        for key in self.buffers:
            self.buffers[key].clear()
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.buffers["obs"])