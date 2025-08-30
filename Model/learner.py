import os
from Model.VDNNet import VDNNet
from Model.QMixNet import QMixNet
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

from Model.basenet import MLP, Actor, Critic

from torch.utils.tensorboard import SummaryWriter



class Learner:
    def __init__(self, args):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.args = args #args
        self.timestep = 0  # 目前训练步
        self.temp_start = self.args.temp_start
        self.min_temp = self.args.min_temp
        self.temp_anneal_steps = self.args.temp_anneal_steps
        self.log_interval = self.args.log_interval

        # 模型保存目录
        self.model_dir = os.path.join(os.getcwd(), 'saved_models')
        os.makedirs(self.model_dir, exist_ok=True)

        # 生成带时间戳的日志目录名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式示例: 20231010_153045
        log_dir_name = f"logs_{args.alg}_{timestamp}"
        self.log_dir = os.path.join(self.model_dir, log_dir_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.n_actions = args.n_actions #一个智能体有多少个action 12
        self.n_agents = args.n_agents  #有多少个智能体 5
        self.state_shape = args.state_shape  #全局状态 25
        self.obs_shape = args.obs_shape #智能体观测状态 9
        input_shape = self.obs_shape


        if self.args.alg in {'qmix', 'qmixr2', 'qmixr1'}:

            # 神经网络
            self.q_networks = [MLP(input_shape,args).to(self.device) for _ in range(self.n_agents)]
            self.target_q_networks = [MLP(input_shape,args).to(self.device) for _ in range(self.n_agents)]

            # 使用 Hyper Networks 生成 Q 网络和混合网络的权重
            self.qmix_net = QMixNet(args).to(self.device)
            self.target_qmix_net = QMixNet(args).to(self.device)

            # 优化器
            params = [param for q_network in self.q_networks for param in q_network.parameters()]
            params += list(self.qmix_net.parameters())

            self.optimizer = torch.optim.RMSprop(params, lr=args.lr)

            # 加载最新模型
            if self.args.load_model:
                self.load_model()

            self.timestep = 0  # 目前训练步

            # 让target_net和eval_net的网络参数相同
            self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())
            for i in range(self.args.n_agents):
                self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())


        elif self.args.alg == 'vdn':
            # 神经网络
            self.q_networks = [MLP(input_shape, args).to(self.device) for _ in range(self.n_agents)]
            self.target_q_networks = [MLP(input_shape, args).to(self.device) for _ in range(self.n_agents)]

            # 使用 Hyper Networks 生成 Q 网络和混合网络的权重
            self.vdn_net = VDNNet().to(self.device)
            self.target_vdn_net = VDNNet().to(self.device)

            # 优化器
            params = [param for q_network in self.q_networks for param in q_network.parameters()]
            params += list(self.vdn_net.parameters())

            self.optimizer = torch.optim.RMSprop(params, lr=args.lr)

            if self.args.load_model:
                self.load_model()

            # 让target_net和eval_net的网络参数相同
            self.target_vdn_net.load_state_dict(self.vdn_net.state_dict())
            for i in range(self.args.n_agents):
                self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())

        elif self.args.alg == 'madqn':
            self.q_networks = [MLP(input_shape, args).to(self.device) for _ in range(self.n_agents)]
            self.target_q_networks = [MLP(input_shape, args).to(self.device) for _ in range(self.n_agents)]
          # 优化器
            self.optimizers = [torch.optim.RMSprop(q_network.parameters(), lr=args.lr) for q_network in self.q_networks]


            if self.args.load_model:
                self.load_model()


            for i in range(self.args.n_agents):
                self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())

        elif self.args.alg == "mappo":
            self.actors = [Actor(self.args).to(self.device) for _ in range(self.args.n_agents)]
            self.old_actors = [Actor(self.args).to(self.device) for _ in range(self.args.n_agents)]
            self.critics = [Critic(self.args).to(self.device) for _ in range(self.args.n_agents)]

            # 创建 actor 优化器
            self.actor_optimizers = [
                torch.optim.RMSprop(actor.parameters(), lr=self.args.actor_lr) for actor in self.actors
            ]

            # 创建 critic 优化器
            self.critic_optimizers = [
                torch.optim.RMSprop(critic.parameters(), lr=self.args.critic_lr) for critic in self.critics
            ]

    def get_current_temp(self):
        """计算当前温度"""
        frac = min(self.timestep / self.temp_anneal_steps, 1.0)
        return self.temp_start - frac * (self.temp_start - self.min_temp)

    def update(self, transition_dict):  # 更新target_net网络的参数
        timestep = self.timestep
        if self.args.alg == 'mappo':
            for old_actor, actor in zip(self.old_actors, self.actors):
                old_actor.load_state_dict(actor.state_dict())
            # 预处理全局数据
            global_states = transition_dict["global_states"]  # [T, state_dim]
            next_global_states = transition_dict["next_global_states"]  # [T, state_dim]
            dones = transition_dict["dones"].float().squeeze(-1)  # [T]
        else:
            (states, obs, actions,
             rewards, next_states, next_obs, terminated) = transition_dict

        if self.args.alg in {'qmix', 'qmixr2', 'qmixr1'}:

            q_values = [q_network(obs[:, i, :]) for i, q_network in
                        enumerate(self.q_networks)]  # 每个智能体的 Q 值 [batch_size, 1]
            q_values = torch.stack(q_values, dim=1).to(self.device)  # [batch_size, n_agents, 1]

            q_values = torch.gather(q_values, dim=2, index=actions).squeeze(2)


            # 计算目标 Q 值
            target_q_values = [target_q_network(next_obs[:, i, :]) for i, target_q_network in
                               enumerate(self.target_q_networks)]  # [batch_size, 1]
            target_q_values = torch.stack(target_q_values, dim=1).to(self.device)  # 堆叠成 [batch_size, n_agents, 1]

            target_q_values = target_q_values.max(dim=2)[0]



            # 使用混合网络计算当前的 Q 值
            q_value = self.qmix_net(q_values, states)  # [batch_size, 1]
            # 使用目标 Q 网络计算下一个状态的 Q 值

            next_q_values = self.target_qmix_net(target_q_values, next_states)  # [batch_size, 1]
            # 计算目标 Q 值：使用奖励和下一个状态的 Q 值
            target_q = rewards + (self.args.gamma * next_q_values * (1 - terminated))  # [batch_size, 1]



            loss = torch.mean(F.mse_loss(q_value, target_q))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        elif self.args.alg == 'vdn':
            q_values = [q_network(obs[:, i, :]) for i, q_network in
                        enumerate(self.q_networks)]  # 每个智能体的 Q 值 [batch_size, 1]
            q_values = torch.stack(q_values, dim=1).to(self.device)  # [batch_size, n_agents, 1]

            q_values = torch.gather(q_values, dim=2, index=actions).squeeze(2)

            # 计算目标 Q 值
            target_q_values = [target_q_network(next_obs[:, i, :]) for i, target_q_network in
                               enumerate(self.target_q_networks)]  # [batch_size, 1]
            target_q_values = torch.stack(target_q_values, dim=1).to(self.device)  # 堆叠成 [batch_size, n_agents, 1]

            target_q_values = target_q_values.max(dim=2)[0]

            # 使用混合网络计算当前的 Q 值
            q_value = self.vdn_net(q_values)  # [batch_size, 1]
            # 使用目标 Q 网络计算下一个状态的 Q 值
            next_q_values = self.target_vdn_net(target_q_values)  # [batch_size, 1]
            # 计算目标 Q 值：使用奖励和下一个状态的 Q 值
            target_q = rewards + (self.args.gamma * next_q_values * (1 - terminated))  # [batch_size, 1]

            loss = F.mse_loss(q_value, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        elif self.args.alg == 'madqn':
            agent_losses = []
            for i in range(self.args.n_agents):
                q_value = self.q_networks[i](obs[:, i, :]).gather(1, actions[:, i, :])

                with torch.no_grad():
                    target_q_value = self.target_q_networks[i](next_obs[:, i, :]).max(1)[0]
                    target_q = rewards[:, i].squeeze(-1) + self.args.gamma * target_q_value * (
                                1 - terminated.squeeze(-1))

                loss = F.mse_loss(q_value.squeeze(), target_q)
                agent_losses.append(loss)  # 直接存储tensor


                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()
            loss = torch.mean(torch.stack(agent_losses))

        elif self.args.alg == 'mappo':
            # 并行处理所有智能体（利用GPU并行性）
            for agent_id, (actor, critic, actor_opt, critic_opt) in (
                    enumerate(zip(self.actors, self.critics, self.actor_optimizers, self.critic_optimizers))):
                # 提取当前智能体数据
                # --- 数据提取（保持GPU张量）---

                obs = transition_dict['obs'][:, agent_id]  # [T, obs_dim]
                actions = transition_dict['actions'][:, agent_id]  # [T]
                rewards = transition_dict['rewards'][:, agent_id]  # [T]
                old_log_probs = transition_dict['old_log_probs'][:, agent_id]

                # 获取 Critic 估计的值函数
                with torch.no_grad():
                    values = critic(global_states).squeeze(-1)  # [T]，计算 V(s_t)
                    next_values = critic(next_global_states).squeeze(-1)

                    # 向量化GAE计算
                    deltas = rewards + self.args.gamma * next_values * (1 - dones) - values
                    # 反向计算GAE
                    advantages = []
                    advantage = 0
                    for t in reversed(range(len(deltas))):
                        advantage = deltas[t] + self.args.gamma * self.args.gae_lambda * (1 - dones[t]) * advantage
                        advantages.insert(0, advantage)
                    advantages = torch.tensor(advantages, device=values.device)

                    # 智能体级标准化
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                for _ in range(self.args.epoch):
                    action_logits = actor(obs)
                    new_dist = torch.distributions.Categorical(logits=action_logits)
                    new_log_probs = new_dist.log_prob(actions)

                    # 熵正则化
                    entropy = new_dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - old_log_probs)
                    # 损失计算

                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.args.clip_eps, 1 + self.args.clip_eps) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Critic损失
                    pred_values = self.critics[agent_id](global_states)
                    target_values = rewards + self.args.gamma * next_values * (1 - dones)
                    critic_loss = F.mse_loss(pred_values.squeeze(), target_values.detach())

                    # 总损失
                    loss = actor_loss + 0.5 * critic_loss - self.args.entropy_coef * entropy

                    actor_opt.zero_grad()
                    critic_opt.zero_grad()
                    loss.backward()

                    actor_opt.step()
                    critic_opt.step()

                    self._log_metrics(agent_id, actor_loss, critic_loss,
                                      timestep)


        # 定期记录详细指标
        if timestep % self.log_interval == 0:
            self.writer.add_scalar('Train/Loss', loss.item(), self.timestep)
            # 记录温度参数
            self.writer.add_scalar('Params/Temperature', self.get_current_temp(), self.timestep)


        if timestep % self.args.save_cycle == 0:
            self.save_model(self.timestep)

        # **目标网络更新**：每隔一定步数将主网络的权重更新到目标网络
        if timestep % self.args.target_update_cycle == 0:
            self.update_target_network()
        # 训练步数增加
        self.timestep += 1

    def select_action(self, obs, explore=True):
        """选择动作并返回旧策略的log概率"""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        n_agents = self.args.n_agents

        log_probs = []
        actions = []
        with torch.no_grad():
            for agent_id in range(n_agents):

                    logit = self.actors[agent_id](obs_tensor[agent_id])
                    dist = torch.distributions.Categorical(logits=logit)

                    if explore:
                        action = dist.sample()
                    else:
                        action = torch.argmax(logit)
                    log_probs.append(dist.log_prob(action).item())
                    actions.append(action.item())

        return np.array(actions), np.array(log_probs)


    def choose_action(self, obs, evaluate=False):  # take actions
        actions = []
        temp = self.get_current_temp()

        inputs = obs.copy()
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)

        for agent_num in range(self.n_agents):
            q_values = self.q_networks[agent_num](inputs_tensor[agent_num, :])
            if evaluate:
                action = torch.argmax(q_values)
            else:
                probs = torch.softmax(q_values / temp, dim = -1)
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()
            actions.append(action.item())
            # if evaluate:
            #     action = torch.argmax(q_values)
            # else:
            #     # 训练模式：ε-greedy策略（ε=0.1）
            #     if np.random.rand() < 0.1:
            #         # 随机探索：从所有动作中均匀采样
            #         action = torch.randint(low=0, high=q_values.size(0), size=(1,))
            #     else:
            #         # 利用：选择Q值最大的动作
            #         action = torch.argmax(q_values)
            # actions.append(action.item())


        return actions





    def update_target_network(self):   #更新网络参数
        if self.args.alg == 'vdn':
            for target_q_network, q_network in zip(self.target_q_networks, self.q_networks):
                target_q_network.load_state_dict(q_network.state_dict())

            # 同步混合网络的目标网络
            self.target_vdn_net.load_state_dict(self.vdn_net.state_dict())

        elif self.args.alg == 'qmix':
            for target_q_network, q_network in zip(self.target_q_networks, self.q_networks):
                target_q_network.load_state_dict(q_network.state_dict())

            # 同步混合网络的目标网络
            self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())

        elif self.args.alg == 'madqn':
            for target_q_network, q_network in zip(self.target_q_networks, self.q_networks):
                target_q_network.load_state_dict(q_network.state_dict())


    def save_model(self, train_step):   #保存模型参数
        try:
            # 计算保存周期编号
            cycle_num = train_step // self.args.save_cycle
            model_name = f"{self.args.alg}_model_step_{train_step}_cycle_{cycle_num}.pt"
            model_path = os.path.join(self.model_dir, model_name)
            os.makedirs(self.model_dir, exist_ok=True)

            # 基础检查点内容
            checkpoint = {
                'timestep': self.timestep,
                'model_step': train_step,
                'temp_config': {
                    'current_temp': self.get_current_temp(),
                    'temp_start': self.temp_start,
                    'min_temp': self.min_temp,
                    'temp_anneal_steps': self.temp_anneal_steps
                },
                'args': vars(self.args),
                'algorithm': self.args.alg
            }
            # 基础网络参数
            if self.args.alg in ['qmix', 'vdn', 'madqn', 'qmixr2', 'qmixr1']:
                checkpoint.update({
                    'q_network_states': [net.state_dict() for net in self.q_networks],
                    'target_q_net_states': [net.state_dict() for net in self.target_q_networks]
                })

            # MAPPO
            if self.args.alg == 'mappo':
                checkpoint.update({
                    'actors_state': [actor.state_dict() for actor in self.actors],
                    'critics_state': [critic.state_dict() for critic in self.critics],
                    'actor_optimizers_state': [opt.state_dict() for opt in self.actor_optimizers],
                    'critic_optimizers_state': [opt.state_dict() for opt in self.critic_optimizers]
                })

            # 优化器状态
            if self.args.alg == 'madqn':
                checkpoint['optimizers_state'] = [opt.state_dict() for opt in self.optimizers]
            elif self.args.alg in ['qmix', 'vdn', 'qmixr2', 'qmixr1']:
                checkpoint['optimizer_state'] = self.optimizer.state_dict()

            # 混合网络参数
            if self.args.alg in {'qmix', 'qmixr2', 'qmixr1'}:
                checkpoint.update({
                    'qmix_net_state': self.qmix_net.state_dict(),
                    'target_qmix_net_state': self.target_qmix_net.state_dict()
                })
            elif self.args.alg == 'vdn':
                checkpoint.update({
                    'vdn_net_state': self.vdn_net.state_dict(),
                    'target_vdn_net_state': self.target_vdn_net.state_dict()
                })


            torch.save(checkpoint, model_path)
            print(f"模型已保存至：{model_path}")
            self._cleanup_old_models(keep=5)

        except Exception as e:
            print(f"保存模型时发生错误：{str(e)}")
            raise

    def _cleanup_old_models(self, keep=3):
        """清理旧模型，保留最近的keep个版本"""
        try:
            # 获取所有模型文件并按修改时间排序
            model_files = sorted(
                [f for f in os.listdir(self.model_dir) if f.endswith('.pt')],
                key=lambda x: os.path.getmtime(os.path.join(self.model_dir, x)),
                reverse=True
            )

            # 删除超出保留数量的旧模型
            for old_file in model_files[keep:]:
                os.remove(os.path.join(self.model_dir, old_file))
                print(f"已清理旧模型：{old_file}")

        except Exception as e:
            print(f"清理旧模型时出错：{str(e)}")

    def load_model(self, model_path=None, map_location=None):
        """
            优化后的模型加载方法
        """
        try:
            if model_path is None:
                model_files = [f for f in os.listdir(self.model_dir)
                               if f.startswith(self.args.alg) and f.endswith('.pt')]
                if not model_files:
                    raise FileNotFoundError(f"未找到{self.args.alg}的模型文件")
                model_path = max(
                    [os.path.join(self.model_dir, f) for f in model_files],
                    key=os.path.getctime
                )

            checkpoint = torch.load(model_path, map_location=self.device, weights_only= False)

            # 算法兼容性校验
            if checkpoint['algorithm'] != self.args.alg:
                raise ValueError(f"算法类型不匹配: 加载的是{checkpoint['algorithm']}，当前是{self.args.alg}")

            # 加载基础网络参数
            if self.args.alg in ['qmix', 'vdn', 'madqn', 'qmixr2', 'qmixr1']:
                for q_net, state in zip(self.q_networks, checkpoint['q_network_states']):
                    q_net.load_state_dict(state)
                for t_net, state in zip(self.target_q_networks, checkpoint['target_q_net_states']):
                    t_net.load_state_dict(state)
            # 加载MAPPO参数
            if self.args.alg == 'mappo':
                # 加载Actor和Critic网络
                for actor, state in zip(self.actors, checkpoint['actors_state']):
                    actor.load_state_dict(state)
                for critic, state in zip(self.critics, checkpoint['critics_state']):
                    critic.load_state_dict(state)

                # 加载优化器
                for opt, state in zip(self.actor_optimizers, checkpoint['actor_optimizers_state']):
                    opt.load_state_dict(state)
                for opt, state in zip(self.critic_optimizers, checkpoint['critic_optimizers_state']):
                    opt.load_state_dict(state)

            # 加载混合网络参数
            if self.args.alg in {'qmix', 'qmixr2', 'qmixr1'}:
                self.qmix_net.load_state_dict(checkpoint['qmix_net_state'])
                self.target_qmix_net.load_state_dict(checkpoint['target_qmix_net_state'])
            elif self.args.alg == 'vdn':
                self.vdn_net.load_state_dict(checkpoint['vdn_net_state'])
                self.target_vdn_net.load_state_dict(checkpoint['target_vdn_net_state'])

            # 加载优化器状态
            if self.args.alg == 'madqn':
                for opt, state in zip(self.optimizers, checkpoint['optimizers_state']):
                    opt.load_state_dict(state)
            elif self.args.alg in ['qmix', 'vdn', 'qmixr2', 'qmixr1']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            # 加载训练状态
            self.timestep = checkpoint.get('timestep', 0)
            temp_config = checkpoint['temp_config']
            self.temp_start = temp_config['temp_start']
            self.min_temp = temp_config['min_temp']
            self.temp_anneal_steps = temp_config['temp_anneal_steps']

            # 加载训练状态
            self.timestep = checkpoint['timestep']
            temp_config = checkpoint['temp_config']
            self.temp_start = temp_config['temp_start']
            self.min_temp = temp_config['min_temp']
            self.temp_anneal_steps = temp_config['temp_anneal_steps']

            print(f"成功加载模型：{model_path}")
            print(f"恢复训练步数：{self.timestep}，当前温度：{self.get_current_temp():.3f}")

        except KeyError as e:
            print(f"模型版本不兼容，缺少关键字段：{str(e)}")
            raise
        except Exception as e:
            print(f"加载模型时发生错误：{str(e)}")
            raise

    def _log_metrics(self, agent_id, actor_loss, critic_loss, step):
        """增强版监控指标记录"""
        # 标量指标
        self.writer.add_scalar(f"Loss/Agent_{agent_id}/Actor", actor_loss.item(), step)
        self.writer.add_scalar(f"Loss/Agent_{agent_id}/Critic", critic_loss.item(), step)



    def cosine_annealing(self, epoch):
        initial_lr = self.args.lr
        min_lr = initial_lr * 0.01
        progress = min(epoch / self.args.total_episodes, 1.0)
        # 余弦计算
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        # 计算新学习率
        new_lr = min_lr + (initial_lr - min_lr) * cosine_decay

        # 根据算法类型选择优化器更新方式
        if self.args.alg == 'madqn':
            # MADQN需要更新多个独立优化器
            if not hasattr(self, 'optimizers'):
                raise AttributeError("MADQN算法需要optimizers属性")
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

        elif self.args.alg in {'qmix', 'vdn', 'qmixr2', 'qmixr1'}:
            # QMIX/VDN使用单个优化器
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr


        elif self.args.alg == 'mappo':
            # 更新 actor_optimizers 的学习率
            for optimizer in self.actor_optimizers:
                optimizer.param_groups[0]['lr'] = new_lr
            # 更新 critic_optimizers 的学习率
            for optimizer in self.critic_optimizers:
                optimizer.param_groups[0]['lr'] = new_lr


        self.writer.add_scalar('Train/LearningRate', new_lr, self.timestep)




    def close(self):
        """关闭 TensorBoard 写入器"""
        self.writer.close()

