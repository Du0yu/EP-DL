import argparse

"""
Here are the param for the training

"""

def get_qmix_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--alg', type=str, default='qmix')

    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
    parser.add_argument('--evaluate_epoch', type=int, default=4, help='number of the epoch to evaluate the agent')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to start evaluation')
    parser.add_argument('--total_episodes', type=int, default=100, help='number of the episode to train the agent')
    args = parser.parse_args()


    #模型设定
    args.obs_shape = 8
    args.n_actions = 12
    args.n_agents = 6
    args.state_shape = 28
    if args.alg == 'qmixr2':
        args.n_actions = 4
    elif args.alg == 'qmixr1':
        args.n_actions = 2



    #网络参数设定
    args.mlp_hidden_dim = 128#64
    args.qmix_hidden_dim = 64#128
    args.two_hyper_layers = True
    args.hyper_hidden_dim = 128
    #学习率，余弦衰减
    args.lr = 5e-4
    args.lr_cosine_annealing = True

    # 经验池设定
    args.minimal_size = 256
    args.batch_size = 128
    args.buffer_size = int(1000)

    if args.alg == 'mappo':
        args.epoch = 2
        args.actor_lr = 1e-3
        args.critic_lr = 1e-3
        args.buffer_size = 256
        args.max_grad_norm = 10
        args.critic_update_cycle = 2

        args.actor_update_cycle = 20

        args.gae_lambda = 0.95
        args.entropy_coef = 0.05
        args.clip_eps = 0.1  # PPO clip参数

    elif args.alg == 'madqn':
        args.epochs = 1

        args.grad_norm_clip = 10



    args.temp_start = 1
    args.min_temp = 0.01
    args.temp_anneal_steps = 300000


    # how often to update the target_net
    args.target_update_cycle = 10



    args.log_interval = 5000

    # how often to save the model
    args.save_cycle = 8000




    return args
