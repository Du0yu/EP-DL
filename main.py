import torch
from Model.Runner import Runner
from Model.args import get_qmix_args




if __name__ == '__main__':
    device_default = torch.cuda.current_device()
    torch.cuda.device(device_default)
    args = get_qmix_args()
    print('torch.cuda.get_device_name: ', torch.cuda.get_device_name(device_default))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    runner = Runner(args, device)

    runner.run(args.total_episodes)


 