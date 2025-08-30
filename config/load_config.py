from pathlib import Path

from Utils.io import read_parameters_from_yaml

base_path = Path(__file__).parent.parent
default_config_path = base_path.joinpath('config/config.yaml').resolve().as_posix().replace('\\', '')


def load_config(config_path=default_config_path):
    parameters = read_parameters_from_yaml(config_path)
    return parameters

if __name__ == '__main__':
    parameters = load_config()
    print(parameters)