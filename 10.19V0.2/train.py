import os
import sys
import numpy as np
import openstudio


#读取参数
def read_parameters_from_txt(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(':')
                parameters[key.strip()] = value.strip()
                print(f"{key.strip()}: {value.strip()}")
    return parameters


#实现空调调控
def HVAC_action(action, temp):
    if action == 0:
        H_new = T_bottom
        C_new = T_top

    elif action == 1:
        H_new = temp[0]
        C_new = temp[1]

    return int(H_new), int(C_new)



if __name__ == '__main__':
    parameters = read_parameters_from_txt('parameters.txt')

    openstudio_path = parameters['openstudio_path']
    EPlus_file = parameters['EPlus_file']
    osm_name_box = parameters['osm_name_box']
    weather_data = parameters['weather_data']
    iddfile = parameters['iddfile']
    save_idf = parameters['save_idf']
    modified_idf_file = parameters['modified_idf_file']

    # replay_buffer=int(parameters['replay_buffer'])
    timestep_per_hour = parameters['timestep_per_hour']
    begin_month = parameters['begin_month']
    begin_day_of_month = parameters['begin_day_of_month']
    end_month = parameters['end_month']
    end_day_of_month = parameters['end_day_of_month']

    state_dim = parameters['state_dim']
    action_dim = parameters['action_dim']
    epochs = int(parameters['epochs'])
    lr = float(parameters['lr'])
    gamma = float(parameters['gamma'])
    epsilon = int(parameters['epsilon'])
    target_update = int(parameters['target_update'])
    buffer_size = int(parameters['buffer_size'])
    minimal_size = int(parameters['minimal_size'])
    batch_size = int(parameters['batch_size'])

    FPS = int(parameters['FPS'])
    T_bottom = int(parameters['T_bottom'])
    T_top = int(parameters['T_top'])



