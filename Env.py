import copy
import datetime
import json
import os
import sys
import threading
import time
from pathlib import Path
from Model.ReplayBuffer import MAPPOReplayBuffer
# from scipy.stats import norm as norm
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, app, Response



from Model.ReplayBuffer import ReplayBuffer


sys.path.insert(0, r"D:/EnergyPlusV24-1-0")
# sys.path.insert(0, "/root/autodl-tmp/EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu22.04-x86_64")
from pyenergyplus.api import EnergyPlusAPI

from Utils.RL_Utils import calculate_reward
from Utils.dataloader import Data_Bank
from Utils.io import delete_folder, output_to_csv, prepare_hvac_datasheet
from config.load_config import load_config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
base_path = Path(__file__).parent.parent
init_flag = True
running_status = "Initializing"
current_action = None
current_reward = None
app = Flask(__name__)

global Qmix_action_list
def init_actions_map(args):
    global Qmix_action_list
    Qmix_action_list = []
    if args.alg == 'qmixr1':
        for HC in [0, 1]:
            Qmix_action_list.append([HC])
    if args.alg == 'qmixr2':
        for HC in [0, 1]:
            for ECW in [0, 5]:
                Qmix_action_list.append([HC, ECW])
    else:
        for HC in [0, 1]:
            for ECW in [0, 1, 2, 3, 4, 5]:
                Qmix_action_list.append([HC, ECW])

#获取适宜温度
def cal_comfort_temp(outside_temp):
    if outside_temp < 10.0 :
        [T_lower,T_upper] = [17.45,19.45]
    elif outside_temp > 30.0 :
        [T_lower, T_upper] = [26.5,28.5]
    else :
        [T_lower, T_upper] = [13.45 + 0.43 * outside_temp, 15.45 + 0.43 * outside_temp]
    return {
        'T_lower': T_lower,
        'T_upper': T_upper
    }
#获取适宜温度


def init_running_env():
    # Setting Running Env
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    TORCH_CUDA_ARCH_LIST = "11.8"

    ep_output_folder_path = "sim_ep_output"
    delete_folder(ep_output_folder_path)
    os.mkdir(ep_output_folder_path)




def get_action_states(index):
    # Ensure the index is within the range of Combine_action_list
    if index < 0 or index >= len(Qmix_action_list):
        return "Index out of range."

    # Retrieve the specific action list for the given index
    action = Qmix_action_list[index]

    # Format each item using f-string
    return (f"Zone_1: {action[0]}, Zone_2: {action[1]}, Zone_3: {action[2]}, "
            f"Zone_4: {action[3]}, Zone_5: {action[4]}, Zone_6: {action[5]}")


#面板显示
@app.route('/metrics')
def metrics_stream():
    def event_stream():
        while True:
            instance = EpEnv.instances[-1]
            data = instance.data_callback()  # 调用 callback 获取最新数据
            yield f"data: {data}\n\n"
            time.sleep(1)
    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/')
def index():
    return render_template('index.html')

def run_app():
    app.run(threaded=True, port=5004)


class EpEnv:
    instances = []

    def __init__(self, args, agents):
        # 在子线程中启动 Flask 应用
        self.episode_returns = []
        thread = threading.Thread(target=run_app)
        thread.start()
        EpEnv.instances.append(self)
        self.args = args
        print(EpEnv.instances)
        # 其他主线程的任务
        print("Flask app is running in a separate thread.")
        if self.args.alg in {'qmix', 'vdn', 'madqn', 'qmixr1', 'qmixr2'}:
            self.buffer = ReplayBuffer(self.args)
        elif self.args.alg == 'mappo':
            self.buffer = MAPPOReplayBuffer()




        init_running_env()
        init_actions_map(self.args)


        parameters = load_config()
        self.running_status = "Loading Config"

        self.EPlus_file = parameters['EPlus_file']
        self.weather_data = parameters['weather_data']
        self.HVAC_output = parameters['HVAC_output']


        self.save_idf = parameters['save_idf']
        self.Roof_Switch = parameters['Roof_Switch']
        self.RL_flag = bool(parameters['RL_flag'])

        self.episode_step = 1

        self.FPS = int(parameters['FPS'])
        self.T_factor_day = float(parameters['T_factor_day'])
        self.E_factor_day = float(parameters['E_factor_day'])
        self.T_factor_night = float(parameters['T_factor_night'])
        self.E_factor_night = float(parameters['E_factor_night'])
        self.T_bottom = int(parameters['T_bottom'])
        self.T_top = int(parameters['T_top'])
        self.running_status = "Checking Running Environment"
        self.filename_to_run = self.save_idf
        self.api = EnergyPlusAPI()
        self.E_state = 0

        self.evaluation = False

        self.E_HVAC_all_RBC = 0
        if self.HVAC_output:
            self.HVAC_file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

        self.agents = agents
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape


        self.obs = 0

        self.state = np.zeros((self.args.n_agents, self.args.n_actions))

        self.o, self.u, self.r, self.s, self.avail_u, self.u_onehot, self.terminate, self.padded = [], [], [], [], [], [], [], []
        self.last_action = np.zeros((self.args.n_agents, self.args.n_actions))

        # 进行RBC Baseline 仿真

        print('torch.version: ', torch.__version__)
        print('torch.version.cuda: ', torch.version.cuda)
        print('torch.cuda.is_available: ', torch.cuda.is_available())
        print('torch.cuda.device_count: ', torch.cuda.device_count())
        print('torch.cuda.current_device: ', torch.cuda.current_device())
        device_default = torch.cuda.current_device()
        torch.cuda.device(device_default)
        print('torch.cuda.get_device_name: ', torch.cuda.get_device_name(device_default))

        '''
            E with rule based
        '''
        print(20 * '-', "Preparing for Read Data", 20 * '-')
        running_status = 'Reading Data'
        self.EPLUS = Data_Bank()
        EPLUS = self.EPLUS

        EPLUS.RL_flag = False
        api = self.api

        E_state = api.state_manager.new_state()
        api.runtime.set_console_output_status(E_state, False)
        api.runtime.callback_begin_zone_timestep_after_init_heat_balance(E_state, self.callback_Qmix)
        running_status = 'Running Standard Simulation'
        api.runtime.run_energyplus(E_state,
                                   [
                                       '-w', self.weather_data,
                                       '-d', 'sim_ep_output/',
                                       self.filename_to_run
                                   ]
                                   )

        # If you need to call run_energyplus again, then reset the state first
        api.state_manager.reset_state(E_state)
        api.state_manager.delete_state(E_state)

        self.E_HVAC_all_RBC = copy.deepcopy(EPLUS.E_HVAC_all)



        print(20*'-', 'Result', 20*'-')
        print('Standard test, annual Energy: ', np.sum(EPLUS.E_HVAC_all))
        print('Standard test, annual T_mean: ', np.mean(EPLUS.T_mean_list))
        print('Standard test, annual T_diff: ', np.mean(EPLUS.T_diff))
        print('Standard test, annual T_var: ', np.mean(EPLUS.T_var))

    def hvac_action(self, action, temp):
        if action == 0:
            H_new = self.T_bottom
            C_new = self.T_top

        elif action == 1:
            H_new = temp[0]
            C_new = temp[1]

        return int(H_new), int(C_new)

    def callback_Qmix(self,state_argument):

        global running_status, current_reward, current_action
        EPLUS = self.EPLUS

        api = self.api

        time_interval = EPLUS.time_interval  # 时间步

        o, u, r, s, avail_u, u_onehot, terminate, padded = self.o, self.u, self.r, self.s, self.avail_u, self.u_onehot, self.terminate, self.padded
        self.o, self.u, self.r, self.s, self.avail_u, self.u_onehot, self.terminate, self.padded = o, u, r, s, avail_u, u_onehot, terminate, padded


        '''
        Read data
        '''
        if not EPLUS.got_handles:
            if not api.exchange.api_data_fully_ready(state_argument):
                return

            EPLUS.get_handle(api, state_argument)

            handle_list = EPLUS.handle_availability()
            if -1 in handle_list:
                print("***Invalid handles, check spelling and sensor/actuator availability")
                sys.exit(1)
            EPLUS.got_handles = True
        if api.exchange.warmup_flag(state_argument):
            return
        ''' Time '''
        current_time_stamp = EPLUS.get_current_time_stamp(api, state_argument)

        '''Temperature'''
        EPLUS.get_and_store_temperature_data(api, state_argument)

        '''
        Store data
        '''

        hour = current_time_stamp["hour"]
        time_step = current_time_stamp["time_step"]

        EPLUS.get_and_store_meter_data(api, state_argument)

        EPLUS.sun_is_up.append(api.exchange.sun_is_up(state_argument))

        EPLUS.is_raining.append(api.exchange.today_weather_is_raining_at_time(state_argument, hour, time_step))

        EPLUS.outdoor_humidity.append(
            api.exchange.today_weather_outdoor_relative_humidity_at_time(state_argument, hour, time_step))

        EPLUS.wind_speed.append(api.exchange.today_weather_wind_speed_at_time(state_argument, hour, time_step))

        EPLUS.diffuse_solar.append(api.exchange.today_weather_diffuse_solar_at_time(state_argument, hour, time_step))

        dt = EPLUS.get_datetime_obj_and_update_time(api, state_argument)

        ''' 
        DQN 

        '''
        if not EPLUS.RL_flag:
            EPLUS.episode_reward.append(0)

        if EPLUS.RL_flag:
            if time_interval == 0:
                EPLUS.episode_reward.append(0)
                EPLUS.action_list.append(np.zeros(self.args.n_agents))

                if self.args.alg == 'mappo':
                    EPLUS.old_log_probs.append(np.zeros([self.args.n_agents, self.args.n_actions]))

            '''
            Replay
            '''

            terminated = False
            is_worktime = EPLUS.work_time[-1]

            if time_interval >= 1:
                SA0 = EPLUS.y_solar[-2][2] #时分角
                SH0 = EPLUS.y_solar[-2][1] #高度角
                O0 = EPLUS.y_outdoor[-2]
                W0 = EPLUS.work_time[-2]


                solar_Win10 = EPLUS.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2001[-2]
                solar_Win20 = EPLUS.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2002[-2]
                solar_Win30 = EPLUS.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2003[-2]
                solar_Win40 = EPLUS.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2004[-2]
                solar_Win50 = EPLUS.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2005[-2]
                solar_Win60 = EPLUS.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2006[-2]

                Win_10 = EPLUS.win_2001[-2]
                Win_20 = EPLUS.win_2002[-2]
                Win_30 = EPLUS.win_2003[-2]
                Win_40 = EPLUS.win_2004[-2]
                Win_50 = EPLUS.win_2005[-2]
                Win_60 = EPLUS.win_2006[-2]

                T_10 = EPLUS.y_zone_temp_2001[-2]
                T_20 = EPLUS.y_zone_temp_2002[-2]
                T_30 = EPLUS.y_zone_temp_2003[-2]
                T_40 = EPLUS.y_zone_temp_2004[-2]
                T_50 = EPLUS.y_zone_temp_2005[-2]
                T_60 = EPLUS.y_zone_temp_2006[-2]

                H_10 = EPLUS.hvac_htg_2001[-2]
                H_20 = EPLUS.hvac_htg_2002[-2]
                H_30 = EPLUS.hvac_htg_2003[-2]
                H_40 = EPLUS.hvac_htg_2004[-2]
                H_50 = EPLUS.hvac_htg_2005[-2]
                H_60 = EPLUS.hvac_htg_2006[-2]

                state_0 = np.array([SA0 / 100, SH0 / 100, O0 / 100, W0 / 2,
                                    solar_Win10 / 1000, solar_Win20 / 1000, solar_Win30 / 1000, solar_Win40 / 1000,
                                    solar_Win50 / 1000, solar_Win60 / 1000,
                                    Win_10 / 100, Win_20 / 100, Win_30 / 100, Win_40 / 100, Win_50 / 100, Win_60 / 100,
                                    T_10 / 100, T_20 / 100, T_30 / 100, T_40 / 100, T_50 / 100, T_60 / 100,
                                    H_10 / 100, H_20 / 100, H_30 / 100, H_40 / 100, H_50 / 100, H_60 / 100])

                obs_agent_00 = np.array([SA0 / 100, SH0 / 100, O0 / 100, W0 / 2,
                                         solar_Win10 / 1000,
                                         Win_10 / 100,
                                         T_10 / 100,
                                         H_10 / 100, ])
                obs_agent_10 = np.array([SA0 / 100, SH0 / 100, O0 / 100, W0 / 2,
                                         solar_Win20 / 1000,
                                         Win_20 / 100,
                                         T_20 / 100,
                                         H_20 / 100, ])
                obs_agent_20 = np.array([SA0 / 100, SH0 / 100, O0 / 100, W0 / 2,
                                         solar_Win30 / 1000,
                                         Win_30 / 100,
                                         T_30 / 100,
                                         H_30 / 100, ])
                obs_agent_30 = np.array([SA0 / 100, SH0 / 100, O0 / 100, W0 / 2,
                                         solar_Win40 / 1000,
                                         Win_40 / 100,
                                         T_40 / 100,
                                         H_40 / 100, ])
                obs_agent_40 = np.array([SA0 / 100, SH0 / 100, O0 / 100, W0 / 2,
                                         solar_Win50 / 1000,
                                         Win_50 / 100,
                                         T_50 / 100,
                                         H_50 / 100, ])
                obs_agent_50 = np.array([SA0 / 100, SH0 / 100, O0 / 100, W0 / 2,
                                         solar_Win60 / 1000,
                                         Win_60 / 100,
                                         T_60 / 100,
                                         H_60 / 100, ])

                obs = np.vstack([obs_agent_00, obs_agent_10, obs_agent_20, obs_agent_30, obs_agent_40, obs_agent_50, ])

                action_0 = EPLUS.action_list[-1]

                if self.args.alg == 'mappo':
                    old_log_probs_0 = EPLUS.old_log_probs[-1]

            SA1 = EPLUS.y_solar[-1][2]
            SH1 = EPLUS.y_solar[-1][1]
            O1 = EPLUS.y_outdoor[-1]
            E1 = EPLUS.E_HVAC[-1]
            W1 = EPLUS.work_time[-1]


            solar_Win11 = EPLUS.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2001[-1]
            solar_Win21 = EPLUS.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2002[-1]
            solar_Win31 = EPLUS.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2003[-1]
            solar_Win41 = EPLUS.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2004[-1]
            solar_Win51 = EPLUS.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2005[-1]
            solar_Win61 = EPLUS.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2006[-1]

            T_11 = EPLUS.y_zone_temp_2001[-1]
            T_21 = EPLUS.y_zone_temp_2002[-1]
            T_31 = EPLUS.y_zone_temp_2003[-1]
            T_41 = EPLUS.y_zone_temp_2004[-1]
            T_51 = EPLUS.y_zone_temp_2005[-1]
            T_61 = EPLUS.y_zone_temp_2006[-1]

            H_11 = EPLUS.hvac_htg_2001[-1]
            H_21 = EPLUS.hvac_htg_2002[-1]
            H_31 = EPLUS.hvac_htg_2003[-1]
            H_41 = EPLUS.hvac_htg_2004[-1]
            H_51 = EPLUS.hvac_htg_2005[-1]
            H_61 = EPLUS.hvac_htg_2006[-1]

            # 17-22
            Win_11 = EPLUS.win_2001[-1]
            Win_21 = EPLUS.win_2002[-1]
            Win_31 = EPLUS.win_2003[-1]
            Win_41 = EPLUS.win_2004[-1]
            Win_51 = EPLUS.win_2005[-1]
            Win_61 = EPLUS.win_2006[-1]

            # t_1
            state_1 = np.array([SA1 / 100, SH1 / 100, O1 / 100, W1 / 2,
                                solar_Win11 / 1000, solar_Win21 / 1000, solar_Win31 / 1000, solar_Win41 / 1000,
                                solar_Win51 / 1000, solar_Win61 / 1000,
                                Win_11 / 100, Win_21 / 100, Win_31 / 100, Win_41 / 100, Win_51 / 100, Win_61 / 100,
                                T_11 / 100, T_21 / 100, T_31 / 100, T_41 / 100, T_51 / 100, T_61 / 100,
                                H_11 / 100, H_21 / 100, H_31 / 100, H_41 / 100, H_51 / 100, H_61 / 100, ])

            obs_agent_01 = np.array([SA1 / 100, SH1 / 100, O1 / 100, W1 / 2,
                                     solar_Win11 / 1000,
                                     Win_11 / 100,
                                     T_11 / 100,
                                     H_11 / 100, ])
            obs_agent_11 = np.array([SA1 / 100, SH1 / 100, O1 / 100, W1 / 2,
                                     solar_Win21 / 1000,
                                     Win_21 / 100,
                                     T_21 / 100,
                                     H_21 / 100, ])
            obs_agent_21 = np.array([SA1 / 100, SH1 / 100, O1 / 100, W1 / 2,
                                     solar_Win31 / 1000,
                                     Win_31 / 100,
                                     T_31 / 100,
                                     H_31 / 100, ])
            obs_agent_31 = np.array([SA1 / 100, SH1 / 100, O1 / 100, W1 / 2,
                                     solar_Win41 / 1000,
                                     Win_41 / 100,
                                     T_41 / 100,
                                     H_41 / 100, ])
            obs_agent_41 = np.array([SA1 / 100, SH1 / 100, O1 / 100, W1 / 2,
                                     solar_Win51 / 1000,
                                     Win_51 / 100,
                                     T_51 / 100,
                                     H_51 / 100, ])
            obs_agent_51 = np.array([SA1 / 100, SH1 / 100, O1 / 100, W1 / 2,
                                     solar_Win61 / 1000,
                                     Win_61 / 100,
                                     T_61 / 100,
                                     H_61 / 100, ])

            next_obs = np.vstack([obs_agent_01, obs_agent_11, obs_agent_21, obs_agent_31, obs_agent_41, obs_agent_51])

            '''

            choose action

            '''


            if self.args.alg in {'qmix', 'vdn', 'madqn', 'qmixr2', 'qmixr1'}:
                actions = self.agents.choose_action(next_obs,
                                                   evaluate=self.evaluation)
            elif self.args.alg == 'mappo':
                actions, old_log_probs= self.agents.select_action(next_obs, explore = not self.evaluation)
                EPLUS.old_log_probs.append(old_log_probs)


            action_1 = np.array(actions)
            EPLUS.action_list.append(action_1)


            zone1_action = Qmix_action_list[action_1[0]]
            zone2_action = Qmix_action_list[action_1[1]]
            zone3_action = Qmix_action_list[action_1[2]]
            zone4_action = Qmix_action_list[action_1[3]]
            zone5_action = Qmix_action_list[action_1[4]]
            zone6_action = Qmix_action_list[action_1[5]]

            self.current_actions = [zone1_action,
                                    zone2_action,
                                    zone3_action,
                                    zone4_action,
                                    zone5_action,
                                    zone6_action]

            if self.args.alg != "qmixr1":

                ec_action1 = zone1_action[1]
                ec_action2 = zone2_action[1]
                ec_action3 = zone3_action[1]
                ec_action4 = zone4_action[1]
                ec_action5 = zone5_action[1]
                ec_action6 = zone6_action[1]

                api.exchange.set_actuator_value(state_argument, EPLUS.zone1_window1_construct_handle,
                                                EPLUS.get_ec_device_handle()[ec_action1])
                api.exchange.set_actuator_value(state_argument, EPLUS.zone2_window1_construct_handle,
                                                EPLUS.get_ec_device_handle()[ec_action2])
                api.exchange.set_actuator_value(state_argument, EPLUS.zone3_window1_construct_handle,
                                                EPLUS.get_ec_device_handle()[ec_action3])
                api.exchange.set_actuator_value(state_argument, EPLUS.zone4_window2_construct_handle,
                                                EPLUS.get_ec_device_handle()[ec_action4])
                api.exchange.set_actuator_value(state_argument, EPLUS.zone4_window1_construct_handle,
                                                EPLUS.get_ec_device_handle()[ec_action4])
                api.exchange.set_actuator_value(state_argument, EPLUS.zone5_window1_construct_handle,
                                                EPLUS.get_ec_device_handle()[ec_action5])
                api.exchange.set_actuator_value(state_argument, EPLUS.zone6_window1_construct_handle,
                                                EPLUS.get_ec_device_handle()[ec_action6])

                # 获取热舒适度温度上下限
            temperature_data = EPLUS.get_temperature_data(api, state_argument)
            comfort_temp = cal_comfort_temp(temperature_data["oa_temp"])
            comfort_temp['T_lower'] = 22
            comfort_temp['T_upper'] = 24

            set_temp = [comfort_temp['T_lower'], comfort_temp['T_upper']]

            # Take action
            H_new_1, C_new_1 = self.hvac_action(zone1_action[0], set_temp)
            H_new_2, C_new_2 = self.hvac_action(zone2_action[0], set_temp)
            H_new_3, C_new_3 = self.hvac_action(zone3_action[0], set_temp)
            H_new_4, C_new_4 = self.hvac_action(zone4_action[0], set_temp)
            H_new_5, C_new_5 = self.hvac_action(zone5_action[0], set_temp)
            H_new_6, C_new_6 = self.hvac_action(zone6_action[0], set_temp)

            api.exchange.set_actuator_value(state_argument,
                                            EPLUS.hvac_htg_2001_handle, H_new_1)
            api.exchange.set_actuator_value(state_argument,
                                            EPLUS.hvac_clg_2001_handle, C_new_1)
            api.exchange.set_actuator_value(state_argument,
                                            EPLUS.hvac_htg_2002_handle, H_new_2)
            api.exchange.set_actuator_value(state_argument,
                                            EPLUS.hvac_clg_2002_handle, C_new_2)
            api.exchange.set_actuator_value(state_argument,
                                            EPLUS.hvac_htg_2003_handle, H_new_3)
            api.exchange.set_actuator_value(state_argument,
                                            EPLUS.hvac_clg_2003_handle, C_new_3)
            api.exchange.set_actuator_value(state_argument,
                                            EPLUS.hvac_htg_2004_handle, H_new_4)
            api.exchange.set_actuator_value(state_argument,
                                            EPLUS.hvac_clg_2004_handle, C_new_4)
            api.exchange.set_actuator_value(state_argument,
                                            EPLUS.hvac_htg_2005_handle, H_new_5)
            api.exchange.set_actuator_value(state_argument,
                                            EPLUS.hvac_clg_2005_handle, C_new_5)
            api.exchange.set_actuator_value(state_argument,
                                            EPLUS.hvac_htg_2006_handle, H_new_6)
            api.exchange.set_actuator_value(state_argument,
                                            EPLUS.hvac_clg_2006_handle, C_new_6)

            EPLUS.action_list.append(action_1)

         
            if self.HVAC_output:
                data = {
                    'date': dt.strftime("%Y-%m-%d"),
                    'time': dt.strftime("%H:%M"),
                    'outdoor temperature': temperature_data["oa_temp"],
                    'indoor temperature 1': temperature_data['zone_temp_2001'],
                    'indoor temperature 2': temperature_data['zone_temp_2002'],
                    'indoor temperature 3': temperature_data['zone_temp_2003'],
                    'indoor temperature 4': temperature_data['zone_temp_2004'],
                    'indoor temperature 5': temperature_data['zone_temp_2005'],
                    'indoor temperature 6': temperature_data['zone_temp_2006'],
                    'EHVAC': [E1]
                }

                output_to_csv(f'./sim_hvac_output/{self.HVAC_file_name}.csv', data)

            ''' 
            reward define 
            '''

            T_values = [T_11, T_21, T_31, T_41, T_51, T_61]
            rewards = calculate_reward(comfort_temp,
                                       is_worktime,
                                       E1,
                                       T_values,
                                       self.E_factor_day,
                                       self.T_factor_day,
                                       self.args)

            reward_1 = rewards['reward']
            current_reward = reward_1
            EPLUS.episode_reward.append(reward_1)
            EPLUS.episode_return = EPLUS.episode_return + reward_1

            if self.args.alg in {'mappo', "madqn"}:
                reward = rewards['reward_per']



            if is_worktime:
                T_mean = np.mean(T_values)
                T_upper = comfort_temp['T_upper'] + 2
                T_lower = comfort_temp['T_lower'] - 2
                if T_mean > T_upper:
                    EPLUS.T_Violation.append(T_mean - T_upper)
                elif T_mean < T_lower:
                    EPLUS.T_Violation.append(T_lower - T_mean)

            if ((time_interval >= 1 and not self.evaluation) and
                    (self.args.alg == 'mappo')):
                self.buffer.store_transition(
                    obs=obs,
                    actions=action_0,
                    rewards=reward,
                    next_obs=next_obs,
                    done=terminated,
                    global_state=state_0,
                    next_global_state=state_1,
                    old_log_probs=old_log_probs_0
                                    )

                if len(self.buffer) >= self.args.buffer_size:
                    batch_data = self.buffer.get_batch()
                    self.buffer.clear()
                    self.agents.update(batch_data)


            elif ((time_interval >= 1 and not self.evaluation) and
                    (self.args.alg in {'qmix' , 'vdn' , 'madqn', 'qmixr2', 'qmixr1'})) :
                if self.args.alg == 'madqn':
                    buffer_reward = reward
                else:
                    buffer_reward = reward_1
                self.buffer.add(state_0,
                                obs,
                                np.reshape(action_0, [self.n_agents, 1]),
                                buffer_reward,
                                state_1,
                                next_obs,
                                terminated)


                if self.buffer.size > self.args.minimal_size:
                    self.agents.update(self.buffer.sample(self.args.batch_size))






        T_mean = EPLUS.T_mean_list

        EPLUS.time_interval = EPLUS.time_interval + 1

        '''
        Plot

        '''
        if time_interval % 1000 == 0:
            if EPLUS.RL_flag:
                reward_T = rewards["reward_T"]
                reward_E = rewards["reward_E"]

                print('%d / %s   %.2f / 22   %.3f / %.3f (T/E)' % (
                    time_interval, dt, T_mean[-1], reward_T, reward_E))

    def generate_ep_episode(self, evaluate=False):
        '''
        DQN
        '''
        self.evaluation = evaluate

        EPLUS = Data_Bank()
        self.EPLUS = EPLUS
        EPLUS.time_interval = 0
        api = self.api

        if self.evaluation:
            self.HVAC_output = True
            if self.HVAC_output:
                self.HVAC_file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
                prepare_hvac_datasheet(self.HVAC_file_name)

        print(20 * '-', "Start Training", 20 * '-')
        time_start = datetime.datetime.now()
        print('\n Training iteration: ', self.episode_step)


        EPLUS.RL_flag = self.RL_flag

        E_state = api.state_manager.new_state()
        self.E_state = E_state

        api.runtime.set_console_output_status(E_state, False)

        api.runtime.callback_begin_zone_timestep_after_init_heat_balance(E_state,
                                                                         self.callback_Qmix)

        api.runtime.run_energyplus(E_state,
                                   [
                                       '-w', self.weather_data,
                                       '-d', 'sim_ep_output/',
                                       self.filename_to_run
                                   ]
                                   )
        T_violation = len(EPLUS.T_Violation) / len(EPLUS.x)

        E_HVAC_all_DQN = copy.deepcopy(EPLUS.E_HVAC_all)
        x_sum_1 = np.sum(self.E_HVAC_all_RBC)
        x_sum_2 = np.sum(E_HVAC_all_DQN)
        E_save = (x_sum_1 - x_sum_2) / x_sum_1

        api.state_manager.reset_state(E_state)
        api.state_manager.delete_state(E_state)

        time_end = datetime.datetime.now()
        time_round = time_end - time_start

        print(f'Training iteration {self.episode_step} finished, total time cost: {time_round}')
        print(f'Energy cost: {x_sum_2}/{x_sum_1}')
        print(f'episode_return:{EPLUS.episode_return}')

        self.agents.writer.add_scalar("Reward", EPLUS.episode_return, self.episode_step)
        if self.args.lr_cosine_annealing:
            self.agents.cosine_annealing(self.episode_step)

        T_violation_offset = np.mean(EPLUS.T_Violation)

        print(f'Energy saving ratio: {E_save * 100} %')
        print(f'Temperature violation: {T_violation * 100} %')
        print(f'Temperature violation offset: {T_violation_offset}')

        self.episode_returns.append(EPLUS.episode_return)
        print('Result has been saved...\n')
        self.episode_step += 1
        return EPLUS.episode_return, E_save, T_violation, T_violation_offset

    # 面板显示
    def data_callback(self):
        global init_flag, running_status, current_action, current_reward
        EPLUS = self.EPLUS
        handle_status = EPLUS.get_handle_status()
        current_action_detail = "No action" if "DQN" not in running_status else (
            "No action" if len(EPLUS.action_list) < 1 else get_action_states(EPLUS.action_list[-1]))
        if handle_status:
            active_handles = handle_status['active_handles']
            inactive_handles = handle_status['inactive_handles']
            current_timestamp = EPLUS.current_time_stamp
            zone_temp = EPLUS.get_current_zone_temp()
            episode_reward = 0 if init_flag else (EPLUS.episode_reward[-1] if EPLUS.episode_reward else 0)
            time_line = 0 if init_flag else (EPLUS.time_line[-1] if EPLUS.time_line else 0)
            T_mean = 0 if init_flag else (EPLUS.T_mean_list[-1] if EPLUS.T_mean_list else 0)
            T_var = 0 if init_flag else (EPLUS.T_var[-1] if EPLUS.T_var else 0)
            score = 0 if init_flag else (EPLUS.score[-1] if EPLUS.score else 0)
            T_Violation = 0 if init_flag else (EPLUS.T_Violation[-1] if EPLUS.T_Violation else 0)
            # print(f"action:{current_action}", f"current_reward:{current_reward}")
            current_actions = self.current_actions
            if current_actions:
                current_action_detail = f"\nzone 1: {current_actions[0]} \n zone 2: {current_actions[1]} \n zone 3: {current_actions[2]} \n zone 4: {current_actions[3]} \n zone 5: {current_actions[4]} \n zone 6: {current_actions[5]}"
            else:
                current_action_detail = "No Action"
            data = {
                'NUM_HVAC': EPLUS.NUM_HVAC,
                'view_distance': EPLUS.view_distance,
                'current_timestamp': current_timestamp,
                'episode_reward': episode_reward,
                'episode_return': EPLUS.episode_return,
                'handle_count': len(active_handles) + len(inactive_handles),
                'active_handle': len(active_handles),
                'inactive_handle': len(inactive_handles),
                'zone_temp': zone_temp,
                'T_mean': T_mean,
                'T_var': T_var,
                'score': score,
                'T_Violation': T_Violation,
                'running_status': running_status,
                'current_reward': current_reward,
                "current_action": current_action_detail,
                "episode_returns": self.episode_returns
            }
            # JSON化数据并返回给前端
            init_flag = False
            return json.dumps(data)
        return

