import os
import random
import sys
from collections import deque

import eppy
import numpy as np
import openstudio
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from matplotlib import pyplot as plt

sys.path.insert(0, r"D:/EnergyPlusV24-1-0")
from pyenergyplus.api import EnergyPlusAPI

config = {
    'openstudio_path': 'D:/openstudioapplication-1.8.0/',
    'EPlus_file': 'D:/openstudioapplication-1.8.0/EnergyPlus',
    'osm_name_box': 'exV0.osm',
    'weather_data': 'CHN_Guangdong.Guangzhou.592870_CSWD.epw',
    'iddfile': 'Energy+.idd',
    'save_idf': 'run.idf',
    'modified_idf_file': "modified_test.idf",
    'replay_buffer': 'replay_buffer.pkl',

    'timestep_per_hour': 12,
    'begin_month': 1,
    'begin_day_of_month': 1,
    'end_month': 12,
    'end_day_of_month': 31,

    'state_dim': 14,
    'action_dim': 64,
    'epochs': 10,
    'lr': 0.001,
    'gamma': 0.9,
    'epsilon': 0,

    'target_update': 10,
    'buffer_size': 1000,
    'minimal_size': 200,
    'batch_size': 128,
    'FPS': 1000

}
output_file = 'config.yaml'
# 将参数保存为yaml文件
with open(output_file, 'w') as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False)

print(f'Parameters saved to {output_file}')


def save_parameters_to_txt(parameters, file_path):
    with open(file_path, 'w') as file:
        for key, value in parameters.items():
            file.write(f'{key}: {value}\n')
            print(f"{key}: {value}")
    print(f'Parameters saved to {file_path}')


save_parameters_to_txt(config, 'parameters.txt')


# 参数保存至txt文件
class Parameters:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


config = Parameters(**config)

current_dir = os.getcwd()
osm_path = os.path.join(current_dir, config.osm_name_box)
osm_path = openstudio.path(osm_path)  # I guess this is how it wants the path for the translator
translator = openstudio.osversion.VersionTranslator()
# Create an example model
m = translator.loadModel(osm_path).get()
timestep = m.getTimestep()
timestep.setNumberOfTimestepsPerHour(config.timestep_per_hour)

# Restrict to one month of simulation
r = m.getRunPeriod()
r.setBeginMonth(config.begin_month)
r.setBeginDayOfMonth(config.begin_day_of_month)
r.setEndMonth(config.end_month)
r.setEndDayOfMonth(config.end_day_of_month)
ft = openstudio.energyplus.ForwardTranslator()
w = ft.translateModel(m)
w.save(openstudio.path(config.save_idf), True)

# 初始化 EnergyPlus API
api = EnergyPlusAPI()

# 获取Exchange接口，用于访问和修改数据
exchange = api.exchange
# 存储仿真时间和温度的列表
time_log = []
temp_log = []


# Read Data
class Data_Bank():

    def __init__(self):
        self.view_distance = 2000
        self.NUM_HVAC = 4
        self.FPS = config.FPS

        self.episode_reward = 0
        self.episode_return = 0

        ''' handles '''
        self.got_handles = False

        self.environment_temp_handle = -1
        self.environment_humd_handle = -1
        self.environment_solar_azi_handle = -1
        self.environment_solar_alt_handle = -1
        self.environment_solar_ang_handle = -1

        self.zone_humd_handle_1 = -1
        self.zone_humd_handle_2 = -1
        self.zone_humd_handle_3 = -1
        self.zone_humd_handle_4 = -1

        self.zone_window_handle_1 = -1
        self.zone_window_handle_2 = -1
        self.zone_window_handle_3 = -1
        self.zone_window_handle_4 = -1

        self.zone_temp_handle_1 = -1
        self.zone_temp_handle_2 = -1
        self.zone_temp_handle_3 = -1
        self.zone_temp_handle_4 = -1

        self.hvac_htg_1_handle = -1
        self.hvac_clg_1_handle = -1
        self.hvac_htg_2_handle = -1
        self.hvac_clg_2_handle = -1
        self.hvac_htg_3_handle = -1
        self.hvac_clg_3_handle = -1
        self.hvac_htg_4_handle = -1
        self.hvac_clg_4_handle = -1

        self.E_Facility_handle = -1
        self.E_HVAC_handle = -1
        self.E_Heating_handle = -1
        self.E_Cooling_handle = -1

        ''' time '''
        self.x = []

        self.years = []
        self.months = []
        self.days = []
        self.hours = []
        self.minutes = []
        self.current_times = []
        self.actual_date_times = []
        self.actual_times = []

        self.weekday = []
        self.isweekday = []
        self.isweekend = []
        self.work_time = []

        ''' building parameters '''
        self.E_Facility = []
        self.E_HVAC = []
        self.E_Heating = []
        self.E_Cooling = []

        ''' DQN '''
        self.action_list = []
        self.episode_reward = []

        self.hvac_htg_2001 = []
        self.hvac_clg_2001 = []
        self.hvac_htg_2002 = []
        self.hvac_clg_2002 = []
        self.hvac_htg_2003 = []
        self.hvac_clg_2003 = []
        self.hvac_htg_2004 = []
        self.hvac_clg_2004 = []
        return

    # 检查句柄可用性
    def handle_availability(self):
        ''' check handle_availability '''
        self.handle_list = [self.environment_temp_handle,
                            self.environment_humd_handle,
                            # self.environment_solar_azi_handle,
                            # self.environment_solar_alt_handle,
                            # self.environment_solar_ang_handle,

                            self.zone_temp_handle_1, self.zone_temp_handle_2,
                            self.zone_temp_handle_3, self.zone_temp_handle_4,
                            self.hvac_htg_1_handle, self.hvac_clg_1_handle,
                            self.hvac_htg_2_handle, self.hvac_clg_2_handle,
                            self.hvac_htg_3_handle, self.hvac_clg_3_handle,
                            self.hvac_htg_4_handle, self.hvac_clg_4_handle,

                            # self.E_Facility_handle,
                            self.E_HVAC_handle,
                            self.E_Heating_handle,
                            self.E_Cooling_handle]
        return self.handle_list


# 定义DQN模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DQN代理
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)[0]).item()
            target_f = self.model(state)
            target_f = target_f.clone()  # Clone to avoid in-place operation
            target_f[0][action] = target
            output = self.model(state)
            loss = self.criterion(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# 定义回调函数，在每个时间步动态修改数据
def modify_data_each_timestep(state):
    # 获取当前仿真时间
    current_time = exchange.current_time(state)

    var_handler = exchange.get_variable_handle(state, u"Zone Air Temperature", u"Thermal Zone 3")
    # print(var_handler)
    # meter_handler = exchange.get_meter_handle(state, u"InteriorLights:Electricity")
    # print(meter_handler)

    # metered_value = exchange.get_meter_value(state, meter_handler)
    # print(f"power:{metered_value}")
    # 获取当前区域温度（例如Zone 1的温度）
    zone_temp = exchange.get_variable_value(state, var_handler)

    # 保存时间和温度数据
    time_log.append(current_time)
    temp_log.append(zone_temp)
    print(f"Current Time: {current_time}, Zone 1 Temperature: {zone_temp}")

    '''
    Read data
    '''
    if not EPLUS.got_handles:
        if not api.exchange.api_data_fully_ready(state):
            return
        # 获取执行器句柄，注意这里需要提供控件类型、控件名称等参数
        EPLUS.environment_temp_handle = api.exchange.get_variable_handle(state, u"SITE OUTDOOR AIR DRYBULB TEMPERATURE",
                                                                         u"ENVIRONMENT")
        EPLUS.environment_humd_handle = api.exchange.get_variable_handle(state, u"Site Outdoor Air Drybulb Temperature",
                                                                         u"ENVIRONMENT")
        # EPLUS.environment_solar_azi_handle   = api.exchange.get_variable_handle(state, u"Site Solar Azimuth Angle", u"ENVIRONMENT")
        # EPLUS.environment_solar_alt_handle   = api.exchange.get_variable_handle(state, u"Site Solar Altitude Angle", u"ENVIRONMENT")
        # EPLUS.environment_solar_ang_handle   = api.exchange.get_variable_handle(state, u"Site Solar Hour Angle", u"ENVIRONMENT")

        EPLUS.zone_window_handle_1 = api.exchange.get_variable_handle(state, "Zone Windows Total Heat Gain Energy",
                                                                      'Thermal Zone 1')
        EPLUS.zone_window_handle_2 = api.exchange.get_variable_handle(state, "Zone Windows Total Heat Gain Energy",
                                                                      'Thermal Zone 2')
        EPLUS.zone_window_handle_3 = api.exchange.get_variable_handle(state, "Zone Windows Total Heat Gain Energy",
                                                                      'Thermal Zone 3')
        EPLUS.zone_window_handle_4 = api.exchange.get_variable_handle(state, "Zone Windows Total Heat Gain Energy",
                                                                      'Thermal Zone 4')

        EPLUS.zone_temp_handle_1 = api.exchange.get_variable_handle(state, "Zone Air Temperature", 'Thermal Zone 1')
        EPLUS.zone_temp_handle_2 = api.exchange.get_variable_handle(state, "Zone Air Temperature", 'Thermal Zone 2')
        EPLUS.zone_temp_handle_3 = api.exchange.get_variable_handle(state, "Zone Air Temperature", 'Thermal Zone 3')
        EPLUS.zone_temp_handle_4 = api.exchange.get_variable_handle(state, "Zone Air Temperature", 'Thermal Zone 4')

        EPLUS.hvac_htg_1_handle = api.exchange.get_actuator_handle(state, 'Zone Temperature Control',
                                                                   'Heating Setpoint', 'Thermal Zone 1')
        EPLUS.hvac_clg_1_handle = api.exchange.get_actuator_handle(state, 'Zone Temperature Control',
                                                                   'Cooling Setpoint', 'Thermal Zone 1')
        EPLUS.hvac_htg_2_handle = api.exchange.get_actuator_handle(state, 'Zone Temperature Control',
                                                                   'Heating Setpoint', 'Thermal Zone 2')
        EPLUS.hvac_clg_2_handle = api.exchange.get_actuator_handle(state, 'Zone Temperature Control',
                                                                   'Cooling Setpoint', 'Thermal Zone 2')
        EPLUS.hvac_htg_3_handle = api.exchange.get_actuator_handle(state, 'Zone Temperature Control',
                                                                   'Heating Setpoint', 'Thermal Zone 3')
        EPLUS.hvac_clg_3_handle = api.exchange.get_actuator_handle(state, 'Zone Temperature Control',
                                                                   'Cooling Setpoint', 'Thermal Zone 3')
        EPLUS.hvac_htg_4_handle = api.exchange.get_actuator_handle(state, 'Zone Temperature Control',
                                                                   'Heating Setpoint', 'Thermal Zone 4')
        EPLUS.hvac_clg_4_handle = api.exchange.get_actuator_handle(state, 'Zone Temperature Control',
                                                                   'Cooling Setpoint', 'Thermal Zone 4')

        # EPLUS.E_Facility_handle = api.exchange.get_meter_handle(state, 'Electricity:Facility')
        EPLUS.E_HVAC_handle = api.exchange.get_meter_handle(state, 'Electricity:HVAC')
        EPLUS.E_Heating_handle = api.exchange.get_meter_handle(state, 'Heating:Electricity')
        EPLUS.E_Cooling_handle = api.exchange.get_meter_handle(state, 'Cooling:Electricity')

        handle_list = EPLUS.handle_availability()
        if -1 in handle_list:
            print("***Invalid handles, check spelling and sensor/actuator availability")
            sys.exit(1)
        EPLUS.got_handles = True
    if api.exchange.warmup_flag(state):
        return

    '''Temperature'''
    environment_temp = api.exchange.get_variable_value(state, EPLUS.environment_temp_handle)

    zone1_temp = api.exchange.get_variable_value(state, EPLUS.zone_temp_handle_1)
    zone2_temp = api.exchange.get_variable_value(state, EPLUS.zone_temp_handle_2)
    zone3_temp = api.exchange.get_variable_value(state, EPLUS.zone_temp_handle_3)
    zone4_temp = api.exchange.get_variable_value(state, EPLUS.zone_temp_handle_4)
    '''Getting Temperature'''
    # Set Heating Temp
    hvac_htg_zone1 = api.exchange.get_actuator_value(state, EPLUS.hvac_htg_1_handle)
    hvac_htg_zone2 = api.exchange.get_actuator_value(state, EPLUS.hvac_htg_2_handle)
    hvac_htg_zone3 = api.exchange.get_actuator_value(state, EPLUS.hvac_htg_3_handle)
    hvac_htg_zone4 = api.exchange.get_actuator_value(state, EPLUS.hvac_htg_4_handle)
    # Set Cooling Temp
    hvac_ctg_zone1 = api.exchange.get_actuator_value(state, EPLUS.hvac_clg_1_handle)
    hvac_ctg_zone2 = api.exchange.get_actuator_value(state, EPLUS.hvac_clg_2_handle)
    hvac_ctg_zone3 = api.exchange.get_actuator_value(state, EPLUS.hvac_clg_3_handle)
    hvac_ctg_zone4 = api.exchange.get_actuator_value(state, EPLUS.hvac_clg_4_handle)
    '''Getting Windows status'''
    zone_window_2001 = api.exchange.get_variable_value(state, EPLUS.zone_window_handle_1)
    zone_window_2002 = api.exchange.get_variable_value(state, EPLUS.zone_window_handle_2)
    zone_window_2003 = api.exchange.get_variable_value(state, EPLUS.zone_window_handle_3)
    zone_window_2004 = api.exchange.get_variable_value(state, EPLUS.zone_window_handle_4)

    # Maybe used to calculate total HVAC energy

    EPLUS.E_Facility.append(api.exchange.get_meter_value(state, EPLUS.E_Facility_handle))
    EPLUS.E_HVAC.append(api.exchange.get_meter_value(state, EPLUS.E_HVAC_handle))
    EPLUS.E_Heating.append(api.exchange.get_meter_value(state, EPLUS.E_Heating_handle))
    EPLUS.E_Cooling.append(api.exchange.get_meter_value(state, EPLUS.E_Cooling_handle))

    # 获取执行器句柄，注意这里需要提供控件类型、控件名称等参数
    zone1_cooling_actuator_handle = exchange.get_actuator_handle(state, "Zone Temperature Control", "Cooling Setpoint",
                                                                 "Thermal Zone 1")
    zone1_heating_actuator_handle = exchange.get_actuator_handle(state, "Zone Temperature Control", "Heating Setpoint",
                                                                 "Thermal Zone 1")
    zone2_cooling_actuator_handle = exchange.get_actuator_handle(state, "Zone Temperature Control", "Cooling Setpoint",
                                                                 "Thermal Zone 2")
    zone2_heating_actuator_handle = exchange.get_actuator_handle(state, "Zone Temperature Control", "Heating Setpoint",
                                                                 "Thermal Zone 2")
    zone3_cooling_actuator_handle = exchange.get_actuator_handle(state, "Zone Temperature Control", "Cooling Setpoint",
                                                                 "Thermal Zone 3")
    zone3_heating_actuator_handle = exchange.get_actuator_handle(state, "Zone Temperature Control", "Heating Setpoint",
                                                                 "Thermal Zone 3")
    zone4_cooling_actuator_handle = exchange.get_actuator_handle(state, "Zone Temperature Control", "Cooling Setpoint",
                                                                 "Thermal Zone 4")
    zone4_heating_actuator_handle = exchange.get_actuator_handle(state, "Zone Temperature Control", "Heating Setpoint",
                                                                 "Thermal Zone 4")
    if zone3_cooling_actuator_handle == -1 or zone3_heating_actuator_handle == -1:
        print("Actuator handle not found")
    else:
        print(
            f"Cooling Actuator Handle: {zone3_cooling_actuator_handle}\n Heating Actuator Handle: {zone3_heating_actuator_handle}")

        # 根据需要修改设定点温度
        new_cooling_setpoint = 26.0  # 修改为新的设定温度值
        new_heating_setpoint = 25.0
        exchange.set_actuator_value(state, zone1_cooling_actuator_handle, new_cooling_setpoint)
        exchange.set_actuator_value(state, zone1_heating_actuator_handle, new_heating_setpoint)
        exchange.set_actuator_value(state, zone2_cooling_actuator_handle, new_cooling_setpoint)
        exchange.set_actuator_value(state, zone2_heating_actuator_handle, new_heating_setpoint)
        exchange.set_actuator_value(state, zone3_cooling_actuator_handle, new_cooling_setpoint)
        exchange.set_actuator_value(state, zone3_heating_actuator_handle, new_heating_setpoint)
        exchange.set_actuator_value(state, zone4_cooling_actuator_handle, new_cooling_setpoint)
        exchange.set_actuator_value(state, zone4_heating_actuator_handle, new_heating_setpoint)
        print(f"Set new temperature setpoint: Cooling: {new_cooling_setpoint} Heating: {new_heating_setpoint}")


# 注册回调函数：每个时间步结束后调用
def start_callback(state):
    api.runtime.callback_after_predictor_after_hvac_managers(state, modify_data_each_timestep)


# 画温度曲线图
def plot_temperature():
    plt.figure(figsize=(10, 6))
    plt.plot(time_log, temp_log, label="Zone 3 Temperature", color="blue", marker="o")
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature (°C)")
    plt.title("Zone 3 Temperature Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("temperature_plot.png")  # 保存图像为PNG文件
    plt.show()  # 显示图像


# 修改并保存IDF文件
def modify_and_save_idf(input_idf_path, output_idf_path, new_cooling_setpoint, new_heating_setpoint):
    with open(input_idf_path, 'r') as idf_file:
        idf_content = idf_file.readlines()

    # Loop through the lines and modify the necessary fields
    for i, line in enumerate(idf_content):
        # Modify the cooling and heating setpoints for "Thermal Zone 3"
        if "ZoneControl:Thermostat" in line and "Thermal Zone 3" in idf_content[i + 1]:
            if "Cooling Setpoint Temperature Schedule" in idf_content[i + 2]:
                idf_content[i + 3] = f"    {new_cooling_setpoint};  !- Cooling Setpoint\n"
            if "Heating Setpoint Temperature Schedule" in idf_content[i + 4]:
                idf_content[i + 5] = f"    {new_heating_setpoint};  !- Heating Setpoint\n"

    # Save the modified content to a new IDF file
    with open(output_idf_path, 'w') as new_idf_file:
        new_idf_file.writelines(idf_content)
    print(f"Modified IDF file saved to: {output_idf_path}")


# 模拟运行
def run_simulation():
    state = api.state_manager.new_state()

    # 注册回调函数
    start_callback(state)

    # 设置输出目录
    output_directory = "sim_output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 修改并保存IDF文件
    modify_and_save_idf(config.save_idf, config.modified_idf_file, 25.0, 25.0)

    # 运行 EnergyPlus 模拟
    api.runtime.run_energyplus(
        state,
        ["-w", config.weather_data, "-r", "-d", output_directory, config.modified_idf_file]
    )

    print(f"Simulation completed. Check results in: {output_directory}")

    # 模拟结束后绘制温度曲线
    plot_temperature()


if __name__ == "__main__":
    EPLUS = Data_Bank()
    run_simulation()
