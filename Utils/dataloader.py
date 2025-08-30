import datetime
import sys

import numpy as np

from config.load_config import load_config

parameters = load_config()
RL_flag = bool(parameters['RL_flag'])
FPS = int(parameters['FPS'])


class Data_Bank:

    def __init__(self):
        self.all_handle_list = None
        self.current_time_stamp = None
        self.view_distance = 2000
        self.NUM_HVAC = 5
        self.FPS = FPS

        self.episode_reward = 0
        self.episode_return = 0

        self.RL_flag = RL_flag
        self.time_interval = 0

        self.time_line = []
        self.T_Violation = []
        self.T_Agents_Violation = []
        self.score = []

        self.T_diff = []
        self.T_mean_list = []
        self.T_var = []

        self.T_map = {}

        ''' handles '''
        self.got_handles = False

        self.oa_temp_handle = -1

        self.oa_solar_azi_handle = -1
        self.oa_solar_alt_handle = -1
        self.oa_solar_ang_handle = -1


        self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2001 = -1
        self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2002 = -1
        self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2003 = -1
        self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2004 = -1
        self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2005 = -1
        self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2006 = -1

        self.zone_window_Heat_loss_Rate_handle_2001 = -1
        self.zone_window_Heat_loss_Rate_handle_2002 = -1
        self.zone_window_Heat_loss_Rate_handle_2003 = -1
        self.zone_window_Heat_loss_Rate_handle_2004 = -1
        self.zone_window_Heat_loss_Rate_handle_2005 = -1
        self.zone_window_Heat_loss_Rate_handle_2006 = -1


        self.zone_humd_handle_2001 = -1
        self.zone_humd_handle_2002 = -1
        self.zone_humd_handle_2003 = -1
        self.zone_humd_handle_2004 = -1
        self.zone_humd_handle_2005 = -1
        self.zone_humd_handle_2006 = -1

        self.zone_ventmass_handle_2001 = -1
        self.zone_ventmass_handle_2002 = -1
        self.zone_ventmass_handle_2003 = -1
        self.zone_ventmass_handle_2004 = -1
        self.zone_ventmass_handle_2005 = -1
        self.zone_ventmass_handle_2006 = -1


        self.zone_temp_handle_2001 = -1
        self.zone_temp_handle_2002 = -1
        self.zone_temp_handle_2003 = -1
        self.zone_temp_handle_2004 = -1
        self.zone_temp_handle_2005 = -1
        self.zone_temp_handle_2006 = -1


        self.hvac_htg_2001_handle = -1
        self.hvac_clg_2001_handle = -1
        self.hvac_htg_2002_handle = -1
        self.hvac_clg_2002_handle = -1
        self.hvac_htg_2003_handle = -1
        self.hvac_clg_2003_handle = -1
        self.hvac_htg_2004_handle = -1
        self.hvac_clg_2004_handle = -1
        self.hvac_htg_2005_handle = -1
        self.hvac_clg_2005_handle = -1
        self.hvac_htg_2006_handle = -1
        self.hvac_clg_2006_handle = -1

        self.E_Light_handle = -1
        self.E_HVAC_handle = -1
        self.E_Heating_handle = -1
        self.E_Cooling_handle = -1

        self.ec_device00_handle = -1
        self.ec_device01_handle = -1
        self.ec_device02_handle = -1
        self.ec_device03_handle = -1
        self.ec_device04_handle = -1
        self.ec_device05_handle = -1

        self.zone1_window1_construct_handle = -1
        self.zone2_window1_construct_handle = -1
        self.zone3_window1_construct_handle = -1
        self.zone4_window1_construct_handle = -1
        self.zone4_window2_construct_handle = -1
        self.zone5_window1_construct_handle = -1
        self.zone6_window1_construct_handle = -1
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
        self.y_temp = []
        self.y_wind = []
        self.y_solar = []

        self.y_zone_humd = []
        self.y_zone_window = []
        self.y_zone_ventmass = []

        self.y_zone_temp = []

        self.y_outdoor = []
        self.y_zone = []
        self.y_htg = []
        self.y_clg = []

        self.y_zone_temp_2001 = []
        self.y_zone_temp_2002 = []
        self.y_zone_temp_2003 = []
        self.y_zone_temp_2004 = []
        self.y_zone_temp_2005 = []
        self.y_zone_temp_2006 = []

        self.y_zone_window_Heat_loss_Rate_2001 = []
        self.y_zone_window_Heat_loss_Rate_2002 = []
        self.y_zone_window_Heat_loss_Rate_2003 = []
        self.y_zone_window_Heat_loss_Rate_2004 = []
        self.y_zone_window_Heat_loss_Rate_2005 = []
        self.y_zone_window_Heat_loss_Rate_2006 = []

        self.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2001 = []
        self.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2002 = []
        self.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2003 = []
        self.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2004 = []
        self.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2005 = []
        self.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2006 = []


        self.sun_is_up = []
        self.is_raining = []
        self.outdoor_humidity = []
        self.wind_speed = []
        self.diffuse_solar = []

        # self.E_Facility = []
        self.E_Light = []
        self.E_HVAC = []
        self.E_Heating = []
        self.E_Cooling = []

        self.E_HVAC_all = []

        ''' DQN '''
        self.action_list = []
        self.old_log_probs = []
        self.old_logits_list = []
        self.episode_reward = []

        self.hvac_htg_2001 = []
        self.hvac_clg_2001 = []
        self.hvac_htg_2002 = []
        self.hvac_clg_2002 = []
        self.hvac_htg_2003 = []
        self.hvac_clg_2003 = []
        self.hvac_htg_2004 = []
        self.hvac_clg_2004 = []
        self.hvac_htg_2005 = []
        self.hvac_clg_2005 = []
        self.hvac_htg_2006 = []
        self.hvac_clg_2006 = []

        self.win_2001 = []
        self.win_2002 = []
        self.win_2003 = []
        self.win_2004 = []
        self.win_2005 = []
        self.win_2006 = []


        return

    def handle_availability(self):
        ''' check handle_availability '''
        self.handle_list = [
            self.oa_temp_handle,
            self.oa_solar_azi_handle,
            self.oa_solar_alt_handle,
            self.oa_solar_ang_handle,

            self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2001,
            self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2002,
            self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2003,
            self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2004,
            self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2005,
            self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2006,

            self.zone_window_Heat_loss_Rate_handle_2001,
            self.zone_window_Heat_loss_Rate_handle_2002,
            self.zone_window_Heat_loss_Rate_handle_2003,
            self.zone_window_Heat_loss_Rate_handle_2004,
            self.zone_window_Heat_loss_Rate_handle_2005,
            self.zone_window_Heat_loss_Rate_handle_2006,


            self.zone_temp_handle_2001, self.zone_temp_handle_2002,
            self.zone_temp_handle_2003, self.zone_temp_handle_2004,
            self.zone_temp_handle_2005, self.zone_temp_handle_2006,

            self.hvac_htg_2001_handle, self.hvac_clg_2001_handle,
            self.hvac_htg_2002_handle, self.hvac_clg_2002_handle,
            self.hvac_htg_2003_handle, self.hvac_clg_2003_handle,
            self.hvac_htg_2004_handle, self.hvac_clg_2004_handle,
            self.hvac_htg_2005_handle, self.hvac_clg_2005_handle,
            self.hvac_htg_2006_handle, self.hvac_clg_2006_handle,

            self.E_Light_handle,
            self.E_HVAC_handle,
            self.E_Heating_handle,
            self.E_Cooling_handle,

            #智能窗设备以及窗户表面插入
            self.ec_device00_handle,
            self.ec_device01_handle,
            self.ec_device02_handle,
            self.ec_device03_handle,
            self.ec_device04_handle,
            self.ec_device05_handle,
            self.zone1_window1_construct_handle,
            self.zone2_window1_construct_handle,
            self.zone3_window1_construct_handle,
            self.zone4_window1_construct_handle,
            self.zone4_window2_construct_handle,
            self.zone5_window1_construct_handle,
            self.zone6_window1_construct_handle,

        ]
        return self.handle_list

    def get_handle(self, api_object, state_argument):
        api = api_object
        if not self.got_handles:
            if not api.exchange.api_data_fully_ready(state_argument):
                return
            self.get_variable_handle_(api, state_argument)
            self.get_actuator_handle_(api, state_argument)
            self.get_meter_handle_(api, state_argument)
            self.get_construct_handle_(api, state_argument)
            handle_list = self.handle_availability()
            self.get_all_handle(api_object, state_argument)
            if -1 in handle_list:
                print("***Invalid handles, check spelling and sensor/actuator availability")
                sys.exit(1)
            self.got_handles = True

    def get_all_handle(self, api, state_argument):
        self.all_handle_list = [
            ("oa_temp_handle", self.oa_temp_handle),
            ("oa_solar_azi_handle", self.oa_solar_azi_handle),
            ("oa_solar_alt_handle", self.oa_solar_alt_handle),
            ("oa_solar_ang_handle", self.oa_solar_ang_handle),

            ("Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2001", self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2001),
            ("Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2002", self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2002),
            ("Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2003", self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2003),
            ("Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2004", self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2004),
            ("Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2005", self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2005),
            ("Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2006", self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2006),

            ("zone_window_Heat_loss_Rate_handle_2001", self.zone_window_Heat_loss_Rate_handle_2001),
            ("zone_window_Heat_loss_Rate_handle_2002", self.zone_window_Heat_loss_Rate_handle_2002),
            ("zone_window_Heat_loss_Rate_handle_2003", self.zone_window_Heat_loss_Rate_handle_2003),
            ("zone_window_Heat_loss_Rate_handle_2004", self.zone_window_Heat_loss_Rate_handle_2004),
            ("zone_window_Heat_loss_Rate_handle_2005", self.zone_window_Heat_loss_Rate_handle_2005),
            ("zone_window_Heat_loss_Rate_handle_2006", self.zone_window_Heat_loss_Rate_handle_2006),



            ("zone_temp_handle_2001", self.zone_temp_handle_2001),
            ("zone_temp_handle_2002", self.zone_temp_handle_2002),
            ("zone_temp_handle_2003", self.zone_temp_handle_2003),
            ("zone_temp_handle_2004", self.zone_temp_handle_2004),
            ("zone_temp_handle_2005", self.zone_temp_handle_2005),
            ("zone_temp_handle_2006", self.zone_temp_handle_2006),
            ("hvac_htg_2001_handle", self.hvac_htg_2001_handle),
            ("hvac_clg_2001_handle", self.hvac_clg_2001_handle),
            ("hvac_htg_2002_handle", self.hvac_htg_2002_handle),
            ("hvac_clg_2002_handle", self.hvac_clg_2002_handle),
            ("hvac_htg_2003_handle", self.hvac_htg_2003_handle),
            ("hvac_clg_2003_handle", self.hvac_clg_2003_handle),
            ("hvac_htg_2004_handle", self.hvac_htg_2004_handle),
            ("hvac_clg_2004_handle", self.hvac_clg_2004_handle),
            ("hvac_htg_2005_handle", self.hvac_htg_2005_handle),
            ("hvac_clg_2005_handle", self.hvac_clg_2005_handle),
            ("hvac_htg_2006_handle", self.hvac_htg_2006_handle),
            ("hvac_clg_2006_handle", self.hvac_clg_2006_handle),
            ("E_Light_handle", self.E_Light_handle),
            ("E_HVAC_handle", self.E_HVAC_handle),
            ("E_Heating_handle", self.E_Heating_handle),
            ("E_Cooling_handle", self.E_Cooling_handle),
        ]
        return self.all_handle_list

    def get_surface_handle(self):
        return {
            "zone1:window1":self.zone1_window1_construct_handle,
            "zone2:window1":self.zone2_window1_construct_handle,
            "zone3:window1":self.zone3_window1_construct_handle,
            "zone4:window1":self.zone4_window1_construct_handle,
            "zone4:window2":self.zone4_window2_construct_handle,
            "zone5:window1":self.zone5_window1_construct_handle,
            "zone6:window1":self.zone6_window1_construct_handle,
        }

    def get_ec_device_handle(self):
        return [
            self.ec_device00_handle,
            self.ec_device01_handle,
            self.ec_device02_handle,
            self.ec_device03_handle,
            self.ec_device04_handle,
            self.ec_device05_handle
        ]

    def switch_windows(self, api, state_argument, surface, device):
        surfaces = self.get_surface_handle()
        devices = self.get_ec_device_handle()
        if surface in surfaces.keys():
            if device in devices.keys():
                api.exchange.set_actuator_value(state_argument, surfaces[surface], devices[device])
            else:
                print(f"***Invalid device name: {device}")
                sys.exit(1)
        else:
            print(f"***Invalid surface name: {surface}")
            sys.exit(1)

    def get_handle_status(self):
        handle_list = self.all_handle_list
        # print(f"handle_list: {handle_list} | got_handles: {self.got_handles}")
        if handle_list:
            active_handles = []
            inactive_handles = []
            if self.got_handles:
                for handle_name, handle_value in handle_list:
                    if handle_value == -1:  # 判断不活动句柄
                        inactive_handles.append(handle_name)
                    else:
                        active_handles.append(handle_name)
                return {"active_handles": active_handles, "inactive_handles": inactive_handles}
        else:
                return None

    def get_variable_handle_(self, api_object, state_argument):
        api = api_object
        if not self.got_handles:
            if not api.exchange.api_data_fully_ready(state_argument):
                return
        self.oa_temp_handle = api.exchange.get_variable_handle(state_argument,
                                                               u"SITE OUTDOOR AIR DRYBULB TEMPERATURE",
                                                               u"ENVIRONMENT")
        self.oa_solar_azi_handle = api.exchange.get_variable_handle(state_argument, u"Site Solar Azimuth Angle",
                                                                    u"ENVIRONMENT")
        self.oa_solar_alt_handle = api.exchange.get_variable_handle(state_argument, u"Site Solar Altitude Angle",
                                                                    u"ENVIRONMENT")
        self.oa_solar_ang_handle = api.exchange.get_variable_handle(state_argument, u"Site Solar Hour Angle",
                                                                    u"ENVIRONMENT")

        self.zone_window_Heat_loss_Rate_handle_2001 = api.exchange.get_variable_handle(state_argument, u"Zone Windows Total Heat Loss Rate",
                                                                                       'Thermal Zone 1')
        self.zone_window_Heat_loss_Rate_handle_2002 = api.exchange.get_variable_handle(state_argument, u"Zone Windows Total Heat Loss Rate",
                                                                                       'Thermal Zone 2')
        self.zone_window_Heat_loss_Rate_handle_2003 = api.exchange.get_variable_handle(state_argument, u"Zone Windows Total Heat Loss Rate",
                                                                                       'Thermal Zone 3')
        self.zone_window_Heat_loss_Rate_handle_2004 = api.exchange.get_variable_handle(state_argument, u"Zone Windows Total Heat Loss Rate",
                                                                                       'Thermal Zone 4')
        self.zone_window_Heat_loss_Rate_handle_2005 = api.exchange.get_variable_handle(state_argument, u"Zone Windows Total Heat Loss Rate",
                                                                                       'Thermal Zone 5')
        self.zone_window_Heat_loss_Rate_handle_2006 = api.exchange.get_variable_handle(state_argument,
                                                                                       u"Zone Windows Total Heat Loss Rate",
                                                                                       'Thermal Zone 6')

        self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2001 = api.exchange.get_variable_handle(state_argument, u"Zone Windows Total Heat Gain Rate",
                                                                                           'Thermal Zone 1')
        self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2002 = api.exchange.get_variable_handle(state_argument, u"Zone Windows Total Heat Gain Rate",
                                                                                           'Thermal Zone 2')
        self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2003 = api.exchange.get_variable_handle(state_argument, u"Zone Windows Total Heat Gain Rate",
                                                                                           'Thermal Zone 3')
        self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2004 = api.exchange.get_variable_handle(state_argument, u"Zone Windows Total Heat Gain Rate",
                                                                                           'Thermal Zone 4')
        self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2005 = api.exchange.get_variable_handle(state_argument, u"Zone Windows Total Heat Gain Rate",
                                                                                           'Thermal Zone 5')
        self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2006 = api.exchange.get_variable_handle(state_argument, u"Zone Windows Total Heat Gain Rate",
                                                                                           'Thermal Zone 6')



        self.zone_humd_handle_2001 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity",
                                                                      'Thermal Zone 1')
        self.zone_humd_handle_2002 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity",
                                                                      'Thermal Zone 2')
        self.zone_humd_handle_2003 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity",
                                                                      'Thermal Zone 3')
        self.zone_humd_handle_2004 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity",
                                                                      'Thermal Zone 4')
        self.zone_humd_handle_2005 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity",
                                                                      'Thermal Zone 5')
        self.zone_humd_handle_2006 = api.exchange.get_variable_handle(state_argument, "Zone Air Relative Humidity",
                                                                      'Thermal Zone 6')




        self.zone_ventmass_handle_2001 = api.exchange.get_variable_handle(state_argument,
                                                                          "Zone Mechanical Ventilation Mass",
                                                                          'Thermal Zone 1')
        self.zone_ventmass_handle_2002 = api.exchange.get_variable_handle(state_argument,
                                                                          "Zone Mechanical Ventilation Mass",
                                                                          'Thermal Zone 2')
        self.zone_ventmass_handle_2003 = api.exchange.get_variable_handle(state_argument,
                                                                          "Zone Mechanical Ventilation Mass",
                                                                          'Thermal Zone 3')
        self.zone_ventmass_handle_2004 = api.exchange.get_variable_handle(state_argument,
                                                                          "Zone Mechanical Ventilation Mass",
                                                                          'Thermal Zone 4')
        self.zone_ventmass_handle_2005 = api.exchange.get_variable_handle(state_argument,
                                                                          "Zone Mechanical Ventilation Mass",
                                                                          'Thermal Zone 5')
        self.zone_ventmass_handle_2006 = api.exchange.get_variable_handle(state_argument,
                                                                          "Zone Mechanical Ventilation Mass",
                                                                          'Thermal Zone 6')


        self.zone_temp_handle_2001 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature",
                                                                      'Thermal Zone 1')
        self.zone_temp_handle_2002 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature",
                                                                      'Thermal Zone 2')
        self.zone_temp_handle_2003 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature",
                                                                      'Thermal Zone 3')
        self.zone_temp_handle_2004 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature",
                                                                      'Thermal Zone 4')
        self.zone_temp_handle_2005 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature",
                                                                      'Thermal Zone 5')
        self.zone_temp_handle_2006 = api.exchange.get_variable_handle(state_argument, "Zone Air Temperature",
                                                                      'Thermal Zone 6')



    def get_actuator_handle_(self, api_object, state_argument):
        api = api_object
        self.hvac_htg_2001_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control',
                                                                     'Heating Setpoint', 'Thermal Zone 1')
        self.hvac_clg_2001_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control',
                                                                     'Cooling Setpoint', 'Thermal Zone 1')
        self.hvac_htg_2002_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control',
                                                                     'Heating Setpoint', 'Thermal Zone 2')
        self.hvac_clg_2002_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control',
                                                                     'Cooling Setpoint', 'Thermal Zone 2')
        self.hvac_htg_2003_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control',
                                                                     'Heating Setpoint', 'Thermal Zone 3')
        self.hvac_clg_2003_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control',
                                                                     'Cooling Setpoint', 'Thermal Zone 3')
        self.hvac_htg_2004_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control',
                                                                     'Heating Setpoint', 'Thermal Zone 4')
        self.hvac_clg_2004_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control',
                                                                     'Cooling Setpoint', 'Thermal Zone 4')
        self.hvac_htg_2005_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control',
                                                                     'Heating Setpoint', 'Thermal Zone 5')
        self.hvac_clg_2005_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control',
                                                                     'Cooling Setpoint', 'Thermal Zone 5')
        self.hvac_htg_2006_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control',
                                                                     'Heating Setpoint', 'Thermal Zone 6')
        self.hvac_clg_2006_handle = api.exchange.get_actuator_handle(state_argument, 'Zone Temperature Control',
                                                                     'Cooling Setpoint', 'Thermal Zone 6')


    def get_meter_handle_(self, api_object, state_argument):
        api = api_object
        # self.E_Facility_handle = api.exchange.get_meter_handle(state_argument, 'Facility:Electricity')
        self.E_Light_handle = api.exchange.get_meter_handle(state_argument, 'InteriorLights:Electricity')
        self.E_HVAC_handle = api.exchange.get_meter_handle(state_argument, 'Electricity:HVAC')
        self.E_Heating_handle = api.exchange.get_meter_handle(state_argument, 'Heating:Electricity')
        self.E_Cooling_handle = api.exchange.get_meter_handle(state_argument, 'Cooling:Electricity')

    def get_construct_handle_(self, api_object, state_argument):
        api = api_object
        self.ec_device00_handle = api.exchange.get_construction_handle(state_argument, 'SimpleECWindow0')
        self.ec_device01_handle = api.exchange.get_construction_handle(state_argument, 'SimpleECWindow1')
        self.ec_device02_handle = api.exchange.get_construction_handle(state_argument, 'SimpleECWindow2')
        self.ec_device03_handle = api.exchange.get_construction_handle(state_argument, 'SimpleECWindow3')
        self.ec_device04_handle = api.exchange.get_construction_handle(state_argument, 'SimpleECWindow4')
        self.ec_device05_handle = api.exchange.get_construction_handle(state_argument, 'SimpleECWindow5')


        self.zone1_window1_construct_handle = api.exchange.get_actuator_handle(state_argument,
                                                                               "Surface",
                                                                               "Construction State",
                                                                               "Zone1:Window1")
        self.zone2_window1_construct_handle = api.exchange.get_actuator_handle(state_argument,
                                                                               "Surface",
                                                                               "Construction State",
                                                                               "Zone2:Window1")
        self.zone3_window1_construct_handle = api.exchange.get_actuator_handle(state_argument,
                                                                               "Surface",
                                                                               "Construction State",
                                                                               "Zone3:Window1")
        self.zone4_window1_construct_handle = api.exchange.get_actuator_handle(state_argument,
                                                                               "Surface",
                                                                               "Construction State",
                                                                               "Zone4:Window1")
        self.zone4_window2_construct_handle = api.exchange.get_actuator_handle(state_argument,
                                                                               "Surface",
                                                                               "Construction State",
                                                                               "Zone4:Window2")
        self.zone5_window1_construct_handle = api.exchange.get_actuator_handle(state_argument,
                                                                               "Surface",
                                                                               "Construction State",
                                                                               "Zone5:Window1")
        self.zone6_window1_construct_handle = api.exchange.get_actuator_handle(state_argument,
                                                                               "Surface",
                                                                               "Construction State",
                                                                               "Zone6:Window1")



    def get_current_time_stamp(self, api, state_argument):
        year = api.exchange.year(state_argument)
        month = api.exchange.month(state_argument)
        day = api.exchange.day_of_month(state_argument)
        hour = api.exchange.hour(state_argument)
        minute = api.exchange.minutes(state_argument)
        current_time = api.exchange.current_time(state_argument)
        actual_date_time = api.exchange.actual_date_time(state_argument)
        actual_time = api.exchange.actual_time(state_argument)
        time_step = api.exchange.zone_time_step_number(state_argument)
        self.current_time_stamp = {
            'year': year,
            'month': month,
            'day': day,
            'hour': hour,
            'minute': minute,
            'current_time': current_time,
            'actual_date_time': actual_date_time,
            'actual_time': actual_time,
            'time_step': time_step
        }
        return self.current_time_stamp

    def get_temperature_data(self, api_object, state_argument):
        api = api_object
        oa_temp = api.exchange.get_variable_value(state_argument, self.oa_temp_handle)
        oa_solar_azi = api.exchange.get_variable_value(state_argument, self.oa_solar_azi_handle)
        oa_solar_alt = api.exchange.get_variable_value(state_argument, self.oa_solar_alt_handle)
        oa_solar_ang = api.exchange.get_variable_value(state_argument, self.oa_solar_ang_handle)

        zone_window_Heat_loss_Rate_2001 = api.exchange.get_variable_value(state_argument,
                                                                          self.zone_window_Heat_loss_Rate_handle_2001)
        zone_window_Heat_loss_Rate_2002 = api.exchange.get_variable_value(state_argument,
                                                                          self.zone_window_Heat_loss_Rate_handle_2002)
        zone_window_Heat_loss_Rate_2003 = api.exchange.get_variable_value(state_argument,
                                                                          self.zone_window_Heat_loss_Rate_handle_2003)
        zone_window_Heat_loss_Rate_2004 = api.exchange.get_variable_value(state_argument,
                                                                          self.zone_window_Heat_loss_Rate_handle_2004)
        zone_window_Heat_loss_Rate_2005 = api.exchange.get_variable_value(state_argument,
                                                                          self.zone_window_Heat_loss_Rate_handle_2005)
        zone_window_Heat_loss_Rate_2006 = api.exchange.get_variable_value(state_argument,
                                                                          self.zone_window_Heat_loss_Rate_handle_2006)



        Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2001 = api.exchange.get_variable_value(state_argument,
                                                                              self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2001)
        Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2002 = api.exchange.get_variable_value(state_argument,
                                                                              self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2002)
        Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2003 = api.exchange.get_variable_value(state_argument,
                                                                              self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2003)
        Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2004 = api.exchange.get_variable_value(state_argument,
                                                                              self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2004)
        Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2005 = api.exchange.get_variable_value(state_argument,
                                                                              self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2005)
        Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2006 = api.exchange.get_variable_value(state_argument,
                                                                              self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2006)



        zone_temp_2001 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2001)
        zone_temp_2002 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2002)
        zone_temp_2003 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2003)
        zone_temp_2004 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2004)
        zone_temp_2005 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2005)
        zone_temp_2006 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2006)

        hvac_htg_2001 = api.exchange.get_actuator_value(state_argument, self.hvac_htg_2001_handle)
        hvac_clg_2001 = api.exchange.get_actuator_value(state_argument, self.hvac_clg_2001_handle)
        hvac_htg_2002 = api.exchange.get_actuator_value(state_argument, self.hvac_htg_2002_handle)
        hvac_clg_2002 = api.exchange.get_actuator_value(state_argument, self.hvac_clg_2002_handle)
        hvac_htg_2003 = api.exchange.get_actuator_value(state_argument, self.hvac_htg_2003_handle)
        hvac_clg_2003 = api.exchange.get_actuator_value(state_argument, self.hvac_clg_2003_handle)
        hvac_htg_2004 = api.exchange.get_actuator_value(state_argument, self.hvac_htg_2004_handle)
        hvac_clg_2004 = api.exchange.get_actuator_value(state_argument, self.hvac_clg_2004_handle)
        hvac_htg_2005 = api.exchange.get_actuator_value(state_argument, self.hvac_htg_2005_handle)
        hvac_clg_2005 = api.exchange.get_actuator_value(state_argument, self.hvac_clg_2005_handle)
        hvac_htg_2006 = api.exchange.get_actuator_value(state_argument, self.hvac_htg_2006_handle)
        hvac_clg_2006 = api.exchange.get_actuator_value(state_argument, self.hvac_clg_2006_handle)


        win_2001 = api.exchange.get_actuator_value(state_argument, self.zone1_window1_construct_handle)
        win_2002 = api.exchange.get_actuator_value(state_argument, self.zone2_window1_construct_handle)
        win_2003 = api.exchange.get_actuator_value(state_argument, self.zone3_window1_construct_handle)
        win_2004 = api.exchange.get_actuator_value(state_argument, self.zone4_window1_construct_handle)
        win_2005 = api.exchange.get_actuator_value(state_argument, self.zone5_window1_construct_handle)
        win_2006 = api.exchange.get_actuator_value(state_argument, self.zone6_window1_construct_handle)



        zone_humd_2001 = api.exchange.get_variable_value(state_argument, self.zone_humd_handle_2001)
        zone_humd_2002 = api.exchange.get_variable_value(state_argument, self.zone_humd_handle_2002)
        zone_humd_2003 = api.exchange.get_variable_value(state_argument, self.zone_humd_handle_2003)
        zone_humd_2004 = api.exchange.get_variable_value(state_argument, self.zone_humd_handle_2004)
        zone_humd_2005 = api.exchange.get_variable_value(state_argument, self.zone_humd_handle_2005)
        zone_humd_2006 = api.exchange.get_variable_value(state_argument, self.zone_humd_handle_2006)


        zone_ventmass_2001 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2001)
        zone_ventmass_2002 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2002)
        zone_ventmass_2003 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2003)
        zone_ventmass_2004 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2004)
        zone_ventmass_2005 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2005)
        zone_ventmass_2006 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2006)


        # 返回变量的字典
        return {
            'oa_temp': oa_temp,
            'oa_solar_azi': oa_solar_azi,
            'oa_solar_alt': oa_solar_alt,
            'oa_solar_ang': oa_solar_ang,

            'zone_window_Heat_loss_Rate_2001': zone_window_Heat_loss_Rate_2001,
            'zone_window_Heat_loss_Rate_2002': zone_window_Heat_loss_Rate_2002,
            'zone_window_Heat_loss_Rate_2003': zone_window_Heat_loss_Rate_2003,
            'zone_window_Heat_loss_Rate_2004': zone_window_Heat_loss_Rate_2004,
            'zone_window_Heat_loss_Rate_2005': zone_window_Heat_loss_Rate_2005,
            'zone_window_Heat_loss_Rate_2006': zone_window_Heat_loss_Rate_2006,

            'Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2001': Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2001,
            'Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2002': Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2002,
            'Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2003': Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2003,
            'Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2004': Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2004,
            'Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2005': Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2005,
            'Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2006': Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2006,


            'zone_temp_2001': zone_temp_2001,
            'zone_temp_2002': zone_temp_2002,
            'zone_temp_2003': zone_temp_2003,
            'zone_temp_2004': zone_temp_2004,
            'zone_temp_2005': zone_temp_2005,
            'zone_temp_2006': zone_temp_2006,


            'hvac_htg_2001': hvac_htg_2001,
            'hvac_clg_2001': hvac_clg_2001,
            'hvac_htg_2002': hvac_htg_2002,
            'hvac_clg_2002': hvac_clg_2002,
            'hvac_htg_2003': hvac_htg_2003,
            'hvac_clg_2003': hvac_clg_2003,
            'hvac_htg_2004': hvac_htg_2004,
            'hvac_clg_2004': hvac_clg_2004,
            'hvac_htg_2005': hvac_htg_2005,
            'hvac_clg_2005': hvac_clg_2005,
            'hvac_htg_2006': hvac_htg_2006,
            'hvac_clg_2006': hvac_clg_2006,

            'win_2001': win_2001,
            'win_2002': win_2002,
            'win_2003': win_2003,
            'win_2004': win_2004,
            'win_2005': win_2005,
            'win_2006': win_2006,

            'zone_humd_2001': zone_humd_2001,
            'zone_humd_2002': zone_humd_2002,
            'zone_humd_2003': zone_humd_2003,
            'zone_humd_2004': zone_humd_2004,
            'zone_humd_2005': zone_humd_2005,
            'zone_humd_2006': zone_humd_2006,
            'zone_ventmass_2001': zone_ventmass_2001,
            'zone_ventmass_2002': zone_ventmass_2002,
            'zone_ventmass_2003': zone_ventmass_2003,
            'zone_ventmass_2004': zone_ventmass_2004,
            'zone_ventmass_2005': zone_ventmass_2005,
            'zone_ventmass_2006': zone_ventmass_2006,
        }

    def get_and_store_temperature_data(self, api_object, state_argument):
        api = api_object
        oa_temp = api.exchange.get_variable_value(state_argument, self.oa_temp_handle)
        oa_solar_azi = api.exchange.get_variable_value(state_argument, self.oa_solar_azi_handle)
        oa_solar_alt = api.exchange.get_variable_value(state_argument, self.oa_solar_alt_handle)
        oa_solar_ang = api.exchange.get_variable_value(state_argument, self.oa_solar_ang_handle)

        zone_window_Heat_loss_Rate_2001 = api.exchange.get_variable_value(state_argument,
                                                                          self.zone_window_Heat_loss_Rate_handle_2001)
        zone_window_Heat_loss_Rate_2002 = api.exchange.get_variable_value(state_argument,
                                                                          self.zone_window_Heat_loss_Rate_handle_2002)
        zone_window_Heat_loss_Rate_2003 = api.exchange.get_variable_value(state_argument,
                                                                          self.zone_window_Heat_loss_Rate_handle_2003)
        zone_window_Heat_loss_Rate_2004 = api.exchange.get_variable_value(state_argument,
                                                                          self.zone_window_Heat_loss_Rate_handle_2004)
        zone_window_Heat_loss_Rate_2005 = api.exchange.get_variable_value(state_argument,
                                                                          self.zone_window_Heat_loss_Rate_handle_2005)
        zone_window_Heat_loss_Rate_2006 = api.exchange.get_variable_value(state_argument,
                                                                          self.zone_window_Heat_loss_Rate_handle_2006)

        Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2001 = api.exchange.get_variable_value(state_argument,
                                                                              self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2001)
        Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2002 = api.exchange.get_variable_value(state_argument,
                                                                              self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2002)
        Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2003 = api.exchange.get_variable_value(state_argument,
                                                                              self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2003)
        Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2004 = api.exchange.get_variable_value(state_argument,
                                                                              self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2004)
        Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2005 = api.exchange.get_variable_value(state_argument,
                                                                              self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2005)
        Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2006 = api.exchange.get_variable_value(state_argument,
                                                                              self.Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_handle_2006)

        zone_temp_2001 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2001)
        zone_temp_2002 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2002)
        zone_temp_2003 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2003)
        zone_temp_2004 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2004)
        zone_temp_2005 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2005)
        zone_temp_2006 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2006)

        hvac_htg_2001 = api.exchange.get_actuator_value(state_argument, self.hvac_htg_2001_handle)
        hvac_clg_2001 = api.exchange.get_actuator_value(state_argument, self.hvac_clg_2001_handle)
        hvac_htg_2002 = api.exchange.get_actuator_value(state_argument, self.hvac_htg_2002_handle)
        hvac_clg_2002 = api.exchange.get_actuator_value(state_argument, self.hvac_clg_2002_handle)
        hvac_htg_2003 = api.exchange.get_actuator_value(state_argument, self.hvac_htg_2003_handle)
        hvac_clg_2003 = api.exchange.get_actuator_value(state_argument, self.hvac_clg_2003_handle)
        hvac_htg_2004 = api.exchange.get_actuator_value(state_argument, self.hvac_htg_2004_handle)
        hvac_clg_2004 = api.exchange.get_actuator_value(state_argument, self.hvac_clg_2004_handle)
        hvac_htg_2005 = api.exchange.get_actuator_value(state_argument, self.hvac_htg_2005_handle)
        hvac_clg_2005 = api.exchange.get_actuator_value(state_argument, self.hvac_clg_2005_handle)
        hvac_htg_2006 = api.exchange.get_actuator_value(state_argument, self.hvac_htg_2006_handle)
        hvac_clg_2006 = api.exchange.get_actuator_value(state_argument, self.hvac_clg_2006_handle)

        win_2001 = api.exchange.get_actuator_value(state_argument, self.zone1_window1_construct_handle)
        win_2002 = api.exchange.get_actuator_value(state_argument, self.zone2_window1_construct_handle)
        win_2003 = api.exchange.get_actuator_value(state_argument, self.zone3_window1_construct_handle)
        win_2004 = api.exchange.get_actuator_value(state_argument, self.zone4_window1_construct_handle)
        win_2005 = api.exchange.get_actuator_value(state_argument, self.zone5_window1_construct_handle)
        win_2006 = api.exchange.get_actuator_value(state_argument, self.zone6_window1_construct_handle)




        zone_humd_2001 = api.exchange.get_variable_value(state_argument, self.zone_humd_handle_2001)
        zone_humd_2002 = api.exchange.get_variable_value(state_argument, self.zone_humd_handle_2002)
        zone_humd_2003 = api.exchange.get_variable_value(state_argument, self.zone_humd_handle_2003)
        zone_humd_2004 = api.exchange.get_variable_value(state_argument, self.zone_humd_handle_2004)
        zone_humd_2005 = api.exchange.get_variable_value(state_argument, self.zone_humd_handle_2005)
        zone_humd_2006 = api.exchange.get_variable_value(state_argument, self.zone_humd_handle_2006)


        zone_ventmass_2001 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2001)
        zone_ventmass_2002 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2002)
        zone_ventmass_2003 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2003)
        zone_ventmass_2004 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2004)
        zone_ventmass_2005 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2005)
        zone_ventmass_2006 = api.exchange.get_variable_value(state_argument, self.zone_temp_handle_2006)


        self.y_temp.append(oa_temp)
        self.y_solar.append([oa_solar_azi, oa_solar_alt, oa_solar_ang])
        self.y_zone_humd.append([zone_humd_2001, zone_humd_2002, zone_humd_2003,
                                 zone_humd_2004, zone_humd_2005, zone_humd_2006

                                 ])


        self.y_zone_ventmass.append([zone_ventmass_2001, zone_ventmass_2002, zone_ventmass_2003,
                                     zone_ventmass_2004, zone_ventmass_2005, zone_ventmass_2006

                                     ])

        self.y_outdoor.append(oa_temp)

        self.y_zone_temp_2001.append(zone_temp_2001)
        self.y_zone_temp_2002.append(zone_temp_2002)
        self.y_zone_temp_2003.append(zone_temp_2003)
        self.y_zone_temp_2004.append(zone_temp_2004)
        self.y_zone_temp_2005.append(zone_temp_2005)
        self.y_zone_temp_2006.append(zone_temp_2006)


        self.y_zone_window_Heat_loss_Rate_2001.append(zone_window_Heat_loss_Rate_2001)
        self.y_zone_window_Heat_loss_Rate_2002.append(zone_window_Heat_loss_Rate_2002)
        self.y_zone_window_Heat_loss_Rate_2003.append(zone_window_Heat_loss_Rate_2003)
        self.y_zone_window_Heat_loss_Rate_2004.append(zone_window_Heat_loss_Rate_2004)
        self.y_zone_window_Heat_loss_Rate_2005.append(zone_window_Heat_loss_Rate_2005)
        self.y_zone_window_Heat_loss_Rate_2006.append(zone_window_Heat_loss_Rate_2006)

        self.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2001.append(Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2001)
        self.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2002.append(Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2002)
        self.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2003.append(Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2003)
        self.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2004.append(Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2004)
        self.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2005.append(Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2005)
        self.y_Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2006.append(Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_2006)

        self.hvac_htg_2001.append(hvac_htg_2001)
        self.hvac_clg_2001.append(hvac_clg_2001)
        self.hvac_htg_2002.append(hvac_htg_2002)
        self.hvac_clg_2002.append(hvac_clg_2002)
        self.hvac_htg_2003.append(hvac_htg_2003)
        self.hvac_clg_2003.append(hvac_clg_2003)
        self.hvac_htg_2004.append(hvac_htg_2004)
        self.hvac_clg_2004.append(hvac_clg_2004)
        self.hvac_htg_2005.append(hvac_htg_2005)
        self.hvac_clg_2005.append(hvac_clg_2005)
        self.hvac_htg_2006.append(hvac_htg_2006)
        self.hvac_clg_2006.append(hvac_clg_2006)

        self.win_2001.append(win_2001)
        self.win_2002.append(win_2002)
        self.win_2003.append(win_2003)
        self.win_2004.append(win_2004)
        self.win_2005.append(win_2005)
        self.win_2006.append(win_2006)

        T_list = (np.array([zone_temp_2001,
                            zone_temp_2002,
                            zone_temp_2003,
                            zone_temp_2004,
                            zone_temp_2005,
                            zone_temp_2006,
                            ]))

        self.y_zone_temp.append(T_list)

        T_mean = np.mean(T_list)

        self.T_mean_list.append(T_mean)
        self.T_diff.append(np.max(T_list) - np.min(T_list))
        self.T_var.append(np.var(T_list))

    def get_and_store_meter_data(self, api_object, state_argument):
        api = api_object
        self.E_HVAC.append(api.exchange.get_meter_value(state_argument, self.E_HVAC_handle))
        self.E_Heating.append(api.exchange.get_meter_value(state_argument, self.E_Heating_handle))
        self.E_Cooling.append(api.exchange.get_meter_value(state_argument, self.E_Cooling_handle))
        self.E_HVAC_all.append(api.exchange.get_meter_value(state_argument, self.E_HVAC_handle))

    def get_datetime_obj_and_update_time(self, api_object, state_argument):
        api = api_object
        current_time_stamp = self.get_current_time_stamp(api, state_argument)
        month = current_time_stamp["month"]
        day = current_time_stamp["day"]
        hour = current_time_stamp["hour"]
        minute = current_time_stamp["minute"]
        current_time = current_time_stamp["current_time"]
        actual_date_time = current_time_stamp["actual_date_time"]
        actual_time = current_time_stamp["actual_time"]
        time_step = current_time_stamp["time_step"]
        year = 2023
        self.years.append(year)
        self.months.append(month)
        self.days.append(day)
        self.hours.append(hour)
        self.minutes.append(minute)

        self.current_times.append(current_time)
        self.actual_date_times.append(actual_date_time)
        self.actual_times.append(actual_time)

        timedelta = datetime.timedelta()
        if hour >= 24.0:
            hour = 23.0
            timedelta += datetime.timedelta(hours=1)
        if minute >= 60.0:
            minute = 59
            timedelta += datetime.timedelta(minutes=1)

        dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)
        dt += timedelta
        self.x.append(dt)
        self.time_line.append(dt)

        if dt.weekday() > 4:
            # print 'Given date is weekend.'
            self.weekday.append(dt.weekday())
            self.isweekday.append(0)
            self.isweekend.append(1)
        else:
            # print 'Given data is weekday.'
            self.weekday.append(dt.weekday())
            self.isweekday.append(1)
            self.isweekend.append(0)

        self.work_time.append(self.isweekday[-1] * self.sun_is_up[-1])

        return dt

    def get_current_zone_temp(self):
        if self.zone_temp_handle_2001 is not None:
            return [self.y_zone_temp_2001[-1],
                    self.y_zone_temp_2002[-1],
                    self.y_zone_temp_2003[-1],
                    self.y_zone_temp_2004[-1],
                    self.y_zone_temp_2005[-1],
                    self.y_zone_temp_2006[-1],
                    self.y_outdoor[-1]]
        return None
