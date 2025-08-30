import numpy as np


def temp_reward_get(indoor_temp, t_lower, t_upper, t_factor):
    reward = np.abs(indoor_temp - (t_lower+t_upper)/2)
    reward = (reward ** 2) * t_factor
    return -reward


def calculate_reward(comfort_temp,is_worktime, E1,T_values,
                     E_factor_day, T_factor_day,
                     args):
    """
    计算奖励值，根据能耗、温度和动作平滑性。

    Parameters:
        comfort_temp:热舒适度上下限
        is_worktime (bool): 是否为工作时间
        E1 (float): 当前能耗值
        T_values (list): 各房间温度值的列表，例如 [T_11, T_21, T_31, T_41, T_51]
        Qmix_action_list (list): 当前和上一步的HVAC动作列表
        E_factor_day (float): 工作时间能耗因子
        T_factor_day (float): 工作时间温度因子
        E_factor_night (float): 非工作时间能耗因子
        T_factor_night (float): 非工作时间温度因子


    Returns:
        dict: 计算出的奖励值字典
    """


    # 设置能耗和温度因子
    E_factor = E_factor_day
    work_flag = 1 if is_worktime else 0
    # 1. 能耗奖励
    reward_avg_E = - (E1 * E_factor / 12)
    E_beta = 5


    T_upper = comfort_temp['T_upper'] + 2
    T_lower = comfort_temp['T_lower'] - 2
    # 2. 温度奖励
    if args.alg in {'mappo' , 'madqn'}:
        reward_per = []
        reward_T_all = 0
        for T in T_values:
            if work_flag:
                if T_lower < T < T_upper:
                    reward_T = 1
                else:
                    reward_T = temp_reward_get( T , T_lower , T_upper , T_factor_day )  #-(T - (23 + 25) / 2) ** 2 * T_factor
            else :
                reward_T = 0
            reward_T_all += reward_T
            reward_per.append(reward_T * (10 - E_beta )  + reward_avg_E * E_beta)
        # 4. 综合奖励

        reward = reward_T_all * (10 - E_beta ) + reward_avg_E * E_beta
        rewards = {"reward": reward, "reward_T": reward_T_all * (10 - E_beta ), "reward_E": reward_avg_E * E_beta,"reward_per": reward_per}
        return rewards

    elif args.alg in {'qmix' , 'vdn', 'qmixr2', 'qmixr1'}:
        reward_T = 0
        for T in T_values:
            if work_flag:
                if T_lower < T < T_upper:
                    reward_T += 1
                else:
                    reward_T += temp_reward_get(T, T_lower, T_upper,
                                                T_factor_day)  # -(T - (23 + 25) / 2) ** 2 * T_factor
            else:
                reward_T += 0
        # 4. 综合奖励

        reward = reward_T * (10 - E_beta) + reward_avg_E * E_beta
        rewards = {"reward": reward, "reward_T": reward_T * (10 - E_beta), "reward_E": reward_avg_E * E_beta}
        return rewards


