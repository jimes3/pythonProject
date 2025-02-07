def forward_algorithm(obs, states, s_p, t_p, e_p):
    # 初始化前向概率矩阵，存储每个时间步t处于每个隐状态的概率
    f_p = [{}]
    # 初始化第一个时间步的前向概率
    for state in states:
        f_p[0][state] = s_p[state] * e_p[state][obs[0]]
    # 递推计算后续时间步的前向概率
    for t in range(1, len(obs)):
        f_p.append({})
        for c_state in states:
            # 计算当前时间步t处于current_state的前向概率
            f_p[t][c_state] = sum(
                f_p[t - 1][p_state] * t_p[p_state][c_state] *
                e_p[c_state][obs[t]] for p_state in states
            )
    # 计算最终观测序列的概率，即最后一个时间步的前向概率之和
    prob = sum(f_p[-1].values())
    return prob

# 示例用法
if __name__ == "__main__":
    # 观测符号集合（心情）
    observations = ["开心", "郁闷", "郁闷", "开心"]
    # 隐状态集合（天气）
    states = ["晴天", "雨天"]
    # 初始状态概率
    start_p = {"晴天": 0.6, "雨天": 0.4}
    # 状态转移概率
    transition_p = {
        "晴天": {"晴天": 0.7, "雨天": 0.3},
        "雨天": {"晴天": 0.4, "雨天": 0.6}
    }
    # 发射概率（心情由天气生成的概率）
    emission_p = {
        "晴天": {"开心": 0.8, "郁闷": 0.2},
        "雨天": {"开心": 0.3, "郁闷": 0.7}
    }
    # 使用前向算法计算观测序列的概率
    prob = forward_algorithm(observations, states, start_p, transition_p, emission_p)
    print("观测序列的概率为:", prob)
