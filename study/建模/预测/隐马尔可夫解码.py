def viterbi_algorithm(ob, states, s_p, t_p, e_p):
    # 初始化动态规划表格，存储每个时间步t处于每个隐状态的最大概率和对应的前一个状态
    dp_table = [{}]
    # 初始化第一个时间步的动态规划表格
    for state in states:
        dp_table[0][state] = {"p": s_p[state] * e_p[state][ob[0]], "p_state": None}
    # 递推计算后续时间步的动态规划表格
    for t in range(1, len(ob)):
        dp_table.append({})
        for c_state in states:
            max_prob = max(
                dp_table[t - 1][prev_state]["p"] * t_p[prev_state][c_state] *
                e_p[c_state][ob[t]] for prev_state in states
            )
            p_state = max(states, key=lambda p_state:
                      dp_table[t-1][p_state]["p"] * t_p[p_state][c_state] * e_p[c_state][ob[t]])
            dp_table[t][c_state] = {"p": max_prob, "p_state": p_state}
    # 回溯找出最可能的隐藏状态序列
    path = []
    max_final_prob = max(dp_table[-1].values(), key=lambda x: x["p"])
    max_final_state = max_final_prob["p_state"]
    path.append(max_final_state)
    for t in range(len(dp_table) - 2, -1, -1):
        max_final_state = dp_table[t + 1][max_final_state]["p_state"]
        path.insert(0, max_final_state)
    return path

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
    # 使用维特比算法解码得到最可能的隐藏状态序列
    path = viterbi_algorithm(observations, states, start_p, transition_p, emission_p)
    print("最可能的隐藏状态序列为:", path)