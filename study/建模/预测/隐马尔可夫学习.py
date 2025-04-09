import random

def forward_algorithm(observations, states, start_prob, transition_prob, emission_prob):
    T = len(observations)  # 观测序列的长度
    # 初始化前向概率矩阵
    forward_prob = [{}]
    # 初始化第一个时间步的前向概率
    for state in states:
        forward_prob[0][state] = start_prob[state] * emission_prob[state][observations[0]]
    # 递推计算后续时间步的前向概率
    for t in range(1, T):
        forward_prob.append({})
        for current_state in states:
            forward_prob[t][current_state] = sum(
                forward_prob[t - 1][prev_state] * transition_prob[prev_state][current_state] *
                emission_prob[current_state][observations[t]] for prev_state in states
            )
    return forward_prob

def backward_algorithm(observations, states, transition_prob, emission_prob):
    T = len(observations)  # 观测序列的长度
    # 初始化后向概率矩阵
    backward_prob = [{} for _ in range(T)]
    # 初始化最后一个时间步的后向概率
    for state in states:
        backward_prob[T - 1][state] = 1.0
    # 递推计算前面时间步的后向概率
    for t in range(T - 2, -1, -1):
        for current_state in states:
            backward_prob[t][current_state] = sum(
                backward_prob[t + 1][next_state] * transition_prob[current_state][next_state] *
                emission_prob[next_state][observations[t + 1]] for next_state in states
            )
    return backward_prob
def initialize_random_probabilities(states, observations):
    # 随机初始化初始状态概率
    start_prob = {state: random.random() for state in states}
    start_prob_sum = sum(start_prob.values())
    start_prob = {state: prob / start_prob_sum for state, prob in start_prob.items()}
    # 随机初始化状态转移概率
    transition_prob = {state: {state: random.random() for state in states} for state in states}
    for state in states:
        transition_prob_sum = sum(transition_prob[state].values())
        transition_prob[state] = {next_state: prob / transition_prob_sum for next_state, prob in transition_prob[state].items()}
    # 随机初始化发射概率
    emission_prob = {state: {observation: random.random() for observation in observations} for state in states}
    for state in states:
        emission_prob_sum = sum(emission_prob[state].values())
        emission_prob[state] = {observation: prob / emission_prob_sum for observation, prob in emission_prob[state].items()}
    return start_prob, transition_prob, emission_prob
def baum_welch_learning(observations, states, max_iterations=1000):
    num_observations = len(observations)
    # 初始化随机的初始状态概率，状态转移概率和发射概率
    start_prob, transition_prob, emission_prob = initialize_random_probabilities(states, observations)
    # 迭代更新模型参数
    for iteration in range(max_iterations):
        # 在E步骤中，计算前向和后向概率
        forward_prob = forward_algorithm(observations, states, start_prob, transition_prob, emission_prob)
        backward_prob = backward_algorithm(observations, states, transition_prob, emission_prob)
        # 在M步骤中，根据前向和后向概率估计模型参数
        new_start_prob = {state: forward_prob[0][state] * backward_prob[0][state] for state in states}
        new_start_prob_sum = sum(new_start_prob.values())
        start_prob = {state: prob / new_start_prob_sum for state, prob in new_start_prob.items()}
        new_transition_prob = {state: {next_state: 0.0 for next_state in states} for state in states}
        for t in range(num_observations - 1):
            for state in states:
                for next_state in states:
                    new_transition_prob[state][next_state] += forward_prob[t][state] * transition_prob[state][next_state] * \
                                                              emission_prob[next_state][observations[t + 1]] * backward_prob[t + 1][next_state]
        for state in states:
            new_transition_prob_sum = sum(new_transition_prob[state].values())
            transition_prob[state] = {next_state: prob / new_transition_prob_sum for next_state, prob in new_transition_prob[state].items()}
        new_emission_prob = {state: {observation: 0.0 for observation in observations} for state in states}
        for t in range(num_observations):
            for state in states:
                new_emission_prob[state][observations[t]] += forward_prob[t][state] * backward_prob[t][state]
        for state in states:
            new_emission_prob_sum = sum(new_emission_prob[state].values())
            emission_prob[state] = {observation: prob / new_emission_prob_sum for observation, prob in new_emission_prob[state].items()}
    return start_prob, transition_prob, emission_prob

# 示例用法
if __name__ == "__main__":
    # 观测符号集合（心情）
    observations = ["开心", "郁闷", "郁闷", "开心"]
    # 隐状态集合（天气）
    states = ["晴天", "雨天"]
    # 使用Baum-Welch算法学习隐马尔可夫模型的参数
    start_prob, transition_prob, emission_prob = baum_welch_learning(observations, states)
    print("学习得到的初始状态概率：", start_prob)
    print("学习得到的状态转移概率：", transition_prob)
    print("学习得到的发射概率：", emission_prob)
