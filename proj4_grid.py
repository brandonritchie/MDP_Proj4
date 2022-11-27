from hiive.mdptoolbox import mdp, example
from hiive.mdptoolbox.mdp import ValueIteration
from hiive.mdptoolbox.mdp import PolicyIteration
from hiive.mdptoolbox.mdp import QLearning
from hiivemdptoolbox.hiive.mdptoolbox.example import openai
from gym.envs.toy_text.frozen_lake import generate_random_map
import re
import numpy as np
import gym
import matplotlib.pyplot as plt
#%%

def generate_map_algo(map, discount, algorithm = 'ValueIteration'):
    
    P, R = openai('FrozenLake-v1', desc = map)
    if algorithm == 'ValueIteration':
        algo = ValueIteration(P, R, discount)
        algo.run()
    if algorithm == 'PolicyIteration':
        algo = PolicyIteration(P, R, discount)
        algo.run()
    if algorithm == 'QLearning':
        algo = QLearning(P, R, gamma=discount, alpha=0.1, alpha_decay=0.9, epsilon=1, epsilon_decay=0.9, n_iter=1000000)
        algo.run()

    return(map, algo, P, R)
map_s = generate_random_map(10, p = .9)
map_l = generate_random_map(25, p = .9)
map_small_vi, vi_algo, P_s, R_s = generate_map_algo(map_s, .98, "ValueIteration")
map_small_pi, pi_algo, P_s, R_s = generate_map_algo(map_s, .98, "PolicyIteration")
map_small_q, q_algo, P_s, R_s = generate_map_algo(map_s, .99, "QLearning")

map_l_vi, vi_algo_l, P_l, R_l = generate_map_algo(map_l, .98, "ValueIteration")
map_l_pi, pi_algo_l, P_l, R_l  = generate_map_algo(map_l, .98, "PolicyIteration")
map_l_q, q_algo_l, P_l, R_l  = generate_map_algo(map_l, .99, "QLearning")


'''
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
'''

#%%
# Policy visualization
def map_frozen_lake(map, algo, size, type = 'Policy', algo_name = 'Value Iteration'):
    bigmap = ''
    for i in range(len(map)):
        bigmap = bigmap + map[i]
    hole_pos = [int(i.start()) for i in re.finditer('H', bigmap)]
    cl = []
    p_l = []
    if type == 'Policy':
        pos = 0
        for i in algo.policy:
            if pos == (len(algo.policy) - 1):
                p_l.append('G')
                cl.append(2)
                pos += 1
                continue
            if pos in hole_pos:
                p_l.append('H')
                cl.append(1)
                pos += 1
                continue
            elif i == 0:
                p_l.append('←')
            elif i == 1:
                p_l.append("↓")
            elif i == 2:
                p_l.append('→')
            elif i == 3:
                p_l.append('↑')
            cl.append(0)
            pos += 1
        title = f'Frozen Lake Policy Map - {algo_name}'

    if type == 'Env':
        for i in bigmap:
            if i in ['S', 'G']:
                cl.append(2)
                continue
            if i == 'H':
                cl.append(1)
                continue
            cl.append(0)
        p_l = list(bigmap)
        title = f'Frozen Lake Environment - {algo_name}'

    if type == 'State':
        cl = [round(val,2) for val in algo.V]
        p_l = cl
        title = f'Frozen Lake State-Value Map - {algo_name}'

	# reshape value function
    V_sq = np.reshape(cl, (size,size))
    V_sq2 = np.reshape(p_l, (size,size))

    # plot the state-value function
    fig_s = size // 1.5
    fig = plt.figure(figsize=(fig_s, fig_s))
    ax = fig.add_subplot(111)
    im = ax.imshow(V_sq, cmap='cool')
    for (j,i),label in np.ndenumerate(V_sq2):
        ax.text(i, j, label, ha='center', va='center', fontsize=14)
    plt.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')
    plt.title(title)
    plt.show()

# Small maps all algorithms
map_frozen_lake(map_small_vi, vi_algo, 10, algo_name = "Value Iteration (Low Gamma)")
map_frozen_lake(map_small_vi, vi_algo, 10, type = 'Env', algo_name = "Value Iteration")
map_frozen_lake(map_small_vi, vi_algo, 10, type = 'State', algo_name = "Value Iteration (Low Gamma)")
map_frozen_lake(map_small_pi, pi_algo, 10, algo_name = "Policy Iteration")
map_frozen_lake(map_small_pi, pi_algo, 10, type = 'Env', algo_name = "Policy Iteration")
map_frozen_lake(map_small_pi, pi_algo, 10, type = 'State', algo_name = "Policy Iteration")
map_frozen_lake(map_small_q, q_algo, 10, algo_name = "Q Learning")
map_frozen_lake(map_small_q, q_algo, 10, type = 'Env', algo_name = "Q Learning")
map_frozen_lake(map_small_q, q_algo, 10, type = 'State', algo_name = "Q Learning")

map_frozen_lake(map_l_vi, vi_algo_l, 25, algo_name = "Value Iteration")
map_frozen_lake(map_l_vi, vi_algo_l, 25, type = 'Env', algo_name = "Value Iteration")
map_frozen_lake(map_l_vi, vi_algo_l, 25, type = 'State', algo_name = "Value Iteration")
map_frozen_lake(map_l_pi, pi_algo_l, 25, algo_name = "Policy Iteration")
map_frozen_lake(map_l_pi, pi_algo_l, 25, type = 'Env', algo_name = "Policy Iteration")
map_frozen_lake(map_l_pi, pi_algo_l, 25, type = 'State', algo_name = "Policy Iteration")
map_frozen_lake(map_l_q, q_algo_l, 25, algo_name = "Q Learning")
map_frozen_lake(map_l_q, q_algo_l, 25, type = 'Env', algo_name = "Q Learning")
map_frozen_lake(map_l_q, q_algo_l, 25, type = 'State', algo_name = "Q Learning")

#%%
# Value Iteration Convergence Plot
def val_iter_plot(P, R):
    mean_vs_vi = []
    rewards_vi = []
    time_vi = []
    iterations_vi = []
    mean_vs_pi = []
    rewards_pi = []
    time_pi = []
    iterations_pi = []
    gammas = list(np.arange(0.8,1,0.06))

    for gamma in gammas:
        vi = mdp.ValueIteration(P, R, gamma)
        pi = mdp.PolicyIteration(P, R, gamma, max_iter=20)
        vi.run()
        pi.run()

        iterations_vi.append(list(range(1,len(vi.run_stats)+1)))
        mean_vs_vi.append([val['Mean V'] for val in vi.run_stats])
        rewards_vi.append([val['Reward'] for val in vi.run_stats])
        time_vi.append([val['Time'] for val in vi.run_stats])

        iterations_pi.append(list(range(1,len(pi.run_stats)+1)))
        mean_vs_pi.append([val['Mean V'] for val in pi.run_stats])
        rewards_pi.append([val['Reward'] for val in pi.run_stats])
        time_pi.append([val['Time'] for val in pi.run_stats])

    f, ax = plt.subplots(2,3, figsize=(25,20))

    ax[0][0].set_title("Mean Value Function - Value Iteration", fontsize=30)
    for i in range(len(mean_vs_vi)):
        ax[0][0].plot(iterations_vi[i], mean_vs_vi[i], alpha=1, label="Gamma = {}".format(np.round(gammas[i],3)), linewidth = 5)
    ax[0][0].legend(fontsize = 20)
    ax[0][0].tick_params(axis='x', labelsize=25)
    ax[0][0].tick_params(axis='y', labelsize=25)
    ax[0][0].set_xlabel("# Iterations", fontsize = 20)

    ax[0][1].set_title("Reward - Value Iteration", fontsize=30)
    for i in range(len(rewards_vi)):
        ax[0][1].plot(iterations_vi[i], rewards_vi[i], alpha=1, label="Gamma = {}".format(np.round(gammas[i],3)), linewidth = 5)
    ax[0][1].legend(fontsize = 20)
    ax[0][1].tick_params(axis='x', labelsize=25)
    ax[0][1].tick_params(axis='y', labelsize=25)
    ax[0][1].set_xlabel("# Iterations", fontsize = 20)

    ax[0][2].set_title("Time - Value Iteration", fontsize=30)
    for i in range(len(rewards_vi)):
        ax[0][2].plot(iterations_vi[i], time_vi[i], alpha=1, label="Gamma = {}".format(np.round(gammas[i],3)), linewidth = 5)
    ax[0][2].legend(fontsize = 20)
    ax[0][2].tick_params(axis='x', labelsize=25)
    ax[0][2].tick_params(axis='y', labelsize=25)
    ax[0][2].set_xlabel("# Iterations", fontsize = 20)
    ax[1][0].set_title("Mean Value Function - Policy Iteration", fontsize=30)
    for i in range(len(mean_vs_pi)):
        ax[1][0].plot(iterations_pi[i], mean_vs_pi[i], alpha=1, label="Gamma = {}".format(np.round(gammas[i],3)), linewidth = 5)
    ax[1][0].legend(fontsize = 20)
    ax[1][0].tick_params(axis='x', labelsize=25)
    ax[1][0].tick_params(axis='y', labelsize=25)
    ax[1][0].set_xlabel("# Iterations", fontsize = 20)

    ax[1][1].set_title("Reward - Policy Iteration", fontsize=30)
    for i in range(len(rewards_pi)):
        ax[1][1].plot(iterations_pi[i], rewards_pi[i], alpha=1, label="Gamma = {}".format(np.round(gammas[i],3)), linewidth = 5)
    ax[1][1].legend(fontsize = 20)
    ax[1][1].tick_params(axis='x', labelsize=25)
    ax[1][1].tick_params(axis='y', labelsize=25)
    ax[1][1].set_xlabel("# Iterations", fontsize = 20)

    ax[1][2].set_title("Time - Policy Iteration", fontsize=30)
    for i in range(len(rewards_pi)):
        ax[1][2].plot(iterations_pi[i], time_pi[i], alpha=1, label="Gamma = {}".format(np.round(gammas[i],3)), linewidth = 5)
    ax[1][2].legend(fontsize = 20)
    ax[1][2].tick_params(axis='x', labelsize=25)
    ax[1][2].tick_params(axis='y', labelsize=25)
    ax[1][2].set_xlabel("# Iterations", fontsize = 20)
    plt.plot()

val_iter_plot(P_s, R_s)

val_iter_plot(P_l, R_l)

# %%
def QLearnGraphGammas(p, r):
    mean_vs = []
    rewards = []
    iterations = []
    time = []
    
    gammas = [.9,.95,.99]
    for gamma in gammas:
        print(gamma)
        vi = mdp.QLearning(p, r, gamma=gamma, alpha=0.1, alpha_decay=0.9, epsilon=1, epsilon_decay=0.9, n_iter=50000)
        vi.run()
        vi.run_stats
        iterations.append(list(range(1,len(vi.run_stats)+1)))
        mean_vs.append([el['Mean V'] for el in vi.run_stats])
        rewards.append([el['Reward'] for el in vi.run_stats])
        time.append([val['Time'] for val in vi.run_stats])

    f, ax = plt.subplots(1,3, figsize=(15,12))

    ax[0].set_title("Mean V", fontsize=30)
    for i in range(len(mean_vs)):
        ax[0].plot(iterations[i], mean_vs[i], alpha=1, label="Gamma: {}".format(np.round(gammas[i],3)), linewidth = 5)
    ax[0].legend(fontsize = 20)
    ax[0].tick_params(axis='x', labelsize=25)
    ax[0].tick_params(axis='y', labelsize=25)
    ax[0].set_xlabel("# Iterations", fontsize = 20)

    ax[1].set_title("Reward", fontsize=30)
    for i in range(len(mean_vs)):
        ax[1].plot(iterations[i], rewards[i], alpha=1, label="Gamma: {}".format(np.round(gammas[i],3)), linewidth = 5)
    ax[1].legend(fontsize = 20)
    ax[1].tick_params(axis='x', labelsize=25)
    ax[1].tick_params(axis='y', labelsize=25)
    ax[1].set_xlabel("# Iterations", fontsize = 20)
    plt.plot()

    ax[2].set_title("Time - Policy Iteration", fontsize=30)
    for i in range(len(mean_vs)):
        ax[2].plot(iterations[i], time[i], alpha=1, label="Gamma = {}".format(np.round(gammas[i],3)), linewidth = 5)
    ax[2].legend(fontsize = 20)
    ax[2].tick_params(axis='x', labelsize=25)
    ax[2].tick_params(axis='y', labelsize=25)
    ax[2].set_xlabel("# Iterations", fontsize = 20)
    plt.plot()

QLearnGraphGammas(P_l, R_l)
# %%
