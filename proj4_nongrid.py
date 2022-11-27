#%%
from hiive.mdptoolbox import mdp, example
from hiive.mdptoolbox.mdp import ValueIteration
from hiive.mdptoolbox.mdp import PolicyIteration
from hiive.mdptoolbox.mdp import QLearning
from hiivemdptoolbox.hiive.mdptoolbox.example import openai
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
########################################
# Black Jack https://gist.github.com/iiLaurens/ba9c479e71ee4ceef816ad50b87d9ebd
########################################

# Code to set up environment
from itertools import product
from functools import reduce
import numpy as np

ACTIONLIST = {
    0: 'skip',
    1: 'draw'
}

CARDS = np.array([2,3,4,5,6,7,8,9,10,10,10,10,11])
BLACKJACK = 21
DEALER_SKIP = 17
STARTING_CARDS_PLAYER = 2
STARTING_CARDS_DEALER = 1

STATELIST = {0: (0,0,0)} # Game start state
STATELIST = {**STATELIST, **{nr+1:state for nr, state in enumerate(product(range(2), range(CARDS.min()*STARTING_CARDS_PLAYER,BLACKJACK + 2), range(CARDS.min()*STARTING_CARDS_DEALER, BLACKJACK+2)))}}


def cartesian(x,y):
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2).sum(axis=1)

def deal_card_probability(count_now, count_next, take=1):
    if take > 1:
        cards = reduce(cartesian, [CARDS]*take)
    else:
        cards = CARDS
        
    return (np.minimum(count_now + cards, BLACKJACK + 1) == count_next).sum() / len(cards)

def is_gameover(skipped, player, dealer):
    return any([
        dealer >= DEALER_SKIP and skipped == 1,
        dealer > BLACKJACK and skipped == 1,
        player > BLACKJACK
     ])

def blackjack_probability(action, stateid_now, stateid_next):
    skipped_now, player_now, dealer_now  = STATELIST[stateid_now]
    skipped_next, player_next, dealer_next = STATELIST[stateid_next]
    
    if stateid_now == stateid_next:
        # Game cannot stay in current state
        return 0.0
    
    if stateid_now == 0:
        if skipped_next == 1:
            # After start of the game the game cannot be in a skipped state
            return 0
        else:
            # State lower or equal than 1 is a start of a new game
            dealer_prob = deal_card_probability(0, dealer_next, take=STARTING_CARDS_DEALER)
            player_prob = deal_card_probability(0, player_next, take=STARTING_CARDS_PLAYER)

            return dealer_prob * player_prob
    
    if is_gameover(skipped_now, player_now, dealer_now):
        # We arrived at end state, now reset game
        return 1.0 if stateid_next == 0 else 0.0
    
    if skipped_now == 1:
        if skipped_next == 0 or player_next != player_now:
            # Once you skip you keep on skipping in blackjack
            # Also player cards cannot increase once in a skipped state
            return 0.0
    
    if ACTIONLIST[action] == 'skip' or skipped_now == 1:
        # If willingly skipped or in forced skip (attempted draw in already skipped game):
        if skipped_next != 1 or player_now != player_next:
            # Next state must be a skipped state with same card count for player
            return 0.0
    
    if ACTIONLIST[action] == 'draw' and skipped_now == 0 and skipped_next != 0:
        # Next state must be a drawable state
        return 0.0
    
    if dealer_now != dealer_next and player_now != player_next:
        # Only the player or the dealer can draw a card. Not both simultaneously!
        return 0.0

    # Now either the dealer or the player draws a card
    if ACTIONLIST[action] == 'draw' and skipped_now == 0:
        # Player draws a card
        prob = deal_card_probability(player_now, player_next, take=1)
    else:
        # Dealer draws a card
        if dealer_now >= DEALER_SKIP:
            if dealer_now != dealer_next:
                # Dealer always stands once it has a card count higher than set amount
                return 0.0
            else:
                # Dealer stands
                return 1.0

        prob = deal_card_probability(dealer_now, dealer_next, take=1)

    return prob

def blackjack_rewards(action, stateid):
    skipped, player, dealer  = STATELIST[stateid]
    
    if not is_gameover(skipped, player, dealer):
        return 0
    elif player > BLACKJACK or (player <= dealer and dealer <= BLACKJACK):
        return -1
    elif player == BLACKJACK and dealer < BLACKJACK:
        return 1.5
    elif player > dealer or dealer > BLACKJACK:
        return 1
    else:
        raise Exception(f'Undefined reward: {skipped}, {player}, {dealer}')
    
def print_blackjack_policy(policy):
    idx = pd.MultiIndex.from_tuples(list(STATELIST.values()), names=['Skipped', 'Player','Dealer'])
    S = pd.Series([1 if i == 1 else 0 for i in policy], index=idx)
    S = S.loc[S.index.get_level_values('Skipped')==0].reset_index('Skipped', drop=True)
    S = S.loc[S.index.get_level_values('Dealer')>0]
    S = S.loc[S.index.get_level_values('Player')>0]
    return S

def print_blackjack_rewards():
    idx = pd.MultiIndex.from_tuples(list(STATELIST.values()), names=['Skipped', 'Player', 'Dealer'])
    S = pd.Series(R[:,0], index=idx)
    S = S.loc[S.index.get_level_values('Skipped')==1].reset_index('Skipped', drop=True)
    S = S.loc[S.index.get_level_values('Player')>0]
    S = S.loc[S.index.get_level_values('Dealer')>0]
    return S


# Define transition matrix
P = np.zeros((len(ACTIONLIST), len(STATELIST), len(STATELIST)))
iter = 0
for a, i, j in product(ACTIONLIST.keys(), STATELIST.keys(), STATELIST.keys()):
    print(f'a = {a}, i = {i}, j = {j}')
    if iter == 799:
        break
    P[a,i,j] = blackjack_probability(a, i, j)
    iter += 1
    
# Define reward matrix
R = np.zeros((len(STATELIST), len(ACTIONLIST)))
for a, s in product(ACTIONLIST.keys(), STATELIST.keys()):
    R[s, a] = blackjack_rewards(a, s)

# Check that we have a valid transition matrix with transition probabilities summing to 1
assert (P.sum(axis=2).round(10) == 1).all()

#%%

def val_iter_plot(P, R):
    mean_vs_vi = []
    rewards_vi = []
    time_vi = []
    iterations_vi = []
    mean_vs_pi = []
    rewards_pi = []
    time_pi = []
    iterations_pi = []
    gammas = list(np.arange(0.2,1,0.15))

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

    f, ax = plt.subplots(2,2, figsize=(25,20))

    ax[0][0].set_title("Reward - Value Iteration", fontsize=30)
    for i in range(len(rewards_vi)):
        ax[0][0].plot(iterations_vi[i], rewards_vi[i], alpha=1, label="Gamma = {}".format(np.round(gammas[i],3)), linewidth = 5)
    ax[0][0].legend(fontsize = 20)
    ax[0][0].tick_params(axis='x', labelsize=25)
    ax[0][0].tick_params(axis='y', labelsize=25)
    ax[0][0].set_xlabel("# Iterations", fontsize = 20)

    ax[0][1].set_title("Time - Value Iteration", fontsize=30)
    for i in range(len(rewards_vi)):
        ax[0][1].plot(iterations_vi[i], time_vi[i], alpha=1, label="Gamma = {}".format(np.round(gammas[i],3)), linewidth = 5)
    ax[0][1].legend(fontsize = 20)
    ax[0][1].tick_params(axis='x', labelsize=25)
    ax[0][1].tick_params(axis='y', labelsize=25)
    ax[0][1].set_xlabel("# Iterations", fontsize = 20)

    ax[1][0].set_title("Reward - Policy Iteration", fontsize=30)
    for i in range(len(rewards_pi)):
        ax[1][0].plot(iterations_pi[i], rewards_pi[i], alpha=1, label="Gamma = {}".format(np.round(gammas[i],3)), linewidth = 5)
    ax[1][0].legend(fontsize = 20)
    ax[1][0].tick_params(axis='x', labelsize=25)
    ax[1][0].tick_params(axis='y', labelsize=25)
    ax[1][0].set_xlabel("# Iterations", fontsize = 20)

    ax[1][1].set_title("Time - Policy Iteration", fontsize=30)
    for i in range(len(rewards_pi)):
        ax[1][1].plot(iterations_pi[i], time_pi[i], alpha=1, label="Gamma = {}".format(np.round(gammas[i],3)), linewidth = 5)
    ax[1][1].legend(fontsize = 20)
    ax[1][1].tick_params(axis='x', labelsize=25)
    ax[1][1].tick_params(axis='y', labelsize=25)
    ax[1][1].set_xlabel("# Iterations", fontsize = 20)
    plt.plot()

val_iter_plot(P, R)


# %%
import numpy as np
import pandas as pd
import seaborn as sns
optimal_p = pd.DataFrame(print_blackjack_policy(vi.policy)).reset_index().rename(columns = {0:"Policy"})

def get_optimal_p(p, d):
    return(optimal_p[(optimal_p.Player == p) & (optimal_p.Dealer == d)].Policy.values[0])

dealer = np.linspace(2, 22, 21)
player = np.linspace(4, 22, 19)

data = np.zeros((21, 19))
for i, x in enumerate(dealer):
    for j, y in enumerate(player):
        data[i, j] = get_optimal_p(y, x)

df = pd.DataFrame(data, index=dealer, columns=player)

fig = sns.heatmap(df)
fig.set(xlabel='Player Hand', ylabel='Dealer Hand')
plt.show()
# %%
