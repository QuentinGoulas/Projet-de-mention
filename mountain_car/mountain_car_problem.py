import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pickle
import matplotlib.pyplot as plt


pos_range = [-1.2, 0.5]
vel_range = [-0.07, 0.07]
possible_actions = np.array([-1, 0, 1])
num_tilings = 10
num_tiles = 9

num_action = len(possible_actions)
pos_offset_unit = np.random.rand(num_tilings,2)
num_tile_feature = num_tilings*num_tiles*num_tiles


def tile_code(state, action):
    features = np.zeros(num_tile_feature * num_action, dtype=int)
    pos_scale = num_tiles / (pos_range[1] - pos_range[0])
    vel_scale = num_tiles / (vel_range[1] - vel_range[0])
    for i in range(num_tilings):
        pos_offset = pos_offset_unit[i,0] / pos_scale
        vel_offset = pos_offset_unit[i,1] / vel_scale
        pos_index = int((state[0] + pos_offset) * pos_scale)
        vel_index = int((state[1] + vel_offset) * vel_scale)
        index = i * num_tiles * num_tiles * num_action + pos_index * num_tiles * num_action + vel_index * num_action + (action + 1)
        features[index] = 1
    return features

def take_action(state, action):
    state[1] = np.clip(state[1] + action*1e-3 - np.cos(3*state[0])*2.5*1e-3,-0.07,0.07) # speed of the car
    state[0] = np.clip(state[0] + state[1], -1.2, 0.5)  # Bound the value between -1.2 and 0.6 | postion of the car
    reward = (state[0] == 0.5) - 1
    if state[0] == -1.2:
        state[1] = 0
    return reward, state
    
# state = [-1.2, -1]
# print(take_action(state,1))
# print(state)

def init_state():
    # return a random state from the ranges
    return [np.random.rand() * (pos_range[1] - pos_range[0]) + pos_range[0], np.random.rand() * (vel_range[1] - vel_range[0]) + vel_range[0]]

def epsilon_greedy(q_values, epsilon=0.1):
    if np.random.rand() > epsilon:
        return np.argmax(q_values)
    else:
        return np.random.choice(len(q_values))
    
if __name__ == "__main__":

    I = 9000 # number of episodes
    epsilon = 0.01
    alpha = 0.05
    gamma = 1
    lmda = 0.9

    F_a = []
    n_features = num_tile_feature * num_action
    w = np.zeros(n_features)

    for i_ep in range(I):
        e = np.zeros(n_features)
        s = init_state()
        actions_feature = [tile_code(s,action) for action in possible_actions]
        q_values = actions_feature@w
        a_idx = epsilon_greedy(q_values,epsilon)
        F_a = actions_feature[a_idx]
        FF = np.copy(F_a)
        step_counter = 0
        while True:
            # if step_counter%100 == 0:
            #     print(f"Step:{step_counter}")
            # if s[0] > 0:
            #     print(s[0])
            step_counter += 1
            e[F_a == 1] = 1
            R, s = take_action(s,possible_actions[a_idx])
            delta = R - F_a @ w
            if R == 0:
                w += alpha*delta*e
                print(f"Episode number {i_ep}/{I} terminated in {step_counter}")
                break
            actions_feature = [tile_code(s,action) for action in possible_actions]
            q_values = actions_feature @ w
            a_idx = epsilon_greedy(q_values,epsilon)
            F_a = actions_feature[a_idx]
            delta += gamma*q_values[a_idx]
            w = w + alpha*delta*e
            e = gamma*lmda*e
            # if step_counter % 100 == 0: print(w,q_values,delta,e,s,F_a)

    # Save the weights and other necessary values for a 3D surf plot
    with open('weights.pkl', 'wb') as f:
        pickle.dump(w, f)

