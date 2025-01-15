from mountain_car_problem import *
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_car_dynamics(w):
    state = init_state()
    positions = []
    velocities = []
    done = False
    step = 0

    while not done:
        actions_feature = [tile_code(state, action) for action in possible_actions]
        q_values = actions_feature @ w
        action = epsilon_greedy(q_values, 0)
        reward, state = take_action(state, action)
        if reward == 0:
            done = True
        positions.append(state[0])
        velocities.append(state[1])
        if done:
            break

    fig, ax = plt.subplots()
    car, = ax.plot([], [], 'bo', ms=10)
    ax.set_xlim(-1.2, 0.6)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Position')
    ax.set_ylabel('')

    def init():
        car.set_data([], [])
        return car,

    def animate(i):
        car.set_data([positions[i]], [np.sin(3*positions[i])])
        return car,

    ani = FuncAnimation(fig, animate, init_func=init, frames=len(positions), interval=10, blit=True)
    x = np.linspace(-1.2, 0.6, 100)
    ax.plot(x, np.sin(3*x), 'r-', lw=2)
    plt.show()

with open('weights.pkl', 'rb') as f:
    w = pickle.load(f)
    f.close()

plot_car_dynamics(w)