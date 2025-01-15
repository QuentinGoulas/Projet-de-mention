from mountain_car_problem import *

with open('weights.pkl', 'rb') as f:
    w = pickle.load(f)
    f.close()
    
pos_values = np.linspace(pos_range[0], pos_range[1], num=50)
vel_values = np.linspace(vel_range[0], vel_range[1], num=50)
pos_grid, vel_grid = np.meshgrid(pos_values, vel_values)
q_grid = np.zeros_like(pos_grid)

for i in range(pos_grid.shape[0]):
    for j in range(pos_grid.shape[1]):
        state = [pos_grid[i, j], vel_grid[i, j]]
        actions_feature = [tile_code(state, action) for action in possible_actions]
        q_values = actions_feature @ w
        q_grid[i, j] = -np.max(q_values)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(pos_grid, vel_grid, q_grid, cmap='viridis')
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_zlabel('Q-value')
plt.show()