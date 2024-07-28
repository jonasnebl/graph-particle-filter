import json
import numpy as np
import pickle
from datetime import datetime
import os
import sys
sys.path.append('build/')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import simulation
from time import perf_counter

# parameters
T_step = 0.05
T_simulation = 20

# Initialize the simulation
N_humans = 1
N_robots = 1
sim = simulation.Simulation(T_step=T_step, N_humans=N_humans, N_robots=N_robots)

# load graph
with open('graph_data.json', 'r') as f:
    graph_data = json.load(f)
nodes = graph_data['nodes']
edges = graph_data['edges']
node_positions = np.array([[node['x'], node['y']] for node in nodes])

# Create a figure and axis
fig, ax = plt.subplots()

# Initialize scatter plots
scat_graph = ax.scatter(node_positions[:, 0], node_positions[:, 1], s=100, c='skyblue', zorder=2)
scat_agents = ax.scatter([], [], s=100, c='red', zorder=3)

# Plot the edges
for edge in edges:
    start, end = edge
    start_pos = node_positions[start]
    end_pos = node_positions[end]
    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'b-', zorder=1)

# Simulate in real-time
N = int(T_simulation / T_step)
sim_state = []
for i in range(N):
    start = perf_counter()

    sim_state += sim.step(1)
    print(sim_state[-1])

    # Update the scatter plot
    positions = [(agent['x'], agent['y']) for agent in sim_state[-1]]
    scat_agents.set_offsets(positions)
    
    # Pause for realtime simulation
    if perf_counter() - start > T_step:
        plt.pause(1e3)
    else:
        plt.pause(T_step - (perf_counter() - start))

LOG_FOLDER = 'logs'
with open(os.path.join(LOG_FOLDER, 'log_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'), 'wb') as outp:
    pickle.dump({
        'T_step': T_step,
        'T_simulation': T_simulation,
        'N_humans': N_humans,
        'N_robots': N_robots,
        'sim_state': sim_state
    }, outp, pickle.HIGHEST_PROTOCOL)

plt.show()
