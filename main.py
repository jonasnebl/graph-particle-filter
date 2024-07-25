import sys
sys.path.append('build/')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import simulation
from time import perf_counter

# parameters
T_step = 0.1
T_simulation = 1

# Initialize the simulation
sim = simulation.Simulation(T_step=T_step, N_humans=1, N_robots=1)

# Create a figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Initialize scatter plot
scat = ax.scatter([], [])

# Simulate in real-time
N = int(T_simulation / T_step)
sim_state = []
for i in range(N):
    start = perf_counter()

    sim_state += sim.step(1)
    print(sim_state[-1])

    # Update the scatter plot
    positions = [(agent['x'], agent['y']) for agent in sim_state[-1]]
    scat.set_offsets(positions)
    
    # Pause for realtime simulation
    if perf_counter() - start > T_step:
        plt.pause(1e3)
    else:
        plt.pause(T_step - (perf_counter() - start))

plt.show()
