import json
import numpy as np
import pickle
from datetime import datetime
import os
import sys
sys.path.append('build/')
import simulation
from time import perf_counter, sleep
from plotter import Plotter
from tracker import Tracker

# parameters
T_step = 0.05
T_simulation = 60

# Initialize the simulation
N_humans = 3
N_robots = 1
sim = simulation.Simulation(T_step=T_step, N_humans=N_humans, N_robots=N_robots)
plotter = Plotter()
tracker = Tracker()

# Simulate in real-time
N = int(T_simulation / T_step)
sim_state = []
for i in range(N):
    start = perf_counter()

    sim_state += sim.step(1)
    current_state = sim_state[-1]

    tracker.add_observation(current_state)
    tracker.predict()

    plotter.update(current_state, tracker.probabilites)    
    
    # Pause for realtime simulation
    sleep(abs(T_step - (perf_counter() - start)))

LOG_FOLDER = 'logs'
with open(os.path.join(LOG_FOLDER, 'log_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'), 'wb') as outp:
    pickle.dump({
        'T_step': T_step,
        'T_simulation': T_simulation,
        'N_humans': N_humans,
        'N_robots': N_robots,
        'sim_state': sim_state
    }, outp, pickle.HIGHEST_PROTOCOL)
