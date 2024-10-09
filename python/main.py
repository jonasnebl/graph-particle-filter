from tqdm import tqdm
import numpy as np
import pickle
import sys
from datetime import datetime
import yaml
from time import time, sleep
from plotter import Plotter
from particleTracker import ParticleTracker
from simulation import Simulation
from constants import *

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# --- Run new simulation ---
if config["run_new_simulation"]:
    T_simulation = config["T_simulation"]
    sim = Simulation(T_step=config["T_step"], N_humans=config["N_humans"], N_robots=config["N_robots"])
    sim_states = []
    N_minutes = int(T_simulation / 60)
    N_hours = int(T_simulation / 3600)
    pbar = tqdm(range(0, N_minutes), desc="Simulation")
    for i in pbar:  # always calculate one minute at once to have C++ speedup
        sim_states += sim.step(int(60 / sim.T_step))
        pbar.set_postfix(
            {
                "Simulated time": "{:d}:{:02d} of {:d}:{:02d} hours".format(
                    int(i / 60), i % 60, int(T_simulation / 3600), int(T_simulation / 60) % 60
                )
            }
        )

    sim_log = {
        "T_step": sim.T_step,
        "T_simulation": T_simulation,
        "N_humans": sim.N_humans,
        "N_robots": sim.N_robots,
        "sim_states": sim_states,
    }

    filename = os.path.join(LOG_FOLDER, "log_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl")
    with open(filename, "wb") as outp:
        pickle.dump(sim_log, outp, pickle.HIGHEST_PROTOCOL)


# --- Load simulation ---
if "sim_log" not in locals():  # no new simulation has been run
    filename = os.path.join(LOG_FOLDER, config["log_file"])
    with open(filename, "rb") as f:
        sim_log = pickle.load(f)
sim_states = sim_log["sim_states"]
T_simulation = sim_log["T_simulation"]
T_step = sim_log["T_step"]
N_humans = sim_log["N_humans"]
N_robots = sim_log["N_robots"]


# --- Run tracker on playback simulation data ---
plot = config["plot"]  # slows down loop significantly!
record_video = config["record_video"]  # slows down loop even more!
if record_video:
    plot = True
if plot:
    plotter = Plotter(record_frames=record_video, print_probabilites=True)

if config["N_humans_tracker"] == -1:
    N_humans_tracker = N_humans
else:
    N_humans_tracker = config["N_humans_tracker"]
particleTracker = ParticleTracker(T_step, N_humans_tracker, config["N_particles"])

pbar = tqdm(range(0, int(len(sim_states))), desc="Simulation")
simulation_time = 0
particleTracker_edge_probabilities = []
particleTracker_execution_times = []
for i in pbar:
    robot_perceptions = [
        {
            "position": agent["position"],
            "perceived_humans": agent["perceived_humans"],
        }
        for agent in sim_states[i]
        if agent["type"] == "robot"
    ]

    start = time()
    particleTracker_edge_probabilities.append(particleTracker.add_observation(robot_perceptions))
    _ = particleTracker.predict()
    particleTracker_execution_times.append(time() - start)

    if plot:
        plotter.update(sim_states[i], particleTracker_edge_probabilities[-1])
    pbar.set_postfix(
        {
            "Simulated time": "{:d}:{:02d} of {:d}:{:02d} hours; T_Tracker: {:.0f}ms".format(
                int(simulation_time / 3600),
                int(simulation_time / 60) % 60,
                int(T_simulation / 3600),
                int(T_simulation / 60) % 60,
                1e3 * particleTracker_execution_times[-1],
            )
        }
    )
    simulation_time += T_step


# --- Evaluate results ---
print(
    "Execution times: Mean: {:.2f}ms, Max: {:.2f}ms".format(
        1e3 * np.mean(particleTracker_execution_times), 1e3 * np.max(particleTracker_execution_times)
    )
)
if record_video:
    plotter.create_video(T_step, speed=config["playback_speed"])
particleTracker.save_training_data()
