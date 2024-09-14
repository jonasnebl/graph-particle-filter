from tqdm import tqdm
import numpy as np
import pickle
import sys
from datetime import datetime

sys.path.append("build/")
from cpp_utils import Simulation, ParticleTracker
from time import time, sleep
from python.plotter import Plotter
from python.constants import *
from python.accurateTracker import AccurateTracker
from python.confidentTracker import ConfidentTracker


##############################################################
T_simulation = 10 * 60  # 10 minutes

sim = Simulation(T_step=0.1, N_humans=3, N_robots=1)
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


##############################################################

if "sim_log" not in locals():
    filename = os.path.join(LOG_FOLDER, "log_2024-08-30_10-47-52.pkl")
    with open(filename, "rb") as f:
        sim_log = pickle.load(f)
sim_states = sim_log["sim_states"]
T_simulation = sim_log["T_simulation"]
T_step = sim_log["T_step"]
N_humans = sim_log["N_humans"]
N_robots = sim_log["N_robots"]

# load simulation object to have access to member utility functions
sim = Simulation(T_step=T_step, N_humans=N_humans, N_robots=N_robots)


##############################################################
plot = True  # slows down loop significantly!
record_video = False  # slows down loop even more!
if record_video:
    plot = True

confidentTracker = ConfidentTracker(N_robots=N_robots, include_observations=True)
accurateTracker = AccurateTracker(N_robots=N_robots, include_observations=False, train=False)
particleTracker = ParticleTracker(T_step, 3, 10000)

if plot:
    plotter = Plotter(record_frames=record_video)

pbar = tqdm(range(0, len(sim_states)), desc="Simulation")

simulation_time = 0
confidentTracker_node_probabilities = []
accurateTracker_node_probabilities = []
particleTracker_edge_probabilities = []

for i in pbar:
    sim_state = sim_states[i]

    # outer list: robots, inner list: perceived humans for every robot
    robot_perceptions = [
        {
            "ego_position": agent["ego_position"],
            "observable_nodes": agent["observable_nodes"],
            "perceived_humans": agent["perceived_humans"],
        }
        for agent in sim_state
        if agent["type"] == "robot"
    ]

    confidentTracker_node_probabilities.append(confidentTracker.add_observation(robot_perceptions))
    _ = confidentTracker.predict()

    accurateTracker_node_probabilities.append(accurateTracker.add_observation(robot_perceptions))
    _ = accurateTracker.predict()

    particleTracker_edge_probabilities.append(particleTracker.add_observation(robot_perceptions))
    _ = particleTracker.predict()

    if plot:
        plotter.update(sim_state, particleTracker_edge_probabilities[-1])

    pbar.set_postfix(
        {
            "Simulated time": "{:d}:{:02d} of {:d}:{:02d} hours".format(
                int(simulation_time / 3600),
                int(simulation_time / 60) % 60,
                int(T_simulation / 3600),
                int(T_simulation / 60) % 60,
            )
        }
    )

    simulation_time += T_step




if record_video:
    plotter.create_video(T_step)


##############################################################
from python.metrics import Confidence, Accuracy, MeanAveragePrecision

confidentTracker_confidence = Confidence(sim_log, confidentTracker_node_probabilities).per_graph()
accurateTracker_confidence = Confidence(sim_log, accurateTracker_node_probabilities).per_graph()
particleTracker_confidence = Confidence(sim_log, particleTracker_node_probabilities).per_graph()

confidentTracker_accuracy = Accuracy(sim_log, confidentTracker_node_probabilities).per_graph()
accurateTracker_accuracy = Accuracy(sim_log, accurateTracker_node_probabilities).per_graph()
particleTracker_accuracy = Accuracy(sim_log, particleTracker_node_probabilities).per_graph()

confidentTracker_map = MeanAveragePrecision(sim_log, confidentTracker_node_probabilities).per_graph()
accurateTracker_map = MeanAveragePrecision(sim_log, accurateTracker_node_probabilities).per_graph()
particleTracker_map = MeanAveragePrecision(sim_log, particleTracker_node_probabilities).per_graph()

from python.figures import results_plot

print("Mean Average Precision:")
print("ConfidentTracker:", confidentTracker_map)
print("AccurateTracker:", accurateTracker_map)
print("ParticleTracker:", particleTracker_map)

results_plot(
    confidentTracker_confidence=confidentTracker_confidence,
    confidentTracker_accuracy=confidentTracker_accuracy,
    accurateTracker_confidence=accurateTracker_confidence,
    accurateTracker_accuracy=accurateTracker_accuracy,
    particleTracker_confidence=particleTracker_confidence,
    particleTracker_accuracy=particleTracker_accuracy,
)
