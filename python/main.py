import argparse
from tqdm import tqdm
import numpy as np
import pickle
from datetime import datetime
import yaml
from time import time
from plotter import Plotter
from particleTracker import ParticleTracker
from simulation import Simulation
from paths import *
from evaluator import *

# --- Load config and parse arguments ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
parser = argparse.ArgumentParser()
for key, value in config.items():
    if isinstance(value, bool):
        parser.add_argument(f"--{key}", type=lambda x: (str(x).lower() == "true"), default=value)
    else:
        parser.add_argument(f"--{key}", type=type(value), default=value)
args = parser.parse_args()
for key in config.keys():
    config[key] = getattr(args, key)


# --- Run new simulation ---
if config["run_new_simulation"]:
    T_simulation = config["T_simulation"]
    sim = Simulation(
        T_step=config["T_step"],
        N_humans=config["N_humans"],
        N_robots=config["N_robots"],
        allow_warehouse_leaving=config["allow_warehouse_leaving"],
    )
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

    if config["folder"] == "":
        folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        folder = config["folder"]
    os.makedirs(os.path.join(LOG_FOLDER, folder), exist_ok=True)
    filepath = os.path.join(LOG_FOLDER, folder, "log.pkl")
    with open(filepath, "wb") as outp:
        pickle.dump(sim_log, outp, pickle.HIGHEST_PROTOCOL)


if config["run_tracker"]:
    # --- Load simulation ---
    if "sim_log" not in locals():  # no new simulation has been run
        folder = config["folder"]
        filepath = os.path.join(LOG_FOLDER, folder, "log.pkl")
        with open(filepath, "rb") as f:
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
        plotter = Plotter(print_probabilites=True, clear_threshold=config["clear_threshold"])

    particleTracker = ParticleTracker(
        T_step=T_step,
        N_tracks_init=config["N_tracks_init"],
        N_particles=config["N_particles"],
        record_training_data=config["record_training_data"],
        clear_threshold=config["clear_threshold"],
    )

    pbar = tqdm(range(0, int(len(sim_states))), desc="Tracker")
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
        particleTracker_edge_probabilities.append(
            particleTracker.add_observation(robot_perceptions)
        )
        _ = particleTracker.predict()
        particleTracker_execution_times.append(time() - start)

        cleared_edges = particleTracker.get_cleared_edges(particleTracker_edge_probabilities[-1])

        if plot:
            plotter.reset()
            plotter.update_sim_state(sim_states[i])
            plotter.update_edge_probabilities(particleTracker_edge_probabilities[-1])
            plotter.update_cleared_edges(cleared_edges)
            # plotter.update_individual_edge_probabilities(particleTracker_edge_probabilities[-1])
            plotter.show(blocking=False)
        if record_video:
            plotter.capture_frame()

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

    # --- Save logs ---
    if record_video:
        plotter.create_video(T_step, speed=config["playback_speed"])
    with open(os.path.join(LOG_FOLDER, folder, "edge_probabilities.pkl"), "wb") as f:
        pickle.dump(particleTracker_edge_probabilities, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(LOG_FOLDER, folder, "N_perceived.pkl"), "wb") as f:
        pickle.dump(particleTracker.N_perceived_humans_log, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(LOG_FOLDER, folder, "N_estimated.pkl"), "wb") as f:
        pickle.dump(particleTracker.N_humans_estimated_log, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(LOG_FOLDER, folder, "N_tracks.pkl"), "wb") as f:
        pickle.dump(particleTracker.N_tracks_log, f, pickle.HIGHEST_PROTOCOL)
    if config["record_training_data"]:
        with open(os.path.join(LOG_FOLDER, folder, "edge_change_data.pkl"), "wb") as f:
            pickle.dump(particleTracker.edge_change_training_data, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(LOG_FOLDER, folder, "duration_data.pkl"), "wb") as f:
            pickle.dump(particleTracker.duration_training_data, f, pickle.HIGHEST_PROTOCOL)

    # --- Evaluate results ---
    print(
        "Execution times: Mean: {:.2f}ms, Max: {:.2f}ms".format(
            1e3 * np.mean(particleTracker_execution_times),
            1e3 * np.max(particleTracker_execution_times),
        )
    )
