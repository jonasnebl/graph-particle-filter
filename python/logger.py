from .constants import LOG_FOLDER
import pickle
import os
import numpy as np
from datetime import datetime

class Logger:
    def __init__(self, simulation):
        """Initialize the logger with the simulation object.
        """
        self.LOG_FOLDER = LOG_FOLDER
        self.T_step = simulation.T_step
        self.N_humans = simulation.N_humans
        self.N_robots = simulation.N_robots
        self.sim_state = []
        self.tracker_probabilites = []

    def add_state(self, state):
        """Add the current state of the simulation to the logger.
        """
        self.sim_state += state

    def add_probabilities(self, probabilities):
        """Add the current probabilities of the tracker to the logger.
        """
        if type(probabilities) == np.ndarray:
            probabilities = [probabilities]
        self.tracker_probabilites + probabilities

    def save(self):
        with open(os.path.join(self.LOG_FOLDER, 'log_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'), 'wb') as outp:
            pickle.dump({
                'T_step': self.T_step,
                'T_simulation': np.arange(0, len(sim_state) * T_step, T_step),
                'N_humans': self.N_humans,
                'N_robots': self.N_robots,
                'sim_state': self.sim_state,
                'tracker_probabilities': self.tracker_probabilites
            }, outp, pickle.HIGHEST_PROTOCOL)
