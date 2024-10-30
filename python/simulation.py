"""
Python wrapper for the C++ simulation class.
Enables documentation and type hints for the Simulation class.
"""

import sys

sys.path.append("build/")  # allows to import cpp_utils
from cpp_utils import Simulation as Simulation_cpp


class Simulation:
    def __init__(self, T_step: float, N_humans: int, N_robots: int):
        """Initialize the simulation.

        :param T_step: double, step time of the simulation.
        :param N_humans: int, Number of humans in tshe simulation.
        :param N_robots: int, Number of robots in the simulation.
        :return: Simulation object.
        """
        self.T_step = T_step
        self.N_humans = N_humans
        self.N_robots = N_robots
        self.sim = Simulation_cpp(T_step, N_humans, N_robots)

    def step(self, N_steps: int) -> list[dict]:
        """Run the simulation for N_steps steps.

        :param N_steps: int, Number of steps to run the simulation for.
        :return: current simulation state as a list of dictionaries, one for every agent
        """
        return self.sim.step(N_steps)
