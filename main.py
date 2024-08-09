import sys
sys.path.append('build/')
import simulation
from time import perf_counter, sleep
from python.plotter import Plotter
from python.tracker import Tracker
from python.logger import Logger

T_simulation = 60
T_step = 0.05
record_video = False
realtime = True
plot = True

sim = simulation.Simulation(
    T_step=T_step, 
    N_humans=3, 
    N_robots=1
    )
if plot: plotter = Plotter(record_frames=record_video)
logger = Logger(simulation=sim)
tracker = Tracker()


# Simulate in real-time
N = int(T_simulation / T_step)
for i in range(N):
    start = perf_counter()

    current_state = sim.step(1)[0]

    tracker.add_observation(current_state)
    tracker.predict()

    logger.add_state(current_state)
    logger.add_probabilities(tracker.probabilites)

    if plot: plotter.update(current_state, tracker.probabilites)    
    
    if realtime: sleep(abs(T_step - (perf_counter() - start)))

logger.save()
if record: plotter.record_video(T_step)
