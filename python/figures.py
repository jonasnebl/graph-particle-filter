import os
from .constants import *
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

# plt.style.use('./figures/latex_matplotlib.mplstyle')
plt.style.use("default")


def results_plot(
    confidentTracker_confidence,
    confidentTracker_accuracy,
    accurateTracker_confidence,
    accurateTracker_accuracy,
    particleTracker_confidence,
    particleTracker_accuracy,
):
    x = [confidentTracker_confidence, accurateTracker_confidence, particleTracker_confidence]
    y = [confidentTracker_accuracy, accurateTracker_accuracy, particleTracker_accuracy]
    legends = ["Konfidenter Tracker", "Genauer Tracker", "Partikel-Tracker"]
    colors = ["red", "green", "blue"]

    plt.scatter(x, y, c=colors)

    for i, legend in enumerate(legends):
        plt.annotate(legend, (x[i], y[i]))

    plt.title("Konfidenz und Genauigkeit")
    plt.xlabel("Konfidenz")
    plt.ylabel("Genauigkeit")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, "results_plot.pdf"))
    plt.show()
