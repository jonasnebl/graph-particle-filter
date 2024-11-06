from paths import *

import pickle

folder = "24hours_4humans_4robots_100part"

with open(os.path.join(LOG_FOLDER, folder, "edge_change_training_data.pkl"), "rb") as f:
    edge_change_training_data = pickle.load(f)

with open(os.path.join(LOG_FOLDER, folder, "edge_change_training_data_magic.pkl"), "rb") as f:
    edge_change_training_data_magic = pickle.load(f)

score = 0
for training_sample in edge_change_training_data:
    if training_sample in edge_change_training_data_magic:
        score += 1


print(
    "{:.2f}% of the non-magic samples are also in the magic samples.".format(
        100 * score / len(edge_change_training_data)
    )
)
print("N_training_samples:", len(edge_change_training_data))
print("N_training_samples_magic:", len(edge_change_training_data_magic))
print(
    "N_normal / N_magic = {:.2f}%".format(
        100 * len(edge_change_training_data) / len(edge_change_training_data_magic)
    )
)
