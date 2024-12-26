import numpy as np
import matplotlib.pyplot as plt

# Load the data from the file
with open('pauses.txt', 'r') as file:
    lines = file.readlines()

# Extract the pause times from the lines
pause_times = []
for line in lines:
    parts = line.split(':')
    if len(parts) == 2 and parts[1].strip():
        try:
            pause_time = float(parts[1].strip())
            pause_times.append(pause_time)
        except ValueError:
            print(f"Skipping invalid line: {line.strip()}")

# Calculate the expected value (mean) and standard deviation
mean_pause_time = np.mean(pause_times)
std_pause_time = np.std(pause_times)

# Calculate quantiles
quantile_67 = np.percentile(pause_times, 67)
quantile_98 = np.percentile(pause_times, 98)
quantile_99 = np.percentile(pause_times, 99)

# Print the results
print(f"Expected Value (Mean): {mean_pause_time}")
print(f"Standard Deviation: {std_pause_time}")
print(f"67% Quantile: {quantile_67}")
print(f"98% Quantile: {quantile_98}")
print(f"99% Quantile: {quantile_99}")

# Plot the histogram
plt.hist(pause_times, bins=20, edgecolor='black', alpha=0.7)
plt.title('Histogram of Pause Times')
plt.xlabel('Pause Time')
plt.ylabel('Frequency')

# Add mean and standard deviation lines
plt.axvline(mean_pause_time, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_pause_time:.2f}')
plt.axvline(mean_pause_time + std_pause_time, color='g', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_pause_time:.2f}')
plt.axvline(mean_pause_time - std_pause_time, color='g', linestyle='dashed', linewidth=1)

# Add quantile lines
plt.axvline(quantile_67, color='b', linestyle='dotted', linewidth=1, label=f'67% Quantile: {quantile_67:.2f}')
plt.axvline(quantile_98, color='m', linestyle='dotted', linewidth=1, label=f'98% Quantile: {quantile_98:.2f}')
plt.axvline(quantile_99, color='c', linestyle='dotted', linewidth=1, label=f'99% Quantile: {quantile_99:.2f}')

plt.legend()
plt.show()