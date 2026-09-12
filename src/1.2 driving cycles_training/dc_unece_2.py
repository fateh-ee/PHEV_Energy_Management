import numpy as np
import matplotlib.pyplot as plt
# source: https://github.com/dabo248/nedc/blob/master/README.md
# Given data
data = [
(0, 0, 0, 20),
(0, 15, 0.69, 6),
(15, 35, 0.51, 11),
(35, 50, 0.42, 10),
(50, 70, 0.4, 14),
(70, 70, 0, 50),
(70, 50, -0.69, 8),
(50, 50, 0, 69),
(50, 70, 0.43, 13),
(70, 70, 0, 50),
(70, 100, 0.24, 35),
(100, 100, 0, 30),
(100, 120, 0.28, 20),
(120, 120, 0, 10),
(120, 80, -0.69, 16),
(80, 50, -1.04, 8),
(50, 0, -1.39, 10),
(0, 0, 0, 20)

]


# Extracting data into separate lists
start_velocities, end_velocities, accelerations, durations = zip(*data)
# Initialize time and velocity lists
time_points = [0]
velocity_points = [start_velocities[0]]
# Calculate time and velocity points based on the given data
for i in range(len(data)):
    time_points.append(time_points[-1] + durations[i])
    velocity_points.append(end_velocities[i])
# Plotting the time vs velocity curve
#plt.figure(figsize=(10, 6))
# Interpolate to each second
times_all = np.arange(0, max(time_points) + 1)
velocity_new = np.interp(times_all, time_points, velocity_points)

plt.figure(1)
plt.plot(times_all, velocity_new)
plt.title('Time vs Velocity Curve')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity')
plt.grid(True)
plt.show(block=False)


# Extracting time and acceleration data
time = 0
times = [time]
accelerations_points = []

for entry in data:
    _, _, acceleration, duration = entry
    time += duration
    times.append(time)
    accelerations_points.append(acceleration)

# Interpolate to each second
times_all = np.arange(0, max(times[:-1]) + 1)
accelerations_new = np.interp(times_all, times[:-1], accelerations_points)
#print(f'acceleration: {accelerations_new:}')

# Plotting time vs acceleration curve
plt.figure(2)
plt.plot(times_all, accelerations_new)
plt.title('Time vs Acceleration Curve')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.grid(True)
plt.show()

np.savetxt('acc_unece_2.txt',accelerations_new)



