import numpy as np
from matplotlib import pyplot as plt

# create a piecewise constant zmp trajectory
times = [0, 1, 2, 3, 4, 5]
zmp = np.zeros(500)

# plot the zmp trajectory
plt.figure()
plt.plot(zmp)
plt.show()

# initialize the DCM
offset = 0.01
dcm = np.array([zmp[0] + offset] * len(zmp))

# integrate the DCM dynamics
dt = 0.01
omega = 3.0
for i in range(0, len(zmp) - 1):
    dcm_dot = omega * (dcm[i] - zmp[i])
    dcm_next = dcm[i] + dcm_dot * dt
    dcm[i + 1] = dcm_next

# plot the DCM trajectory
plt.figure()
plt.plot(dcm)
plt.show()

# replan ZMP every 0.5 s to reset offset
dcm_replanned = np.array([zmp[0] + offset] * len(zmp))
zmp_replanned = zmp.copy()

for i in range(len(zmp) - 1):
    # at replan times, reset the ZMP so that dcm - zmp = offset
    if i % 50 == 0 and i > 0:
        zmp_replanned[i:] = dcm_replanned[i] - offset

    # integrate DCM dynamics
    dcm_dot = omega * (dcm_replanned[i] - zmp_replanned[i])
    dcm_replanned[i + 1] = dcm_replanned[i] + dcm_dot * dt

# plot the replanned DCM trajectory
plt.figure()
plt.plot(dcm_replanned, label='DCM')
plt.plot(zmp_replanned, label='ZMP')
plt.legend()
plt.title('DCM with ZMP replanning every 0.5s')
plt.show()
