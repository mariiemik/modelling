import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib
from scipy.integrate import odeint
matplotlib.use('Agg')

import matplotlib.pyplot as plt

guiFlag = False

dt = 1/240 # pybullet simulation step
th0 = 0.5  # starting position (radian)
thd = 1.0  # desired position (radian)
kp = 40.0  # proportional coefficient
ki = 40.0
kd = 20.0
g = 10     # m/s^2
L = 0.8    # m
L1 = L
L2 = L
m = 1      # kg
f0 = 10    # applied const force

xd = 0.5
zd = 1

physicsClient = p.connect(p.GUI if guiFlag else p.DIRECT) # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-g)
# planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF("./two-link.urdf.xml", useFixedBase=True)

# get rid of all the default damping forces
# think of it as imagined "air drag"
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

# go to the starting position
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=th0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

pos0 = p.getLinkState(boxId, 4)[0]
X0 = np.array([[pos0[0]],[pos0[2]]])

maxTime = 5 # seconds
logTime = np.arange(0, maxTime, dt)
sz = len(logTime)
logXsim = np.zeros(sz)
logZsim = np.zeros(sz)
idx = 0
T = 2
for t in logTime:
    th1 = p.getJointState(boxId, 1)[0]
    vel = p.getJointState(boxId, 1)[1]
    th2 = p.getJointState(boxId, 3)[0]
    ve2 = p.getJointState(boxId, 3)[1]

    pos = p.getLinkState(boxId, 4)[0]
    logXsim[idx] = pos[0]
    logZsim[idx] = pos[2]


    zero_vec = [0.0, 0.0]
    joint_positions = [th1, th2]
    jac_lin, jac_ang = p.calculateJacobian(bodyUniqueId=boxId,
                                        linkIndex=4,
                                        localPosition=[0,0,0],
                                        objPositions=joint_positions,
                                        objVelocities=zero_vec,
                                        objAccelerations=zero_vec)[:2]

    jac = np.array([
        [jac_lin[0][0], jac_lin[0][1]],  # x ось
        [jac_lin[2][0], jac_lin[2][1]]   # z ось
    ])

    jac_inv = np.linalg.inv(jac)

    X = np.array([[pos[0]],[pos[2]]])
    Xd = np.array([[xd],[zd]])

    s = 1
    if t < T:
        s = (3/T**2) * t**2 -2/(T**3) * t**3
    Xd_curr = X0 + s * (Xd - X0)

    vel_d = -100.0 * jac_inv @ (X-Xd_curr)
    vel_d = vel_d.flatten()

    p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=[1,3], targetVelocities=vel_d, controlMode=p.VELOCITY_CONTROL)
    p.stepSimulation()

    idx += 1
    if guiFlag:
        time.sleep(dt)
p.disconnect()

plt.subplot(2,1,1)
plt.plot(logTime, logXsim)
plt.subplot(2,1,2)
plt.plot(logTime, logZsim)
plt.tight_layout()
plt.savefig("plot.png")
plt.close()
print("График сохранён в plot.png")

