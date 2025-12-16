import numpy as np

T = np.array([
    [-0.0391813,  0.997477,        0.0591828,     0.21651135],
    [0.357553362,   0.0693022830228,  -0.93137746639,  0.319527506828],
    [-0.9330704212, -0.015329210088, -0.359366953373, 0.757318019867],
    [0.0,           0.0,            0.0,           1.0]
])

# extract rotation and translation
R = T[:3, :3]
t = T[:3, 3]

# inverse of rigid transform
R_inv = R.T
t_inv = -R_inv @ t

T_inv = np.eye(4)
T_inv[:3, :3] = R_inv
T_inv[:3, 3] = t_inv

np.savetxt(
    "transforms.txt",
    T_inv,
    fmt="%.8f"
)
