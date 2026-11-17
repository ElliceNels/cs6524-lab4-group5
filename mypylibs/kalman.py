"""Constant-velocity Kalman filter over the state [x, y, vx, vy] in pixels and pixels/frame."""
import numpy as np

F = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], float)
# Optical flow measures both position and velocity, so the whole state is observed
H = np.eye(4)


def process_noise(sigma_a):
    """Unmodelled acceleration with std `sigma_a` px/frame^2 acting as white noise, dt = 1."""
    block = np.array([[1 / 4, 1 / 2],
                      [1 / 2, 1]]) * sigma_a ** 2
    q = np.zeros((4, 4))
    q[np.ix_([0, 2], [0, 2])] = block
    q[np.ix_([1, 3], [1, 3])] = block
    return q


class KalmanFilter:
    def __init__(self, x, P, Q):
        self.x = np.asarray(x, float)
        self.P = np.asarray(P, float)
        self.Q = Q
        self.K = np.zeros((4, 4))

    def copy(self):
        twin = KalmanFilter(self.x.copy(), self.P.copy(), self.Q)
        twin.K = self.K.copy()
        return twin

    def predict(self):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def innovation(self, z, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        return y, S

    def mahalanobis2(self, z, R):
        y, S = self.innovation(z, R)
        return float(y @ np.linalg.solve(S, y))

    def update(self, z, R):
        y, S = self.innovation(z, R)
        self.K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + self.K @ y
        # Joseph form keeps P symmetric positive definite despite rounding
        I_KH = np.eye(4) - self.K @ H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ R @ self.K.T
        return self.x
