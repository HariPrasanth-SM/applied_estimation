# references: 
# [1] https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf
import numpy as np

from helper_functions import normalize_angles

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for vehicle motion estimation.
    Motion model: Eq. (5.9) in [1]
    Observes 2D location (x, y).
    """

    def __init__(self, initial_state, initial_covariance):
        """ 
        Initialize EKF.

        Args:
            initial_state (numpy.array): Initial state estimate: [x_, y_, theta]^T
            initial_covariance (numpy.array): Initial estimation error covariance
        """
        self.state = initial_state
        self.covariance = initial_covariance

    def update(self, observation, observation_covariance):
        """Update state and covariance based on observation of (x_, y_).

        Args:
            observation (numpy.array): Observation for [x_, y_]^T
            observation_covariance (numpy.array): Observation noise covariance
        """
        # Compute Kalman gain
        H = np.array([
            [1., 0., 0.],
            [0., 1., 0.]
        ])  # Jacobian of observation function

        K = self.covariance @ H.T @ np.linalg.inv(H @ self.covariance @ H.T + observation_covariance)

        # Update state
        x, y, theta = self.state
        predicted_observation = np.array([x, y])  # Expected observation from the estimated state
        self.state = self.state + K @ (observation - predicted_observation)

        # Update covariance
        self.covariance = self.covariance - K @ H @ self.covariance

    def propagate(self, control_input, time_interval, transition_noise_covariance):
        """Propagate state and covariance based on state transition model defined as Eq. (5.9) in [1].

        Args:
            control_input (numpy.array): Control input: [v, omega]^T
            time_interval (float): Time interval in seconds
            transition_noise_covariance (numpy.array): State transition noise covariance
        """
        # Propagate state
        x, y, theta = self.state
        v, omega = control_input
        r = v / omega if abs(omega) > 1e-6 else 0.0  # Turning radius

        dtheta = omega * time_interval
        dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
        dy = r * np.cos(theta) - r * np.cos(theta + dtheta)

        self.state += np.array([dx, dy, dtheta])

        # Propagate covariance
        G = np.array([
            [1., 0., - r * np.cos(theta) + r * np.cos(theta + dtheta)],
            [0., 1., - r * np.sin(theta) + r * np.sin(theta + dtheta)],
            [0., 0., 1.]
        ])  # Jacobian of state transition function

        self.covariance = G @ self.covariance @ G.T + transition_noise_covariance
