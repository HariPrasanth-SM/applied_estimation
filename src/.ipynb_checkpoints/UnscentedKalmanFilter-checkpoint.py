import numpy as np
from scipy.linalg import cholesky

from helper_functions import normalize_angles

class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for vehicle motion estimation.
    Motion model: Eq. (5.9) in [1]
    Observes 2D location (x, y).
    """

    def __init__(self, initial_state, initial_covariance, alpha=0.001, beta=2.0, kappa=0.0):
        """ 
        Initialize UKF.

        Args:
            initial_state (numpy.array): Initial state estimate: [x_, y_, theta]^T
            initial_covariance (numpy.array): Initial estimation error covariance
            alpha (float): UKF parameter for tuning spread of sigma points. Default is 0.001.
            beta (float): UKF parameter for incorporating prior knowledge about the distribution. Default is 2.0.
            kappa (float): UKF parameter controlling the distribution of sigma points around the mean. Default is 0.0.
        """
        self.state = initial_state
        self.covariance = initial_covariance
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.state_dimension = len(initial_state)
        self.sigma_points_count = 2 * self.state_dimension + 1
        self.lambda_ = self.alpha**2 * (self.state_dimension + self.kappa) - self.state_dimension

        # Initialize weights for mean and covariance calculations
        self.weights_mean = np.zeros(self.sigma_points_count)
        self.weights_covariance = np.zeros(self.sigma_points_count)
        self.weights_mean[0] = self.lambda_ / (self.state_dimension + self.lambda_)
        self.weights_covariance[0] = self.lambda_ / (self.state_dimension + self.lambda_) + (1 - self.alpha**2 + self.beta)

        for i in range(1, self.sigma_points_count):
            self.weights_mean[i] = 1 / (2 * (self.state_dimension + self.lambda_))
            self.weights_covariance[i] = 1 / (2 * (self.state_dimension + self.lambda_))

    def sigma_points(self):
        """
        Calculate sigma points.

        Returns:
            numpy.array: Matrix of sigma points.
        """
        sigma_points = np.zeros((self.state_dimension, self.sigma_points_count))
        sqrt_covariance = cholesky((self.state_dimension + self.lambda_) * self.covariance)  # Square root of the matrix

        # First sigma point is the mean
        sigma_points[:, 0] = self.state

        # Calculate other sigma points
        for i in range(self.state_dimension):
            sigma_points[:, i + 1] = self.state + sqrt_covariance[:, i]
            sigma_points[:, i + 1 + self.state_dimension] = self.state - sqrt_covariance[:, i]

        return sigma_points

    def predict_sigma_points(self, sigma_points, control_input, time_interval):
        """
        Predict sigma points using the state transition model.

        Args:
            sigma_points (numpy.array): Matrix of sigma points.
            control_input (tuple): Control input (forward_velocity, yaw_rate).
            time_interval (float): Time interval.

        Returns:
            numpy.array: Predicted sigma points.
        """
        self.predicted_sigma_points = np.zeros((self.state_dimension, self.sigma_points_count))

        for i in range(self.sigma_points_count):
            x, y, theta = sigma_points[:, i]
            v, omega = control_input
            r = v / omega if abs(omega) > 1e-6 else 0.0  # Turning radius

            dtheta = omega * time_interval
            dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
            dy = r * np.cos(theta) - r * np.cos(theta + dtheta)
            
            self.predicted_sigma_points[:, i] = [x + dx, y + dy, theta + dtheta]

        return self.predicted_sigma_points

    def predict_mean_covariance(self, predicted_sigma_points):
        """
        Predict mean and covariance from predicted sigma points.

        Args:
            predicted_sigma_points (numpy.array): Predicted sigma points.

        Returns:
            tuple: Predicted state and covariance.
        """
        predicted_state = np.dot(predicted_sigma_points, self.weights_mean)
        residual = predicted_sigma_points - predicted_state[:, np.newaxis]
        predicted_covariance = residual @ np.diag(self.weights_covariance) @ residual.T

        return predicted_state, predicted_covariance

    def update(self, observation, observation_covariance):
        """
        Update state and covariance based on observation of (x_, y_).

        Args:
            observation (numpy.array): Observation vector [x_, y_].
            observation_covariance (numpy.array): Covariance matrix of the observation.
        """
        updated_sigma_points = self.sigma_points()

        # Extract the first two elements (x, y) as the observation model
        predicted_observation_sigma_points = updated_sigma_points[:2, :]
        predicted_observation, predicted_observation_covariance = self.predict_mean_covariance(predicted_observation_sigma_points)
        predicted_observation_covariance = predicted_observation_covariance + observation_covariance

        cross_covariance = np.dot(self.predicted_sigma_points - self.state[:, np.newaxis], np.diag(self.weights_covariance)) @ (predicted_observation_sigma_points - predicted_observation[:, np.newaxis]).T
    
        kalman_gain = cross_covariance @ np.linalg.inv(predicted_observation_covariance) 
        self.state = self.state + kalman_gain @ (observation - predicted_observation)  # Kalman gain @ innovation
        self.covariance = self.covariance - kalman_gain @ predicted_observation_covariance @ kalman_gain.T

    def propagate(self, control_input, time_interval, transition_noise_covariance):
        """
        Propagate state and covariance based on state transition model defined as Eq. (5.9) in [1].

        Args:
            control_input (tuple): Control input (forward_velocity, yaw_rate).
            time_interval (float): Time interval.
            transition_noise_covariance (numpy.array): Covariance matrix of the transition noise.
        """
        sigma_points = self.sigma_points()
        predicted_sigma_points = self.predict_sigma_points(sigma_points, control_input, time_interval)
        predicted_state, predicted_covariance = self.predict_mean_covariance(predicted_sigma_points)

        self.state = predicted_state
        self.covariance = predicted_covariance + transition_noise_covariance
