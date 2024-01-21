import numpy as np
import pykitti
import sys
import matplotlib.pyplot as plt

from helper_functions import *


class LoadKittiData:
    """
    Load the data from rawfile
    """
    def __init__(self, path_to_dir, date, id):
        self.ground_truth_states = None
        
        self.dataset_pack = pykitti.raw(path_to_dir, date, id)
        self.dataset = self.dataset_pack.oxts
        self.N = len(self.dataset)

        timestamps = np.array(self.dataset_pack.timestamps)
        elapsed = np.array(timestamps) - timestamps[0]
        self.timestamps = np.array([t.total_seconds() for t in elapsed])
        
        self.find_ground_truth_states()

    def find_ground_truth_states(self):
        self.unpack_data()
        self.convert_lla_to_enu()

        self.ground_truth_states = self.trajectory_xyz.copy()
        self.ground_truth_states[2, :] = self.theta

    def unpack_data(self):
        self.trajectory_lla = list()
        self.theta = list()
        self.velocity = list() # Forward (or Linear) velocity
        self.omega = list() # Angular velocity

        for oxts_data in self.dataset:
            packet = oxts_data.packet
            self.trajectory_lla.append([
                packet.lon,
                packet.lat,
                packet.alt
            ])
            self.theta.append(packet.yaw)
            self.velocity.append(packet.vf)
            self.omega.append(packet.wz)

        # Converting python lists to numpy arrays
        self.trajectory_lla = np.array(self.trajectory_lla).T
        self.theta = np.array(self.theta)
        self.velocity = np.array(self.velocity)
        self.omega = np.array(self.omega)

    def convert_lla_to_enu(self, plot=True):
        self.origin = self.trajectory_lla[:, 0]
        self.trajectory_xyz = lla_to_enu(self.trajectory_lla, self.origin)

    def plot_ground_truth_data(self):
        fig, ax = plt.subplots(1, 1, figsize=(12,9))
        ax.plot(self.ground_truth_states[0, :], self.ground_truth_states[1, :], lw=2, label='ground-truth')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid()
        ax.legend()
        #fig.show()
        
        fig, ax = plt.subplots(1, 3, figsize=(12,4))
        
        ax[0].plot(self.timestamps, self.ground_truth_states[0, :], lw=1, label='ground-truth')
        ax[0].set_xlabel('time elapsed [sec]')
        ax[0].set_ylabel('x [m]')
        ax[0].legend()

        ax[1].plot(self.timestamps, self.ground_truth_states[1, :], lw=1, label='ground-truth')
        ax[1].set_xlabel('time elapsed [sec]')
        ax[1].set_ylabel('y [m]')
        ax[1].legend()

        ax[2].plot(self.timestamps, self.ground_truth_states[2, :], lw=1, label='ground-truth')
        ax[2].set_xlabel('time elapsed [sec]')
        ax[2].set_ylabel('yaw angle [rad]')
        ax[2].legend()

    def add_gaussian_noise_observation(self, obs_noise_std=[5, 5, np.pi]):
        self.observation_states = self.ground_truth_states[:2, :].copy()
        
        x_obs_noise = np.random.normal(0.0, obs_noise_std[0], (self.N))  # generate gaussian noise
        self.observation_states[0, :] += x_obs_noise  # add the noise to ground-truth positions

        y_obs_noise = np.random.normal(0.0, obs_noise_std[1], (self.N))  # generate gaussian noise
        self.observation_states[1, :] += y_obs_noise  # add the noise to ground-truth positions

        #theta_obs_noise = np.random.normal(0.0, obs_noise_std[2], (self.N))  # gen gaussian noise
        #self.observation_states[2, :] += theta_obs_noise  # add the noise to ground-truth positions
        #self.observation_states[2, :] = normalize_angles(self.observation_states[2, :])

    def add_gaussian_noise_control_input(self, input_noise_std=[0.3, 0.02]):
        self.control_inputs = np.zeros((2, self.N))
        self.control_inputs[0, :] = self.velocity
        self.control_inputs[1, :] = self.omega

        self.noisy_control_inputs = self.control_inputs.copy()
        
        velocity_inputs_noise = np.random.normal(0.0, input_noise_std[0], (self.N))  # generate gaussian noise
        self.noisy_control_inputs[0, :] += velocity_inputs_noise  # add the noise to velocity (control input [0])

        omega_inputs_noise = np.random.normal(0.0, input_noise_std[1], (self.N))  # generate gaussian noise
        self.noisy_control_inputs[1, :] += omega_inputs_noise  # add the noise to omega (control input [1])

    def plot_observation_data(self):
        fig, ax = plt.subplots(1, 1, figsize=(12,9))
        ax.plot(self.ground_truth_states[0, :], self.ground_truth_states[1, :], lw=2, label='ground-truth')
        ax.plot(self.observation_states[0, :], self.observation_states[1, :], lw=0, marker='.', markersize=5, alpha=0.4, label='observed')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid()
        ax.legend()
        
        fig, ax = plt.subplots(1, 3, figsize=(12,4))
        
        ax[0].plot(self.timestamps, self.ground_truth_states[0, :], lw=1, label='ground-truth')
        ax[0].plot(self.timestamps, self.observation_states[0, :], lw=0, marker='.', alpha=0.4, label='observed')
        ax[0].set_xlabel('time elapsed [sec]')
        ax[0].set_ylabel('x [m]')
        ax[0].legend()

        ax[1].plot(self.timestamps, self.ground_truth_states[1, :], lw=1, label='ground-truth')
        ax[1].plot(self.timestamps, self.observation_states[1, :], lw=0, marker='.', alpha=0.4, label='observed')
        ax[1].set_xlabel('time elapsed [sec]')
        ax[1].set_ylabel('y [m]')
        ax[1].legend()

        ax[2].plot(self.timestamps, self.ground_truth_states[2, :], lw=1, label='ground-truth')
        #ax[2].plot(self.timestamps, self.observation_states[2, :], lw=0, marker='.', alpha=0.4, label='observed')
        ax[2].set_xlabel('time elapsed [sec]')
        ax[2].set_ylabel('yaw angle [rad]')
        ax[2].legend()

    def plot_control_input_data(self):
        fig, ax = plt.subplots(1, 2, figsize=(12,4))
        
        ax[0].plot(self.timestamps, self.control_inputs[0, :], lw=1, label='velocity')
        ax[0].plot(self.timestamps, self.noisy_control_inputs[0, :], lw=0, marker='.', alpha=0.4, label='noisy velocity')
        ax[0].set_xlabel('time elapsed [sec]')
        ax[0].set_ylabel('v [m/s]')
        ax[0].legend()

        ax[1].plot(self.timestamps, self.control_inputs[1, :], lw=1, label='omega')
        ax[1].plot(self.timestamps, self.noisy_control_inputs[1, :], lw=0, marker='.', alpha=0.4, label='noisy omega')
        ax[1].set_xlabel('time elapsed [sec]')
        ax[1].set_ylabel('omega [rad/sec]')
        ax[1].legend()

    def plot_estimated_data(self, estimated_states):
        fig, ax = plt.subplots(1, 1, figsize=(12,9))
        ax.plot(self.ground_truth_states[0, :], self.ground_truth_states[1, :], lw=2, label='ground-truth')
        ax.plot(self.observation_states[0, :], self.observation_states[1, :], lw=0, marker='.', markersize=5, alpha=0.4, label='observed')
        ax.plot(estimated_states[0, :], estimated_states[1, :], lw=2, label='estimated trajectory', color='r')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid()
        ax.legend()
        
        fig, ax = plt.subplots(1, 3, figsize=(12,4))
        
        ax[0].plot(self.timestamps, self.ground_truth_states[0, :], lw=1, label='ground-truth')
        ax[0].plot(self.timestamps, self.observation_states[0, :], lw=0, marker='.', alpha=0.4, label='observed')
        ax[0].plot(self.timestamps, estimated_states[0, :], lw=1, label='estimated', color='r')
        ax[0].set_xlabel('time elapsed [sec]')
        ax[0].set_ylabel('x [m]')
        ax[0].legend()

        ax[1].plot(self.timestamps, self.ground_truth_states[1, :], lw=1, label='ground-truth')
        ax[1].plot(self.timestamps, self.observation_states[1, :], lw=0, marker='.', alpha=0.4, label='observed')
        ax[1].plot(self.timestamps, estimated_states[1, :], lw=1, label='estimated', color='r')
        ax[1].set_xlabel('time elapsed [sec]')
        ax[1].set_ylabel('y [m]')
        ax[1].legend()

        ax[2].plot(self.timestamps, self.ground_truth_states[2, :], lw=1, label='ground-truth')
        #ax[2].plot(self.timestamps, self.observation_states[2, :], lw=0, marker='.', alpha=0.4, label='observed')
        ax[2].plot(self.timestamps, estimated_states[2, :], lw=1, label='estimated', color='r')
        ax[2].set_xlabel('time elapsed [sec]')
        ax[2].set_ylabel('yaw angle [rad]')
        ax[2].legend()

    def evaluate_filter(self, estimated_states):
        if self.ground_truth_states.shape != estimated_states.shape:
            raise "Error: Dimensions of ground truth and estmated values do not match"

        error = np.subtract(self.ground_truth_states, estimated_states)
        error[2, :] = normalize_angles(error[2, :])
        abs_error = np.abs(error)
        squared_error = error ** 2

        rmse = np.sqrt(np.mean(squared_error, axis=1))
        mae = np.mean(abs_error, axis=1)
        mse = np.mean(squared_error, axis=1)

        # Normalised Root Mean Squared Error (NRMSE)
        range_gt = np.max(self.ground_truth_states) - np.min(self.ground_truth_states)
        nrmse = rmse / range_gt

        mean_error = np.mean(error, axis=1)
        std_error = np.std(error, axis=1)

        mean_angular_error = np.mean(error[2])
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MSE': mse,
            'NRMSE': nrmse,
            'Mean_Error': mean_error,
            'Std_Error': std_error,
            'Mean_Angular_Error': mean_angular_error
        }