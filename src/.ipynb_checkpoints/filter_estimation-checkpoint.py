import numpy as np

from ExtendedKalmanFilter import ExtendedKalmanFilter
from UnscentedKalmanFilter import UnscentedKalmanFilter

from helper_functions import normalize_angles

class Filter_Estimation:
    def __init__(self, x, P):
        self.estimated_state = [x]
        self.estimated_variance = [[P[0,0], P[1,1], P[2,2]]]

    def append_state(self, x_new):
        self.estimated_state.append(x_new)

    def append_variance(self, P_new):
        self.estimated_variance.append([P_new[0,0], P_new[1,1], P_new[2,2]])

    def return_estimated_state(self):
        estimated_state = np.array(self.estimated_state).T
        estimated_state[2, :] = normalize_angles(estimated_state[2, :])
        return estimated_state

def apply_estimation_filter(data, filter, x_0, P_0, R, Q):
    if filter == 'EKF':
        filter = ExtendedKalmanFilter(x_0, P_0)
    elif filter == 'UKF':
        filter = UnscentedKalmanFilter(x_0, P_0)
    else:
        raise Exception("Filter Error: Wrong filter argument")

    estimator = Filter_Estimation(x_0, P_0)
    
    t_last = data.timestamps[0] #
    for t_idx in range(1, data.N):
        t = data.timestamps[t_idx]
        dt = t - t_last
        
        # get control input `u = [v, omega] + noise`
        u = data.noisy_control_inputs[:, t_idx]
        
        # propagate
        filter.propagate(u, dt, R)
        
        # get measurement `z = [x, y] + noise`
        z = data.observation_states[:, t_idx]
        
        # update
        filter.update(z, Q)

        t_last = t
        
        estimator.append_state(filter.state)
        estimator.append_variance(filter.covariance)
        
    return estimator.return_estimated_state()