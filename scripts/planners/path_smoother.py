import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, k, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        k (int): The degree of the spline fit.
            For this assignment, k should equal 3 (see documentation for
            scipy.interpolate.splrep)
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        t_smoothed (np.array [N]): Associated trajectory times
        traj_smoothed (np.array [N,7]): Smoothed trajectory
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    # Nominal Time for each point in the path using V_des
    nominal_times = []
    for index, point in enumerate(path):
        if index == 0:
            nominal_times.append(0)
        else:
            distance = np.linalg.norm(np.array(point) - np.array(path[index - 1]))
            curr_nominal_time = distance / V_des
            nominal_times.append(nominal_times[index - 1] + curr_nominal_time)

    #print(type(path))
    #splrep to determine cubic coefficients that best fit given path in x, y
    #tck = scipy.interpolate.splprep(np.array(path)[:,0], np.array(path)[:,1], k=k, s = alpha)
    tck_x = scipy.interpolate.splrep(nominal_times, np.array(path)[:,0], k=k, s=alpha)
    tck_y = scipy.interpolate.splrep(nominal_times, np.array(path)[:,1], k=k, s=alpha)
    
    #Use splev to determine smoothed paths
        
    t_smoothed_len = len(nominal_times)
    t_smoothed = np.arange(0, nominal_times[t_smoothed_len - 1], dt)
    # x_d, y_d = scipy.interpolate.splev(t_smoothed, tck)
    # print("xd yd: ", x_d, y_d)
    
    x_d = scipy.interpolate.splev(t_smoothed, tck_x)       
    y_d = scipy.interpolate.splev(t_smoothed, tck_y)
    xd_d = scipy.interpolate.splev(t_smoothed, tck_x, der = 1)
    yd_d = scipy.interpolate.splev(t_smoothed, tck_y, der = 1)
    xdd_d = scipy.interpolate.splev(t_smoothed, tck_x, der = 2)
    ydd_d = scipy.interpolate.splev(t_smoothed, tck_y, der = 2)
    theta_d = np.arctan2(yd_d, xd_d)
    #print("xd yd final: ", x_d, y_d)
    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return t_smoothed, traj_smoothed
