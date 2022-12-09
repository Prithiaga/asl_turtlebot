import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch # Switch occurs at t_final - t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state
            t: Current time

        Outputs:
            V, om: Control actions
        """
        # Hint: Both self.traj_controller and self.pose_controller have compute_control() functions.
        #       When should each be called? Make use of self.t_before_switch and
        #       self.traj_controller.traj_times.
        ########## Code starts here ##########
        t_f = self.traj_controller.traj_times[len(self.traj_controller.traj_times) - 1]
        if t < (t_f - self.t_before_switch) :
            V, om = self.traj_controller.compute_control(x, y, th, t)
        else:
            V, om = self.pose_controller.compute_control(x, y, th, t)
        return V, om
        ########## Code ends here ##########

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
    assert(path and k > 2 and k < len(path))
    ########## Code starts here ##########
    # Hint 1 - Determine nominal time for each point in the path using V_des
    # Hint 2 - Use splrep to determine cubic coefficients that best fit given path in x, y
    # Hint 3 - Use splev to determine smoothed paths. The "der" argument may be useful.
    
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

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    Hint: Take a close look at the code within compute_traj_with_limits() and interpolate_traj()
          from P1_differential_flatness.py
    """
    ########## Code starts here ##########
    V, om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)
    finalstate_index = len(traj) - 1
    x_f = traj[finalstate_index, 0]
    y_f = traj[finalstate_index, 1]
    theta_f = traj[finalstate_index, 2]
    s_f = State(x_f, y_f, V_max, theta_f)
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
