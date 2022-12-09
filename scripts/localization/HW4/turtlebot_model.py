import numpy as np

EPSILON_OMEGA = 1e-3

def compute_Gx(xvec, u, dt):
    """
    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
    Outputs:
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
    """
    ########## Code starts here ##########
    # TODO: Compute Gx
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    (x, y, theta) = xvec
    (V, om) = u
    if abs(om) > EPSILON_OMEGA:
        theta_t = theta + om * dt
        dx_dth = (V / om) * (np.cos(theta_t) - np.cos(theta))
        dy_dth = (V / om) * (np.sin(theta_t) - np.sin(theta))
        
    else:
        dx_dth = - V * dt * np.sin(theta)
        dy_dth = V * dt * np.cos(theta)
    
    Gx = np.array([
        [1, 0 , dx_dth],
        [0, 1, dy_dth],
        [0, 0, 1]
    ], dtype=float)
        
    ########## Code ends here ##########
    return Gx
    

def compute_Gu(xvec, u, dt):
    """
    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
    Outputs:
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute Gu
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    (x, y, theta) = xvec
    (V, om) = u
    if abs(om) > EPSILON_OMEGA:
        theta_t = theta + om * dt
        dx_dV = 1 / om * (np.sin(theta_t) - np.sin(theta))
        dy_dV = -1 / om * (np.cos(theta_t) - np.cos(theta))
        dx_dom = (V / (om ** 2)) * ( om * dt * np.cos(theta_t) - np.sin(theta_t) + np.sin(theta))
        # original = dy_dom = (V / (om ** 2)) * ( om * dt * np.sin(theta_t) + np.cos(theta_t) - np.sin(theta))
        dy_dom = (V / (om ** 2)) * ( om * dt * np.sin(theta_t) + np.cos(theta_t) - np.cos(theta))
        
    else:
        dx_dV = dt * np.cos(theta)
        dy_dV = dt * np.sin(theta)
        dx_dom = - (V / 2) * (dt ** 2) * np.sin(theta)      #dx(xdot)*dt
        dy_dom = (V / 2) * (dt ** 2) * np.cos(theta)        #ydotdot*dt
    
    Gu = np.array([
        [dx_dV, dx_dom],
        [dy_dV, dy_dom],
        [0, dt]
    ], dtype=float)    
    ########## Code ends here ##########
    return Gu


def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    (x, y, theta) = xvec
    (V, om) = u
    if abs(om) > EPSILON_OMEGA:
        theta_t = theta + om * dt
        x_t = x + (V / om) * (np.sin(theta_t) - np.sin(theta))
        y_t = y - (V / om) * (np.cos(theta_t) - np.cos(theta))
    else:
        theta_t = theta + om * dt
        x_t = x + (V * dt * np.cos(theta_t))
        y_t = y + (V * dt * np.sin(theta_t))
        
    g = np.array([x_t, y_t, theta_t])
    
    Gx = compute_Gx(xvec, u, dt)
    Gu = compute_Gu(xvec, u, dt)

    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)
    x_bw, y_bw, th_bw = x
    x_cb, y_cb, th_cb = tf_base_to_camera

    x_cw = x_bw + x_cb*np.cos(th_bw) - y_cb*np.sin(th_bw)
    y_cw = y_bw + x_cb*np.sin(th_bw) + y_cb*np.cos(th_bw)
    th_cw = th_bw + th_cb

    a_c = alpha - th_cw
    r_c = r - x_cw*np.cos(alpha)-y_cw*np.sin(alpha)

    h = np.array([a_c, r_c])
    da_dx = 0
    da_dy = 0
    da_dth = -1
    dr_dx = -np.cos(alpha)
    dr_dy = -np.sin(alpha)
    dr_dth = -(-x_cb*np.sin(th_bw)-y_cb*np.cos(th_bw))*np.cos(alpha) - (x_cb*np.cos(th_bw)-y_cb*np.sin(th_bw))*np.sin(alpha)

    Hx = np.array([[da_dx, da_dy, da_dth], [dr_dx, dr_dy, dr_dth]])
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
