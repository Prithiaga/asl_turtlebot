import numpy as np
import scipy.linalg  # you may find scipy.linalg.block_diag useful

import os
from . import turtlebot_model as tb

class Ekf(object):
    """
    Base class for EKF Localization and SLAM.

    Usage:
        ekf = EKF(x0, Sigma0, R)
        while True:
            ekf.transition_update(u, dt)
            ekf.measurement_update(z, Q)
            localized_state = ekf.x
    """

    def __init__(self, x0, Sigma0, R):
        """
        EKF constructor.

        Inputs:
                x0: np.array[n,]  - initial belief mean.
            Sigma0: np.array[n,n] - initial belief covariance.
                 R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.x = x0  # Gaussian belief mean
        self.Sigma = Sigma0  # Gaussian belief covariance
        self.R = R  # Control noise covariance (corresponding to dt = 1 second)

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating (self.x, self.Sigma).

        Inputs:
             u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.x, self.Sigma) should be updated.
        """
        g, Gx, Gu = self.transition_model(u, dt)

        ########## Code starts here ##########
        # TODO: Update self.x, self.Sigma.
        self.x = g
        currSigma = self.Sigma
        currR = self.R
        self.Sigma = np.matmul(Gx, np.matmul(currSigma, Gx.T)) + dt * np.matmul(Gu, np.matmul(currR, Gu.T))
        ########## Code ends here ##########

    def transition_model(self, u, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
             u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Outputs:
             g: np.array[n,]  - result of belief mean propagated according to the
                                system dynamics with control u for dt seconds.
            Gx: np.array[n,n] - Jacobian of g with respect to belief mean self.x.
            Gu: np.array[n,2] - Jacobian of g with respect to control u.
        """
        raise NotImplementedError(
            "transition_model must be overriden by a subclass of EKF"
        )

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[I,2]   - matrix of I rows containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) row of z_raw.
        Output:
            None - internal belief state (self.x, self.Sigma) should be updated.
        """
        z, Q, H = self.measurement_model(z_raw, Q_raw)
        if z is None:
            # Don't update if measurement is invalid
            # (e.g., no line matches for line-based EKF localization)
            return

        ########## Code starts here ##########
        # TODO: Update self.x, self.Sigma.
        S = H @ self.Sigma @ H.T + Q
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ z
        self.Sigma = self.Sigma - K @ S @ K.T
        ########## Code ends here ##########

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction). Also returns the associated Jacobian for EKF
        linearization.

        Inputs:
            z_raw: np.array[I,2]   - I lines extracted from scanner data in
                                     rows representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[2K,]   - measurement mean.
            Q: np.array[2K,2K] - measurement covariance.
            H: np.array[2K,n]  - Jacobian of z with respect to the belief mean self.x.
        """
        raise NotImplementedError(
            "measurement_model must be overriden by a subclass of EKF"
        )


class EkfLocalization(Ekf):
    """
    EKF Localization.
    """

    def __init__(self, x0, Sigma0, R, map_lines, tf_base_to_camera, g):
        """
        EkfLocalization constructor.

        Inputs:
                       x0: np.array[3,]  - initial belief mean.
                   Sigma0: np.array[3,3] - initial belief covariance.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
                map_lines: np.array[J,2] - J map lines in rows representing (alpha, r).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.map_lines = (
            map_lines  # Matrix of J map lines with (alpha, r) as rows
        )
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, Sigma0, R)

    def transition_model(self, u, dt):
        """
        Turtlebot dynamics (unicycle model).
        """

        ########## Code starts here ##########
        # TODO: Compute g, Gx, Gu using tb.compute_dynamics().
        g, Gx, Gu = tb.compute_dynamics(self.x, u, dt)

        ########## Code ends here ##########

        return g, Gx, Gu

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature.
        """
        v_list, Q_list, H_list = self.compute_innovations(z_raw, Q_raw)
        if not v_list:
            print(
                "Scanner sees {} lines but can't associate them with any map entries.".format(
                    z_raw.shape[0]
                )
            )
            return None, None, None

        ########## Code starts here ##########
        # TODO: Compute z, Q.
        # HINT: The scipy.linalg.block_diag() function may be useful.
        # HINT: A list can be unpacked using the * (splat) operator.
        z = np.array(v_list).flatten() # [2K]
        Q = scipy.linalg.block_diag(*Q_list) # [2K, 2K]
        H = np.array(H_list).reshape(-1, 3) # [2K, n]

        assert(z.shape[0] == Q.shape[0] == Q.shape[1] == H.shape[0])

        ########## Code ends here ##########

        return z, Q, H

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[I,2]   - I lines extracted from scanner data in
                                     rows representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) row of z_raw.
        Outputs:
            v_list: [np.array[2,]]  - list of at most I innovation vectors
                                      (predicted map measurement - scanner measurement).
            Q_list: [np.array[2,2]] - list of covariance matrices of the
                                      innovation vectors (from scanner uncertainty).
            H_list: [np.array[2,3]] - list of Jacobians of the innovation
                                      vectors with respect to the belief mean self.x.
        """

        def angle_diff(a, b):
            a = a % (2.0 * np.pi)
            b = b % (2.0 * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2.0 * (diff < 0.0) - 1.0
                    diff += sign * 2.0 * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2.0 * (diff[idx] < 0.0) - 1.0
                diff[idx] += sign * 2.0 * np.pi
            return diff

        hs, Hs = self.compute_predicted_measurements()

        ########## Code starts here ##########
        # TODO: Compute v_list, Q_list, H_list
        # HINT: hs contains the J predicted lines, z_raw contains the I observed lines
        # HINT: To calculate the innovation for alpha, use angle_diff() instead of plain subtraction
        # HINT: Optionally, efficiently calculate all the innovations in a matrix V of shape [I, J, 2]. np.expand_dims() and np.dstack() may be useful.
        # HINT: For each of the I observed lines, 
        #       find the closest predicted line and the corresponding minimum Mahalanobis distance
        #       if the minimum distance satisfies the gating criteria, add corresponding entries to v_list, Q_list, H_list

        v_list = []
        Q_list = []
        H_list = []
        I, J = z_raw.shape[0], hs.shape[0]
    
        
        for i in range(I):
            d_best = np.Inf
            for j in range(J):
                v = z_raw[i] - hs[j]
                S = Hs[j]@self.Sigma@(Hs[j].T) + Q_raw[i]
                d_temp = v.T@np.linalg.inv(S)@v
                if d_temp < d_best:
                    d_best = d_temp
                    Q_best = Q_raw[i]
                    v_best = v
                    H_best = Hs[j]

            # note: gate after inner loop, make sure to find minimum mal. dist before considering gate
            if d_best < self.g**2:
                v_list.append(v_best)
                Q_list.append(Q_best)
                H_list.append(H_best)


        ########## Code ends here ##########

        return v_list, Q_list, H_list

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Outputs:
                 hs: np.array[J,2]  - J line parameters in the scanner (camera) frame.
            Hx_list: [np.array[2,3]] - list of Jacobians of h with respect to the belief mean self.x.
        """
        hs = np.zeros_like(self.map_lines)
        Hx_list = []
        for j in range(self.map_lines.shape[0]):
            ########## Code starts here ##########
            # TODO: Compute h, Hx using tb.transform_line_to_scanner_frame() for the j'th map line.
            # HINT: This should be a single line of code.


            """
            def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):

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
            alpha, r = line
            """


            h, Hx = tb.transform_line_to_scanner_frame(self.map_lines[j], self.x, self.tf_base_to_camera,compute_jacobian=True)
            ########## Code ends here ##########

            h, Hx = tb.normalize_line_parameters(h, Hx)
            hs[j, :] = h
            Hx_list.append(Hx)

        return hs, Hx_list


class EkfSlam(Ekf):
    """
    EKF SLAM.
    """

    def __init__(self, x0, Sigma0, R, tf_base_to_camera, g):
        """
        EKFSLAM constructor.

        Inputs:
                       x0: np.array[3+2J,]     - initial belief mean.
                   Sigma0: np.array[3+2J,3+2J] - initial belief covariance.
                        R: np.array[2,2]       - control noise covariance
                                                 (corresponding to dt = 1 second).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, Sigma0, R)

    def transition_model(self, u, dt):
        """
        Combined Turtlebot + map dynamics.
        Adapt this method from EkfLocalization.transition_model().
        """
        g = np.copy(self.x)
        Gx = np.eye(self.x.size)
        Gu = np.zeros((self.x.size, 2))
        

        ########## Code starts here ##########
        # TODO: Compute g, Gx, Gu.
        # HINT: This should be very similar to EkfLocalization.transition_model() and take 1-5 lines of code.
        # HINT: Call tb.compute_dynamics() with the correct elements of self.x
        g_rob, Gx_rob, Gu_rob = tb.compute_dynamics(self.x[:3],u,dt)
        g[0:3] = g_rob
        Gx[0:3,0:3] = Gx_rob
        Gu[0:3,0:2] = Gu_rob


        ########## Code ends here ##########

        return g, Gx, Gu

    def measurement_model(self, z_raw, Q_raw):
        """
        Combined Turtlebot + map measurement model.
        Adapt this method from EkfLocalization.measurement_model().

        The ingredients for this model should look very similar to those for
        EkfLocalization. In particular, essentially the only thing that needs to
        change is the computation of Hx in self.compute_predicted_measurements()
        and how that method is called in self.compute_innovations() (i.e.,
        instead of getting world-frame line parameters from self.map_lines, you
        must extract them from the state self.x).
        """
        v_list, Q_list, H_list = self.compute_innovations(z_raw, Q_raw)
        if not v_list:
            print(
                "Scanner sees {} lines but can't associate them with any map entries.".format(
                    z_raw.shape[0]
                )
            )
            return None, None, None

        ########## Code starts here ##########
        # TODO: Compute z, Q, H.
        # Hint: Should be identical to EkfLocalization.measurement_model().
        z = np.array(v_list).reshape(-1,1).flatten() # [2K]
        Q = scipy.linalg.block_diag(*Q_list) # [2K, 2K]
        H = np.array(H_list).reshape(-1, len(self.x)) # [2K, n]


        ########## Code ends here ##########

        return z, Q, H

    def compute_innovations(self, z_raw, Q_raw):
        """
        Adapt this method from EkfLocalization.compute_innovations().
        """

        def angle_diff(a, b):
            a = a % (2.0 * np.pi)
            b = b % (2.0 * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2.0 * (diff < 0.0) - 1.0
                    diff += sign * 2.0 * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2.0 * (diff[idx] < 0.0) - 1.0
                diff[idx] += sign * 2.0 * np.pi
            return diff

        hs, Hs = self.compute_predicted_measurements()

        ########## Code starts here ##########
        # TODO: Compute v_list, Q_list, H_list.
        # HINT: Should be almost identical to EkfLocalization.compute_innovations(). What is J now?
        # HINT: Instead of getting world-frame line parameters from self.map_lines, you must extract them from the state self.x.
        v_list = []
        Q_list = []
        H_list = []

        I, J = z_raw.shape[0], hs.shape[0]
        
        for i in range(I):
            d_best = np.Inf
            for j in range(J):
                v = np.array([angle_diff(z_raw[i][0],hs[j][0]),z_raw[i][1]-hs[j][1]])
                S = Hs[j]@self.Sigma@(Hs[j].T) + Q_raw[i]
                d_temp = v.T@np.linalg.inv(S)@v
                if d_temp < d_best:
                    d_best = d_temp
                    Q_best = Q_raw[i]
                    v_best = v
                    H_best = Hs[j]

            # note: gate after inner loop, make sure to find minimum mal. dist before considering gate
            if d_best < self.g**2:
                v_list.append(v_best)
                Q_list.append(Q_best)
                H_list.append(H_best)

        ########## Code ends here ##########

        return v_list, Q_list, H_list

    def compute_predicted_measurements(self):
        """
        Adapt this method from EkfLocalization.compute_predicted_measurements().
        """
        J = (self.x.size - 3) // 2
        hs = np.zeros((J, 2))
        Hx_list = []
        for j in range(J):
            idx_j = 3 + 2 * j
            alpha, r = self.x[idx_j : idx_j + 2]

            Hx = np.zeros((2, self.x.size))
            line = (alpha,r)

            ########## Code starts here ##########
            # TODO: Compute h, Hx.
            # HINT: Call tb.transform_line_to_scanner_frame() for the j'th map line.
            # HINT: The first 3 columns of Hx should be populated using the same approach as in EkfLocalization.compute_predicted_measurements().
            # HINT: The first two map lines (j=0,1) are fixed so the Jacobian of h wrt the alpha and r for those lines is just 0.
            # HINT: For the other map lines (j>2), write out h in terms of alpha and r to get the Jacobian Hx.

            # First two map lines are assumed fixed so we don't want to propagate
            # any measurement correction to them.
            h, Hx[0:2,0:3] = tb.transform_line_to_scanner_frame(line, self.x[0:3], self.tf_base_to_camera,compute_jacobian=True)
            x_b, y_b, th_b = self.x[0:3]
            x_cam, y_cam, th_cam = self.tf_base_to_camera
            cam_array = np.array([x_cam,y_cam,1])
            r_t_transform = np.array([[np.cos(th_b), -np.sin(th_b), x_b],
                                          [np.sin(th_b),  np.cos(th_b), y_b], 
                                          [0,0,1]])
    
            x_cam_w, y_cam_w, th_cam_w = r_t_transform.dot(cam_array)
            if j >= 2:
                Hx[:,idx_j:idx_j+2] = np.eye(2)  # FIX ME!
                Hx[1,idx_j] = x_cam_w*np.sin(alpha) - y_cam_w*np.cos(alpha)

            ########## Code ends here ##########

            h, Hx = tb.normalize_line_parameters(h, Hx)
            hs[j, :] = h
            Hx_list.append(Hx)

        return hs, Hx_list
