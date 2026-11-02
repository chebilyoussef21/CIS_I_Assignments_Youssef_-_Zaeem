"""
The calibrated kinematic model: ImprovedFwdKins / ImprovedInvKins.
"""

from typing import Dict, Optional

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame

from ..robot import Robot
from .workflow import CalibrationData

from uncertainty_networks.se3 import  log_so3, exp_se3, log_se3, exp_so3
from scipy.optimize import least_squares

class ImprovedKinematics(Robot):
    """
    TODO: implement -- the core deliverable.

    A Robot whose fwd_kins / inv_kins account for the calibrated error
    model. You choose the parameterization; the handout's error model is
        theta*_k = theta_k + delta_theta_k,sys(theta_k)   (all joints; ||.|| <= delta_sys, slope <= omega_sys)
        d*_k = R(phi_k) d_k                                (k = 2..6, ||phi_k|| <= phi_max; d*_1 = d_1)
        F*_we = F_we dF_ee                                 (||alpha_ee|| <= 0.01, ||epsilon_ee|| <= 0.5)
    plus the constant base frame F_X relating the robot base to the
    tracker (see workflow.estimate_base_frame). HomeRobot() gives you
    delta_theta_sys at home directly (workflow.measure_home_offsets); pass
    it as joint_offsets. Every parameter you include must be identifiable
    from your data -- look at the singular values of the Jacobian.

    A parameter vector that covers this model, for example:
        params[0:6]     xi_X          twist of the base frame about F_X_init, F_X = F_X_init * Exp(xi_X)
        params[6:16]    (t1, t2) x 5  tilt of d_2, d_3, n_4, n_5, n_6, two components each in the
                                      plane perpendicular to the nominal axis
        params[16:22]   xi_we         twist of the tool frame, F_we = F_WE * Exp(xi_we)
        params[22:]     c_k,j         per-joint distortion polynomial coefficients
                                      delta_theta_k,sys = offset_k + sum_j c_k,j (theta_k / scale_k)^j
    with every entry zero meaning "the nominal robot at F_X_init".

    The nominal Robot's attributes (AXES, F_CW, F_WE, limits, timing,
    wrist_solutions) are inherited and may be reused.
    """

    def __init__(self, robot: Robot, F_X_init: Frame, joint_offsets=None, n_poly: int = 2):
        super().__init__(robot.name + "_improved", robot.F_rob, robot.theta_min, robot.theta_max,
                         robot.t_settle, robot.theta_dot_max, robot.theta_ddot_max)
        self.F_X_init = F_X_init  # starting estimate of the base frame (from the nominal fit)
        self.joint_offsets = np.zeros(6) if joint_offsets is None else np.asarray(joint_offsets, dtype=np.float64).reshape(6)
        self.n_poly = n_poly      # e.g. the degree of a per-joint distortion polynomial
        self.params = self.initial_params()

    # ------------------------------------------------------------------ parameterization

    def initial_params(self) -> np.ndarray:
        """
        TODO: implement.

        output: the parameter vector that makes this model identical to the
                nominal kinematics with base frame F_X_init -- with the
                layout in the class docstring that is simply
                np.zeros(6 + 10 + 6 + 6 * self.n_poly)
        """
        # base-frame correction: we start with by assuming F_X_nom is correct so params[0:6]=0
        # axis tilts: we start by assuming nominal joint axes are correct so params [6:16]=0
        # tool-frame correction: we start by assuming nominal F_we is correct so params[16:22]=0
        # per-joint distortion polynomial coefficients: we start with by assuming that there is no distortion beyond home offsets so params[22:]=0
        
        n_params = 6+10+6+6*self.n_poly
        return np.zeros(n_params)


    def base_frame(self, params: Optional[np.ndarray] = None) -> Frame:
        """
        TODO: implement.

        output: F_X (robot base relative to the tracker) for params (default:
                self.params) -- e.g. F_X_init * Exp(params[0:6]), see
                uFrame.perturb / uncertainty_networks.se3.exp_se3
        """

        #eta = [alpha_x, alpha_y, alpha_z, eps_x, eps_y, eps_z]
        eta = np.zeros(6)

        if params is None: #since it's optional as an argument
            eta = self.params[0:6] #from the initial parameters function so 0 vector
        else:
            eta = params[0:6]
        
        T = exp_se3(eta)  # exp_se3 returns a 4x4 homogeneous transformation matrix in numpy array format
        delta_F = Frame(Rot(matrix=T[:3, :3]), vct3(*T[:3, 3])) # converte numpy array matrix into Frame datatype
        # delta_F = exp_se3(eta) #converts 6D correction vector eta into frame delta_F

        F_X_real = self.F_X_init * delta_F    #frame uncertainty formula to determine actual frame
        return F_X_real

    # ------------------------------------------------------------------ kinematics

    def _corrected_axes(self, params):
        class Tilt:
            def __init__(self, tilt_2_component_array):
                self.tilt_1 = tilt_2_component_array[0]
                self.tilt_2 = tilt_2_component_array[1]
    
        tilts = []
        TILT_PARAMS_IDX_OFFSET = 6
        for i in range (5):
            idx = TILT_PARAMS_IDX_OFFSET + 2*i
            tilts.append(Tilt(params[idx : idx+2]))

        # nominal axes d1 ... d6 (since they rely on x,y,z axis we will just create 3 axes and use them twice for each set of axis d1-d3 and d4-d6)
        dx = np.array([1.0, 0.0, 0.0])
        dy = np.array([0.0, 1.0, 0.0])
        dz = np.array([0.0, 0.0, 1.0])

        # we build rotation vectors by combining tilt components and setting rotation axis to 0 to use to determine real axis di_real
        # note: d1 is fixed by model BUT for every other axis we only need 2 rotation components bcs rotation about axis itself doesn't change its direction.
        # then we find real axis di_real by converting rotation vector into rotation matrix and matrix multiplying it with nominal axis di
        phi2 = np.array([0.0, 0.0, tilts[0].tilt_2])   # nominal y -> x/z tilt
        d2_real = exp_so3(phi2) @ dy                # d2 = dy  
        phi3 = np.array([tilts[1].tilt_1, tilts[1].tilt_2, 0.0])   # nominal z -> x/y tilt
        d3_real = exp_so3(phi3) @ dz                # d3 = dz   

        phi4 = np.array([0.0, tilts[2].tilt_1, tilts[2].tilt_2])   # nominal x -> y/z tilt
        d4_real = exp_so3(phi4) @ dx                # d4 = dx   
        phi5 = np.array([tilts[3].tilt_1, 0.0, tilts[3].tilt_2])   # nominal y -> x/z tilt
        d5_real = exp_so3(phi5) @ dy                # d5 = dy   
        phi6 = np.array([tilts[4].tilt_1, tilts[4].tilt_2, 0.0])   # nominal z -> x/y tilt
        d6_real = exp_so3(phi6) @ dz                # d6 = dz   

        return np.array([dx, d2_real, d3_real, d4_real, d5_real, d6_real])


    def fwd_kins(self, theta, params: Optional[np.ndarray] = None) -> Frame:
        """
        ImprovedFwdKins(theta).

        TODO: implement.

        input: theta - 6 encoder joint values; params - defaults to self.params
        output: F_ee relative to the robot base frame, with the calibrated
                error model (joint offsets + distortion, tilted axes, tool
                frame offset) applied
        """
        if params is None:
            params = self.params

        #corrects thetas using per-joint distortion polynomial coefficients
        #delta_theta_k,sys = offset_k + sum_j c_k,j (theta_k / scale_k)^j
        theta = np.asarray(theta, dtype=np.float64).reshape(6)
        theta_real = theta.copy()
        #bounds are already stored in theta_min and theta_max attribute of robot which this class inherits
        joint_scales = np.maximum(np.abs(self.theta_min), np.abs(self.theta_max))

        for k in range(6):
            #systematic error initialized at home offsets which we set joint_offsets to in init() 
            correction = self.joint_offsets[k]

            POLY_COEFF_PARAMS_IDX_OFFSET = 22 #the index offset for polynomial coeff parameters
            for j in range(self.n_poly):
                idx = POLY_COEFF_PARAMS_IDX_OFFSET + k * self.n_poly + j
                if idx >= len(params): #just to make sure we don't overflow params
                    break
                #we are told to scale theta values "scale_k" because we have a big difference in values between the mm theta params and radians theta params
                correction += pow(theta[k] / joint_scales[k], j+1) * params[idx] #j+1 because we already added constant term (j=0) from joint_offset 

            theta_real[k] += correction

        [dx, d2_real, d3_real, d4_real, d5_real, d6_real] = self._corrected_axes(params)
        
        # corrects p_c using real thetas theta 1 - 3 and real axes d1, d2, d3 (d1_real=d1=dx)
        p_c = vct3(*(theta_real[0] * dx + theta_real[1] * d2_real  + theta_real[2] * d3_real)) #we use * to unpack vector elements bcs vct3 takes elements not vector as argument
        
        # corrects R_w using real thetas theta 4 - 6 and real axes d4, d5, d6
        R_w = (Rot(vct3(*d4_real), theta_real[3]) * Rot(vct3(*d5_real), theta_real[4]) * Rot(vct3(*d6_real), theta_real[5]))

        #corrects F_WE using uncertainty formula with params[16:22]: twist of the tool frame, F_we = F_WE * Exp(xi_we)
        T_WE = exp_se3(params[16:22])
        delta_F_WE = Frame( Rot(matrix=T_WE[:3, :3]), vct3(*T_WE[:3, 3])) #exp_se3 returns a 4x4 numpy matrix, so we convert it to the Frame datatype
        # delta_F_WE = exp_se3(params[16:22])

        F_WE_real = self.F_WE * delta_F_WE

        return Frame(Rot(), p_c) * self.F_CW * Frame(R_w, vct3()) * F_WE_real

    def predict_in_tracker(self, theta, params: Optional[np.ndarray] = None) -> Frame:
        """The end-effector pose in the tracker frame: base_frame * fwd_kins."""
        return self.base_frame(params) * self.fwd_kins(theta, params)

    def inv_kins(self, F: Frame, seed=None, tol: float = 1e-10, max_iter: int = 50) -> np.ndarray:
        """
        ImprovedInvKins(F) -- provided.

        There is no closed form once the axes are tilted, so this iterates
        on your fwd_kins: both nominal wrist branches (Robot.inv_kins'
        closed form) are refined with damped Gauss-Newton / Levenberg-
        Marquardt steps on the pose error [alpha; epsilon] of
        fwd_kins(theta)^-1 F, and the in-limit solution nearest `seed`
        (default: the current command) is returned. Works as soon as your
        fwd_kins does.

        input: F - desired F_ee relative to the robot base frame
        output: theta with fwd_kins(theta) ~= F; raises ValueError if no
                branch satisfies the joint limits
        """
        seed = self.theta_cmd if seed is None else np.asarray(seed, dtype=np.float64).reshape(6)
        p_w = F.p - F.R * self.F_WE.p - self.F_CW.p   # nominal wrist center -> seeds for the linear joints
        step_max = np.array([50.0] * 3 + [0.3] * 3)
        h = np.array([1e-3] * 3 + [1e-6] * 3)

        def pose_delta(theta):
            dF = self.fwd_kins(theta).inv() * F
            return np.concatenate([log_so3(dF.R.matrix), dF.p.vec.ravel()])

        candidates = []
        for wrist in self.wrist_solutions(F.R):
            theta = np.concatenate([[p_w.x, p_w.y, p_w.z], wrist])
            lam, delta = 1e-4, pose_delta(theta)
            for _ in range(max_iter):
                cost = float(delta @ delta)
                if np.sqrt(cost) < tol:
                    break
                J = np.zeros((6, 6))
                for k in range(6):
                    th = theta.copy()
                    th[k] += h[k]
                    J[:, k] = (pose_delta(th) - delta) / h[k]
                while True:  # Levenberg-Marquardt: shrink the step until the cost decreases
                    JtJ = J.T @ J
                    step = -np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), J.T @ delta)
                    trial = theta + np.clip(step, -step_max, step_max)
                    delta_trial = pose_delta(trial)
                    if float(delta_trial @ delta_trial) < cost or lam > 1e6:
                        theta, delta, lam = trial, delta_trial, max(lam / 10.0, 1e-12)
                        break
                    lam *= 10.0
            theta[3] = (theta[3] + np.pi) % (2 * np.pi) - np.pi
            theta[5] = (theta[5] + np.pi) % (2 * np.pi) - np.pi
            if self.within_limits(theta):
                candidates.append(theta)
        if not candidates:
            raise ValueError(f"inv_kins: no joint solution within limits for {F!r}")
        return min(candidates, key=lambda th: float(np.linalg.norm(th - seed)))

    # ------------------------------------------------------------------ fitting

    def residuals(self, params: np.ndarray, data: CalibrationData) -> np.ndarray:
        """
        TODO: implement.

        input: params - candidate parameter vector; data - CalibrationData
        output: flat residual vector comparing predict_in_tracker(theta_j,
                params) with data.frames[j] for every sample (choose how to
                weigh rotation vs translation)
        """

        residuals = []
        
        for j in range(len(data)):
            #prediction of real F_ee using predict in tracker function which uses improved forward kinematics function and base frame transformation
            F_ee_real = self.predict_in_tracker(data.thetas[j], params)
            
            F_ee = data.frames[j]
            delta_F_ee = F_ee.inv() * F_ee_real #because F_ee_real = F_ee * delta_F_ee

            #find small angle rotation vector
            # delta_R = delta_F_ee.R.matrix #delta_R = I + sk(alpha_ee)
            # alpha_ee = 0.5 * np.array([
            #     delta_R[2, 1] - delta_R[1, 2],
            #     delta_R[0, 2] - delta_R[2, 0],
            #     delta_R[1, 0] - delta_R[0, 1]
            # ])

            # #find translational error vector
            # eps_ee = delta_F_ee.p.vec.ravel()

            # #combine 2 error vectors into 6D vector
            # eta = np.concatenate((alpha_ee, eps_ee))

            #easier method using log_se3 function which is inverse of exp_se3 function we used in fwd_kins
            T = np.eye(4)
            T[:3, :3] = delta_F_ee.R.matrix
            T[:3, 3] = delta_F_ee.p.vec.ravel()
            eta = log_se3(T)
            eta[:3] *= 100.0   # rad -> roughly mm scale (0.1 mm noise / 1e-3 rad noise)

            # eta = log_se3(delta_F_ee)

            #append to residuals vector 
            residuals.extend(eta)

        return np.asarray(residuals, dtype=np.float64)

    def fit(self, data: CalibrationData, **kwargs):
        """
        TODO: implement.

        Non-linear least squares over the parameters (scipy.optimize is
        allowed). Must set self.params. Return whatever diagnostics you want
        to report (final cost, iterations, ...).
        """

        # we start from our current parameter estimate
        params_initial = self.params.copy()

        # then we find params that minimize all calibration residuals using LSQ
        result = least_squares(self.residuals, params_initial, args=(data,), **kwargs)

        # after fitting the params we save them so all other functions use them by default
        self.params = result.x.copy()

        # return useful information for notebook (labels from least_squares.py comments)
        return {
            "Success (convergence criteria satisfied?)": result.success,
            "Value of the cost function at the solution": result.cost,
            "Number of function evaluations done": result.nfev, 
            "Verbal description of the termination reason": result.message
        }

    def jacobian_rank_report(self, data: CalibrationData, params: Optional[np.ndarray] = None) -> dict:
        """
        TODO: implement.

        Numerically differentiate residuals w.r.t. params and report the
        singular values (and which parameter directions are poorly
        determined). Use it to justify your parameterization and your
        choice of calibration poses in the report.
        """
        if params is None:
                params = self.params
        params = params.copy()

        #we need residuals to tell us needed size of Jacobian
        residuals = self.residuals(params, data)

        #find jacobian
        J = np.zeros((len(residuals), len(params)))
        
        h = 1e-6  # small perturbation

        # we numerically differentiate each param direction by changing 1 param at a time and seeing how residuals change
        for i in range(len(params)):
            params_positive_direction = params.copy()
            params_negative_direction = params.copy()

            params_positive_direction[i] += h
            params_negative_direction[i] -= h

            residual_positive = self.residuals(params_positive_direction, data)
            residual_negative = self.residuals(params_negative_direction, data)

            J[:, i] = (residual_positive - residual_negative) / (2 * h)

        # SVD singular values tell us which parameter directions are observable
        U, singular_values, Vt = np.linalg.svd(J, full_matrices=False)

        # standard numerical rank threshold
        rank = np.linalg.matrix_rank(J)

        return {
            "rank": int(rank),
            "n_params":  len(params),
            "singular_values": singular_values,
        }

    # ------------------------------------------------------------------ reporting

    def parameters_report(self) -> Dict[str, np.ndarray]:
        """
        TODO: implement.

        The handout (method 1 (e)) asks your program to report the updated
        kinematic parameters in physical terms, and your debugging runs to
        compare them with ground truth (rob.ground_truth_parameters()).
        Report at least: the axis tilt vectors / unit axes, the angles
        between the axes, the per-joint offsets and scale factors, and the
        tool-frame offset. Note which of these are only defined up to the
        base-frame gauge (and so are better compared as angles between axes).
        """
        params = self.params

        # axis tilt parameters
        axes = self._corrected_axes(params) 
        # axes = [dx, d2_real, d3_real, d4_real, d5_real, d6_real] 
        
        # angle between every pair of axes
        axis_angles_deg = np.zeros((6, 6))

        for i in range(6):
            for j in range(6):
                dot = np.clip(np.dot(axes[i], axes[j]), -1.0, 1.0)
                axis_angles_deg[i, j] = np.degrees(np.arccos(dot))

        # polynomial coefficients
        poly_coeffs = params[22:].reshape(6, self.n_poly)

        # same scaling used in fwd_kins
        joint_theta_scale = np.maximum(
            np.abs(self.theta_min),
            np.abs(self.theta_max)
        )

        # linear part of our correction model
        joint_scale = np.ones(6)

        if self.n_poly >= 1:
            joint_scale += poly_coeffs[:, 0] / joint_theta_scale

        return {
            "axis_angles_deg": axis_angles_deg,
            "axis_vectors": axes,
            "joint_offset": self.joint_offsets.copy(),
            "joint_scale": joint_scale,
            "joint_polynomial": poly_coeffs,
            "tool_offset": params[16:22].copy()
        }
