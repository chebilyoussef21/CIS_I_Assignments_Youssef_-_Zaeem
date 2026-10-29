"""
Data-collection workflow: which robot poses to visit, how to read them, and
how to register the robot base to the tracker.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from data_types.nominal_types import vct3, Frame

from ..robot import Robot
from assignments.assignment1.src.tracker import Tracker
from ..simulation import PlacedBody
from ..simulation.tracker import (sample_marker_bodies)
from ..student.registration import (average_frames)

@dataclass
class CalibrationData:
    """One calibration data set.

    thetas[j] is the commanded (encoder) joint vector and frames[j] the
    calibration body's observed pose relative to the tracker, F_ee,j (this
    handout's tracker has no jiggle, so the tracker frame is a fixed
    reference). t_collect is the simulated time the collection took.
    """
    thetas: List[np.ndarray] = field(default_factory=list)
    frames: List[Frame] = field(default_factory=list)
    t_collect: float = 0.0

    def __len__(self):
        return len(self.thetas)


def measure_home_offsets(rob: Robot, n_homings: int = 5) -> np.ndarray:
    """
    TODO: implement.

    HomeRobot() drives the robot to theta* = 0 and returns the encoder
    values there, i.e. -delta_theta_sys(home) - delta_theta_rnd. Repeat it
    and average to estimate the systematic joint offsets at home (after the
    first homing the move is tiny, so each repeat costs about t_settle).

    input:
        rob - the robot (rob.home_robot())
        n_homings - how many homings to average
    output: estimated delta_theta_sys at home, np.ndarray of 6
    """
    offset_sum = np.zeros(6)

    # each homing places the actual robot at theta* = 0 and returns the encoder value measured at that physical zero
    for _ in range(n_homings):
        encoder_value = rob.home_robot()
        offset_sum += encoder_value

    # we average repeated homings to reduce zero-mean random joint error
    mean_encoder_at_home = offset_sum / n_homings
    
    # at home theta_encoder* = 0 so theta_encoder = -delta_theta_sys - delta_theta_rnd
    # so after averaging, delta_theta_sys ~= -mean(theta_encoder)
    delta_theta_sys = -mean_encoder_at_home
    return delta_theta_sys

def append_joint_sweep(poses, joint_index, joint_min, joint_max, spacing, rob, trk, mbee_local_positions):

    # iterations w.r.t range and spacing making sure spacing is never exceeded
    for i in range(int(np.ceil((joint_max - joint_min) / spacing)) + 1):

        displacement = min(spacing * i, joint_max - joint_min)

        # we move only the selected joint so we make all other theta values 0 for simplicity and so we limit other sources of error
        theta = np.zeros(6)
        theta[joint_index] = joint_min + displacement

        # We get the end effector location to use later to see if all markers attached to it are visible
        # for this we use the forward kinematics function we make in kinematics.py to predict the end-effector frame using the theta values we just set
        F_ee_workspace = rob.F_rob * rob.fwd_kins(theta)

        # Here we loop through all calibration markers and make sure every each one is visible
        visible = True
        for marker in mbee_local_positions:
            marker_workspace = F_ee_workspace * marker
            if not trk.is_visible(marker_workspace):
                visible = False
                break
        if visible:
            poses.append(theta)

    # return to home after finishing this joint sweep
    poses.append(np.zeros(6))

# this should contain a desired accuracy which will affect the number of calibration poses
def plan_calibration_poses(rob: Robot, trk: Tracker, mbee_local_positions: List[vct3], desired_accuracy: float, **kwargs) -> List[np.ndarray]:
    """
    TODO: implement -- this is a design task.

    Choose the joint vectors the robot will visit during calibration to reach the desired accuracy. Every
    parameter of your kinematic model must be excited by the data (axis
    tilts need travel along each axis; the wrist axes and F_we need wrist
    rotations; the joint distortion needs several stops along each joint),
    every pose must keep all calibration-body markers inside the tracker's
    viewing volume (Tracker.is_visible, using the nominal rob.F_rob and
    rob.fwd_kins -- they are only tens of mm off), and the order of the
    poses determines the motion time. The wrist moves at 0.1 rad/s: a pi/2
    turn costs ~16 s.

    The handout's method 1: decide how closely spaced the samples along an
    axis must be from the error bounds (delta_sys^trk, omega_sys^trk,
    delta_sys^rob, omega_sys^rob), then sweep each axis in turn from the
    home position. Joints 1-3 are decoupled from 4-6.

    input:
        rob - the Robot (nominal kinematics + limits)
        trk - the Tracker (for visibility checks)
        mbee_local_positions - the calibration body's marker layout a*_k
        desired_accuracy - target RMS orientation error in rad (the limiting
            metric); use it to size the plan -- how many distinct poses and
            how often each is read (a pose listed twice in a row is
            re-commanded and re-read, costing only t_settle + delta_t_samp)
        kwargs - whatever your design needs (grid sizes, margins, ...)
    output: ordered List of joint vectors (np.ndarray of 6)
    """
    poses = []

    spacing_mm = ...
    spacing_rad = ...

    home = np.zeros(6)
    poses.append(home.copy())

    # TRANSLATIONAL JOINTS
    # theta 1
    append_joint_sweep(poses=poses, joint_index=0, joint_min=0, joint_max=600, spacing=spacing_mm, rob=rob, trk=trk, mbee_local_positions=mbee_local_positions)

    # theta 2
    append_joint_sweep(poses=poses, joint_index=1, joint_min=-300, joint_max=300, spacing=spacing_mm, rob=rob, trk=trk, mbee_local_positions=mbee_local_positions)

    # theta 3
    append_joint_sweep(poses=poses, joint_index=2, joint_min=0, joint_max=600, spacing=spacing_mm, rob=rob, trk=trk, mbee_local_positions=mbee_local_positions)

    # ROTATIONAL JOINTS

    # theta 4
    append_joint_sweep(poses=poses, joint_index=3, joint_min=-0.75*np.pi, joint_max=0.75*np.pi, spacing=spacing_rad, rob=rob, trk=trk, mbee_local_positions=mbee_local_positions)

    # theta 5
    append_joint_sweep(poses=poses, joint_index=4, joint_min=-0.75*np.pi, joint_max=0.75*np.pi, spacing=spacing_rad, rob=rob, trk=trk, mbee_local_positions=mbee_local_positions)

    # theta 6
    append_joint_sweep(poses=poses, joint_index=5, joint_min=-np.pi, joint_max=np.pi, spacing=spacing_rad, rob=rob, trk=trk, mbee_local_positions=mbee_local_positions)

    return poses


def collect_calibration_data(rob: Robot, trk: Tracker, mbee_placed: PlacedBody,
                             thetas: List[np.ndarray], repeats: int = 1) -> CalibrationData:
    """
    TODO: implement.

    Visit each joint vector with rob.move_joints, read the calibration body
    with sample_marker_bodies and store its frame relative to the tracker.
    Decide what to do when the body is not visible (None), and whether/how
    to use repeats (re-sampling costs 0.03 s; re-commanding the same pose
    costs t_settle and re-rolls the joint repeatability noise).

    input:
        rob, trk - robot and tracker
        mbee_placed - the robot calibration body (attached)
        thetas - joint vectors to visit, in order
        repeats - readings per pose
    output: CalibrationData
    """

    data = CalibrationData()

    for theta in thetas:

        # command the robot to the selected calibration pose
        dt = rob.move_joints(theta)
        data.t_collect += dt

        observed_frames = []

        # take repeated tracker readings at the same physical robot pose
        for repeat in range(repeats):
            # We already checked if marker would be visible to tracker in the plan_calibration_poses function 
            # BUT that was a prediction and in this case we're actually moving the robot so the tracker will tell us in real time if the mariker is actually visible
            # the prediction helped us limit the poses we do if we can predict that it's not visible but if we can't then we'll have to check again to make sure
            sample_marker_bodies(trk)

            # each tracker sample costs 0.03 s
            data.t_collect += 0.03 #constants.SAMPLE_DELTA_T_SEC

            # get the most recent measured frame of the calibration body which was stored in sample_marker_bodies fct
            F_ee = mbee_placed.observed_frame
            # None means the calibration body was not visible
            if F_ee is not None:
                # store the joint vector for thetas and the measured EE frame for frames entry of data
                data.thetas.append(theta.copy())
                data.frames.append(F_ee)
                # observed_frames.append(F_ee)
                break

        # # if every reading failed, skip this calibration pose
        # if len(observed_frames) != 0:
        #     #if length = 1 (only observation) no need to average 
        #     if len(observed_frames) == 1:
        #         F_ee_avg = observed_frames[0]
        #     # average repeated noisy tracker observations
        #     else:
        #         F_ee_avg = average_frames(observed_frames)

        #     # store the joint vector for thetas and the measured EE frame for frames entry of data
        #     data.thetas.append(theta.copy())
        #     data.frames.append(F_ee_avg)

    return data

def estimate_base_frame(data: CalibrationData, kins: Robot) -> Frame:
    """
    TODO: implement.

    With the calibration body's frame being exactly F_ee, every sample
    satisfies  F_ee,j = F_X * kins.fwd_kins(theta_j)  for one constant
    unknown F_X (the robot base expressed in the tracker's frame). Estimate
    F_X from all samples (your average_frames may help).

    input:
        data - CalibrationData
        kins - the kinematics to use (the nominal Robot at first; later your
            ImprovedKinematics)
    output: F_X
    """

    if len(data) == 0:
        raise ValueError("No calibration data available")

    F_X_estimates = []

    # each calibration sample gives us one estimate of the base frame
    for theta, F_ee_measured in zip(data.thetas, data.frames):

        # end effector pose predicted relative to the robot base
        F_ee_robot = kins.fwd_kins(theta)

        # F_ee_measured = F_X * F_ee_robot
        # so F_X = F_ee_measured * inv(F_ee_robot)
        F_X = F_ee_measured * F_ee_robot.inv()

        F_X_estimates.append(F_X)

    # all samples estimate the same base frame, so average them
    F_X_avg = average_frames(F_X_estimates)

    return F_X_avg
    
    raise NotImplementedError("TODO: implement estimate_base_frame")
