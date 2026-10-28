"""
Rigid registration helpers: point-set registration, frame averaging and
marker-body refinement.
"""

from typing import List

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame

from assignments.assignment1.src.tracker import Tracker
from ..simulation import PlacedBody, sample_marker_bodies, most_recent_observed_markers, update_marker_body


def register_points(a: List[vct3], b: List[vct3]) -> Frame:
    """
    TODO: implement (your own -- do not call the simulator's solver).

    Rigid point-set registration: find the least-squares Frame F such that
    F * a[i] ~= b[i] for every corresponding pair. Think about what happens
    with fewer than 3 points, with collinear points, and with a reflection.

    input:
        a - points in the source frame (e.g. a marker body's local layout)
        b - the same points observed in the target frame, same order
    output: Frame mapping a -> b
    """

    if len(a) != len(b):
        raise ValueError("a and b must contain the same number of points")

    #condition 1: making sure we have enough points (at least 3) to figure out orientation of the frame
    if len(a) < 3:
        raise ValueError("We need at least 3 points")

    #we convert the vct3 points to numpy arrays first so we can then use numpy operations for simplcity
    A = np.array([[p.x, p.y, p.z] for p in a])
    B = np.array([[p.x, p.y, p.z] for p in b])

    #looking for centroids of each point set so we can then center the points around it to remove the translation while solving for rotation
    #b_k = R*a_k + p  &  b_mean = R*a_mean + p so b_k - b_mean = R(a_k - a_mean) + (p-p) = R(a_k - a_mean)
    a_mean = np.mean(A, axis=0)
    b_mean = np.mean(B, axis=0)
    A_centered = A - a_mean
    B_centered = B - b_mean

    #condition 2: making sure the points aren't collinear so we have enough information to determine orientation of the frame with these points
    if np.linalg.matrix_rank(A_centered) < 2:
        raise ValueError("points are collinear")

    #Singular Value Decomposition to find the rotation matrix R
    H = A_centered.T @ B_centered # builds the point-set alignment matrix
    U, _, Vt = np.linalg.svd(H) # extracts principle directions from H (U contains principle directions of A_centered and V contains those of B_centered but here we get V transposed hence Vt)
    R = Vt.T @ U.T #comparing the 2 sets of directions lets us construct the rotation between them (we transpose Vt to remove the initial transpose)

    #condition 3: making sure that we have no reflection to get a valid rotation so fixing rotation if it has negative determinant
    if np.linalg.det(R) < 0: #det(R) must be +1 for valid rotation
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    #Using derived R to find the translational vector p
    p = b_mean - R @ a_mean

    return Frame(Rot(matrix=R), vct3(*p))


def average_frames(frames: List[Frame]) -> Frame:
    """
    TODO: implement.

    Average of several noisy estimates of the same Frame. Rotations cannot
    simply be averaged element-wise -- the result must be a proper
    rotation matrix.

    input:
        frames - List[Frame], all estimates of the same transform
    output: the averaged Frame
    """

    if len(frames) < 1:
        raise ValueError("We need at least 1 frame nto do the avg")

    # average translation normally
    p_sum = vct3(0, 0, 0)
    R_sum = np.zeros((3, 3))

    for F in frames:
        p_sum += F.p
        R_sum += F.R.matrix

    p_avg = p_sum / len(frames)
    R_avg = R_sum / len(frames)

    # we project the averaged matrix back onto a valid rotation
    U, _, Vt = np.linalg.svd(R_avg)
    R = U @ Vt

    # we make sure we did not get a reflection
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return Frame(Rot(matrix=R), p_avg)


def refine_marker_body(trk: Tracker, placed: PlacedBody, n_readings: int = 30) -> List[vct3]:
    """
    TODO: implement.

    "It is possible to improve your estimate of a_m,k through a calibration
    process involving multiple observations of the marker body using the
    tracking camera and then averaging them."

    Take n_readings SampleMarkerBodies readings, use
    most_recent_observed_markers(...) plus your register_points to bring
    each reading's b_m,k back into the body's own frame, average, and push
    the result into the body with update_marker_body(...). Note that each
    reading costs delta_t_samp of simulated time, and that the tracker
    jiggles between readings.

    input:
        trk - Tracker the body is tracked by
        placed - PlacedBody whose body's nominal marker positions should be refined
        n_readings - number of readings to average
    output: the new List[vct3] of marker positions (also stored on the body)
    """

    if n_readings < 1:
        raise ValueError("We need sensor readings and finite amount of passes to do refinement")
   
    # current estimate of marker positions in the marker-body frame
    a = placed.body.marker_positions

    # there are multiple readings for a single marker, so we will sum them up and then average them to get a refined reading of the marker position
    # sum_point = {} 
    sums = [vct3(0, 0, 0) for _ in a]

    for read in range(n_readings): # camera readings 
        # we take one tracker reading
        # reading = (sample_marker_bodies(trk, [placed])[placed.body.name]) # dict with the body name as key and the reading as value
        sample_marker_bodies(trk)

        # we observed marker positions in tracker coordinates
        b = most_recent_observed_markers(placed.body) #marker.p

        # we estimate marker-body -> tracker frame
        F = register_points(a, b)

        # we bring measured markers back into the marker-body frame
        F_inv = F.inv() #F_trk_mb

        for i, b_i in enumerate(b):
            sums[i] += F_inv * b_i # sum up the marker positions for marker i converted from tracker frame to marker body frame
            # if i not in sum_point:
            #     sum_point[i] = vct3(0, 0, 0) 
    
    # refined reading = average_reading
    average_reading = []
    for sum in sums:
        avg_reading = sum / n_readings # average positions for marker x 
        average_reading.append(avg_reading)

    # store the improved marker locations on the body
    update_marker_body(placed.body, average_reading) # set the marker positions of the placed body to the refined (average) reading in the marker body frame
    
    return average_reading   