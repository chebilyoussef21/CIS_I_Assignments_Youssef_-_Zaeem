"""Statistical estimate of the registration error and the predicted TRE."""

from typing import Tuple

import numpy as np

from data_types.nominal_types import Frame
from ..mesh import Mesh
from .icp import IcpResult
from uncertainty_networks.se3 import  log_so3, exp_se3, log_se3, exp_so3


def registration_covariance(result: IcpResult, F0_reg: Frame) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO: implement.
    Left/CT convention: F*_reg = Exp(delta) . F_reg, eta_reg = [alpha; eps] relative to F0_reg.
    input: IcpResult, the initial guess F0_reg
    output: mu (6,) estimate of eta_reg, C_reg (6,6) covariance of delta
    """
    #use initial guess of F_reg and F_reg from ICP result and use to find eta_reg between them
    #corrected/refined frame transformation: F_reg = Exp(eta_reg) * F0_reg AND delta_F = Exp(eta_reg) 
    delta_F = result.F_reg * F0_reg.inv()  #Frame object 
    #extract components of Frame object to construct 4x4 matrix T
    delta_T = np.eye(4)
    delta_T[:3, :3] = delta_F.R.matrix
    delta_T[:3, 3] = np.asarray(delta_F.p).reshape(3)
    #mu = log(delta) = estimate of eta_reg
    mu = np.asarray(log_se3(delta_T)).reshape(6)

    #use closest pts, normals, and residuals from ICP result to build point-to-plane Jacobian
    # BUT we only use the ones we mask 
    closest_points = result.closest[result.inlier_mask]
    normals = result.normals[result.inlier_mask]
    residuals = result.residuals[result.inlier_mask]

    # point_to_plane_jacobian = normals.transpose() @ [-exp_se3(points), np.eye(3)]
    #1 row for each sampled closest point, 6 columns: each column gives us the change (derivative) of the residual r_i w.r.t one of the 6 components of eta
    point_to_point_jacobian = np.zeros((len(closest_points), 6)) 
    # point_to_plane_jacobian = np.zeros((len(closest_points), 6)) 
    N_samples = len(closest_points)
    for k in range(N_samples):
        c_i = closest_points[k] 
        x_k = c_i ## approximation x_k ~= c_i bcs x_k = result.F_reg * d but we dont have d so no other choice
        n_i = normals[k]
        skew_x_k = np.array([
            [0, -x_k[2],  x_k[1]],
            [x_k[2],  0, -x_k[0]],
            [-x_k[1], x_k[0],  0]
        ])
        point_to_point_jacobian[k, :] = np.hstack((-skew_x_k, np.eye(3)))
        point_to_plane_jacobian = n_i @ point_to_point_jacobian

    A = point_to_point_jacobian
    # A = point_to_plane_jacobian

    # if np.linalg.matrix_rank(A) < 6: #EDGE CASE
    #     raise ValueError("Registration geometry is degenerate.")

    # Residual variance
    # if len(closest_points) <= 6: #EDGE CASE
    #     raise ValueError("Need more than 6 inlier samples to estimate registration covariance.")
    DOF = max(N_samples - 6, 1) #degrees of freedom 
    residual_variance = np.sum(np.power(residuals,2)) / DOF

    #2nd : Residual covariance from registration
    C_reg = residual_variance * np.linalg.pinv(A.transpose() @ A)

    return mu, C_reg

def predicted_tre(mesh: Mesh, F_reg: Frame, C_reg: np.ndarray) -> dict:
    """
    TODO: implement.
    input: mesh, estimated F_reg, C_reg (6,6)
    output: {"rms", "mean", "max", "p90"} of the predicted TRE over the mesh vertices
    """
    #positional_displacement
    vertices_from_mesh = np.asarray(mesh.vertices)
    r = np.zeros(len(vertices_from_mesh))

    for k in range(len(vertices_from_mesh)):
        x_k = vertices_from_mesh[k]

        skew_x_k = np.array([
                    [0, -x_k[2],  x_k[1]],
                    [x_k[2],  0, -x_k[0]],
                    [-x_k[1], x_k[0],  0]
                ])
        # A_i = np.array([-skew_x, np.eye(3)])
        A_k = np.hstack((-skew_x_k, np.eye(3)))
        C_x = A_k @ C_reg @ A_k.transpose() 
        r[k] = np.sqrt(C_x.trace())

    RMS = np.sqrt(np.mean(np.power(r,2)))

    return {
        "rms": float(RMS), 
        "mean": float(r.mean()), 
        "max": float(r.max()),
        "p90": float(np.percentile(r, 90))
    }