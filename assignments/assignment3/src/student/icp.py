"""Rigid registration: point-set registration, coarse initialisation and ICP."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from assignments.assignment3.src.student.closest_point import closest_points_on_mesh
from data_types.nominal_types import vct3, Frame
from ..mesh import Mesh, PatchBox


@dataclass
class IcpResult:
    """Output of icp()."""
    F_reg: Frame                  # reference body -> CT
    residuals: np.ndarray         # (K,) ||F_reg d_k - closest_k|| for every sample
    inlier_mask: np.ndarray       # (K,) bool, False for rejected samples
    closest: np.ndarray           # (K,3) closest mesh points of the last iteration
    normals: np.ndarray           # (K,3) mesh normals at those points
    n_iter: int
    converged: bool
    history: List[float] = field(default_factory=list)   # rms inlier residual per iteration


def register_points(a: List[vct3], b: List[vct3]) -> Frame:
    """
    TODO: implement (your own; do not call the simulator's solver).
    input: paired points a_i, b_i (>= 3)
    output: least-squares Frame F with F * a_i ~= b_i
    """
    assert len(a) == len(b) and len(a) >= 3
    a = np.asarray(a)
    b = np.asarray(b)
    a_centroid = np.mean(a, axis=0)
    b_centroid = np.mean(b, axis=0)
    translation_removed_a = a - a_centroid
    translation_removed_b = b - b_centroid
    U, S, Vt = np.linalg.svd(np.dot(translation_removed_a.T, translation_removed_b))
    diag = np.eye(3)
    diag[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
    X = np.matmul(V.T,U.T)
    R = X
    if np.linalg.det(X) == -1:
        R = np.matmul(np.matmul(Vt.T,diag),U.T)
    translation_ = b_centroid - np.matmul(R, a_centroid)
    return Frame(R, translation_)


def coarse_registration(d_by_patch: Dict[int, np.ndarray], boxes: List[PatchBox]) -> Frame:
    """
    TODO: implement.
    input: {patch id: (n_i,3) tip points in reference coordinates}, the patch boxes
    output: a rough F_reg (reference -> CT) from patch centroids <-> box centres
    """
    assert len(d_by_patch) == len(boxes)
    box_centres = np.array([box.p for box in boxes])
    average_R = np.zeros((3, 3))
    average_translation = np.zeros(3)
    reference_centroids = np.zeros((len(d_by_patch), 3))
    box_centroids = np.zeros((len(d_by_patch), 3))
    for patch_id, tip_points in d_by_patch.items():
        tip_point_centroid = np.mean(tip_points, axis=0)
        box_centroid = box_centres[patch_id]
    return register_points(reference_centroids, box_centroids)
    """"
        reference_centroids[patch_id] = tip_point_centroid
        box_centroids[patch_id] = box_centroid
    overall_reference_centroid = np.mean(reference_centroids, axis=0)
    overall_box_centroid = np.mean(box_centroids, axis=0)
    translation_removed_reference_centroids = reference_centroids - overall_reference_centroid
    translation_removed_box_centroids = box_centroids - overall_box_centroid
    U, S, Vt = np.linalg.svd(np.dot(translation_removed_reference_centroids.T, translation_removed_box_centroids))
    diag = np.eye(3)
    diag[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
    R = np.matmul(np.matmul(Vt.T,diag),U.T)
    translation_ = box_centroid - np.matmul(R, tip_point_centroid)
    return Frame(R, translation_)
    """

def icp(d: np.ndarray, mesh: Mesh, F_init: Frame, max_iter: int = 100, tol: float = 1e-4,
        reject: Optional[str] = "mad") -> IcpResult:
    """
    TODO: implement.
    input: d (K,3) tip points in reference coordinates, mesh, initial F_reg,
           iteration cap, convergence tolerance, outlier-rejection method
    output: IcpResult
    """
    F_ = F_init
    converged_check = False
    rms_history = []

    for iteration in range(max_iter):
        registration_guess = (F_@d.T).T
        closest_points,_,_ = closest_points_on_mesh(registration_guess, mesh)
        residuals = np.linalg.norm(registration_guess- closest_points, axis=1)
        #if np.mean(residuals) < tol:
        #    break
        inliers = np.ones_like(residuals, dtype=bool)
        if reject == "mad":
            cutoff = np.median(residuals) + 3 * np.median(np.abs(residuals - np.median(residuals))) # CHANGE DIS
            outliers = residuals > cutoff
            inliers = ~outliers

        if np.sum(inliers) < 3:
            raise ValueError("Not enough inliers for ICP registration.")
        d_inliers = d[inliers]

        root_mean_square = np.sqrt(np.mean(residuals[inliers]**2))
        rms_history.append(root_mean_square)
        if iteration>0 and np.abs(rms_history[-1]-rms_history[-2]) < tol:
            converged_check = True  
            break

        closest_inliers = closest_points[inliers]
        F_ = register_points(d_inliers, closest_inliers)
 
        
    registration_guess_final = (F_@d.T).T
    closest_final, triangle_indexes_final, _ = closest_points_on_mesh(registration_guess_final, mesh)
    final_residuals = np.linalg.norm(registration_guess_final - closest_final, axis=1)
    final_inliers = np.ones_like(residuals, dtype=bool)
    if reject == "mad":
        cutoff = np.median(residuals) + 3 * np.median(np.abs(residuals - np.median(residuals))) # CHANGE DIS
        final_inliers = residuals < cutoff
    
    normal_final = mesh.face_normals[triangle_indexes_final]

    return IcpResult(F_reg=F_,residuals=final_residuals, inlier_mask=final_inliers, closest=closest_final, normals=normal_final, n_iter=iteration+1, converged=converged_check)   
