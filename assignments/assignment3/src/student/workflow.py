"""The end-to-end registration workflow: design, patch plan, data collection, registration."""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..mesh import Mesh, PatchBox
from ..session import Session, SceneDesign, PointerPlacement
from .report import RegistrationReport

from .icp import coarse_registration, icp, IcpResult
from .uncertainty import predicted_tre, registration_covariance
from data_types.nominal_types import Frame
from src import vct3, Rot, Frame

import time

@dataclass
class SampleSet:
    """Tip positions collected through the tracker."""
    handles: List[PointerPlacement] = field(default_factory=list)
    patch_ids: List[int] = field(default_factory=list)
    d_ref: List[np.ndarray] = field(default_factory=list)   # (3,) tip in reference-body coordinates
    t_collect: float = 0.0

    def __len__(self) -> int:
        return len(self.d_ref)

    def as_array(self) -> np.ndarray:
        return np.array(self.d_ref).reshape(-1, 3)


def set_marker_positions(dimension: int):
    marker_positions = [
        vct3([-dimension /2, - dimension /2, 0]),
        vct3([ dimension /2, - dimension /2, 0]),
        vct3([-dimension /2,  dimension /2, 0]),
        vct3([ dimension /2,  dimension /2, dimension/2])
    ]
    return marker_positions

def design_scene() -> SceneDesign:
    """
    return the same values the notebook's Step 1 cell uses
    (grading calls this instead of the notebook).
    """
    R_tracker = Rot(matrix=np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    p_tracker = vct3([-1600, 0, 0])
    F_tracker_approx = Frame(R_tracker, p_tracker)

    REF_MB_DIMENSION = 60
    ref_marker_positions = set_marker_positions(REF_MB_DIMENSION)

    POINTER_MB_DIMENSION = 50
    ptr_marker_positions = set_marker_positions(POINTER_MB_DIMENSION)

    p_tip = vct3([0, 0, -150])

    return SceneDesign(F_tracker_approx, ref_marker_positions, ptr_marker_positions, p_tip)


def plan_patches(boxes: List[PatchBox], n_patches: int, samples_per_patch: int) -> List[int]:
    """
    input: the patch boxes, how many patches to use, samples per patch
    output: ordered list of patch ids to call place_pointer with (repeats allowed)
    """
    
    if n_patches == 0:          #EDGE CASE 1
        raise ValueError("n_patches must be greater than 0.")
    if n_patches > len(boxes):  #EDGE CASE 2
        raise ValueError("n_patches cannot exceed the number of available patches.")
    if samples_per_patch <= 0:  #EDGE CASE 3
        raise ValueError("samples_per_patch must be greater than 0.")

    plan = []

    #we want to choose sample patches with enough separation between centers and ones that are large enough for repeated ptr placement
    spacing = len(boxes) / n_patches

    for i in range(n_patches):
        patch_id_chosen = int(i*spacing)
        for j in range(samples_per_patch):  #repeats ALLOWED so we can put same sample more than once
            plan.append(patch_id_chosen)
    return plan  

def collect_samples(session: Session, plan: List[int]) -> SampleSet:
    """
    For every patch id: session.place_pointer, session.sample_marker_bodies,
    d_k = F_ref^-1 F_ptr p_tip; skip (and log) samples where a body was invisible.
    input: session, patch plan
    output: SampleSet
    """
    sample_set = SampleSet()
    t_start = session.elapsed()
    for patch_id in plan: 
        handle = session.place_pointer(patch_id)
        observations = session.sample_marker_bodies()
        #if one of bodies isn't visible we skip the sample
        F_ref = observations["ref"].frame
        F_ptr = observations["ptr"].frame
        if F_ref is None or F_ptr is None: 
            continue
        # Pointer-tip position expressed in reference-body coordinates
        d_k = F_ref.inv() * F_ptr * session.design.p_tip         #ptip*=ptip so no uncertainty

        sample_set.handles.append(handle)
        sample_set.patch_ids.append(patch_id)
        sample_set.d_ref.append(d_k)

    sample_set.t_collect = session.elapsed() - t_start

    return sample_set


def run_registration(session: Session, mesh: Mesh, boxes: List[PatchBox], **kwargs) -> RegistrationReport:
    """
    the headless entry point used for grading:
    plan -> collect -> coarse_registration -> icp -> uncertainty -> RegistrationReport.
    input: session, mesh, boxes, your own keyword options
    output: RegistrationReport
    """
    t_wall_start = time.perf_counter()
    t_simulated_start = session.elapsed() 
    n_patches = 46          #copied from value chosen in notebook template 
    samples_per_patch = 4   #copied from value chosen in notebook template 
    plan = plan_patches(boxes, n_patches, samples_per_patch)
    samples_by_patch = collect_samples(session, plan)
    F_init = coarse_registration(samples_by_patch, boxes)
    result = icp(d=samples_by_patch.as_array(), mesh=mesh, F_init=F_init, max_iter= 100, tol= 1e-4, reject= "mad") #reject attribute optional defaults to 'mad'
    t_simulated = session.elapsed() - t_simulated_start
    mu, C_reg = registration_covariance(result=result, F0_reg=F_init)
    TRE = predicted_tre(mesh=mesh, F_reg=result.F_reg, C_reg=C_reg)
    t_wall = time.perf_counter() - t_wall_start
    return RegistrationReport(
        F_reg = result.F_reg,
        patches = boxes,
        samples = samples_by_patch,
        residual_stats = result.residuals,
        t_simulated = t_simulated,
        t_wall = t_wall,
        n_icp_iterations = result.n_iter,
        converged = result.converged,
        eta_reg = mu,
        predicted_tre = TRE
    )

    # sample report
    samples_report = []

    for i in range(len(samples_by_patch)):
        used = bool(result.inlier_mask[i])

        samples_report.append({
            "sample_index": samples_by_patch.handles[i].sample_index,
            "patch_id": samples_by_patch.patch_ids[i],
            "t_sim": samples_by_patch.handles[i].t_sim,
            "d_ref": np.asarray(samples_by_patch.d_ref[i]).tolist(),
            "residual_mm": float(result.residuals[i]),
            "used": used,
            "reason": "" if used else "ICP outlier"
        })

    # patch report
    patches_report = []

    for patch_id in sorted(set(samples_by_patch.patch_ids)):

        indices = [
            i for i, pid in enumerate(samples_by_patch.patch_ids)
            if pid == patch_id
        ]

        n_samples_patch = len(indices)
        n_used_patch = sum(bool(result.inlier_mask[i]) for i in indices)

        patches_report.append({
            "patch_id": patch_id,
            "n_samples": n_samples_patch,
            "n_used": n_used_patch,
            "n_rejected": n_samples_patch - n_used_patch
        })

    # Residual statistics over ICP inliers
    used_residuals = result.residuals[result.inlier_mask]

    residual_stats = {
        "n_used": int(len(used_residuals)),
        "mean": float(np.mean(used_residuals)),
        "std": float(np.std(used_residuals)),
        "max": float(np.max(used_residuals)),
        "p90": float(np.percentile(used_residuals, 90))
    }

    # Registration uncertainty report
    eta_reg = {
        "mu": np.asarray(mu).tolist(),
        "C_reg": np.asarray(C_reg).tolist(),
        "convention": "left_ct"
    }

    t_wall = time.perf_counter() - t_wall_start

    return RegistrationReport(
        F_reg=result.F_reg,
        patches=patches_report,
        samples=samples_report,
        residual_stats=residual_stats,
        t_simulated=t_simulated,
        t_wall=t_wall,
        n_icp_iterations=result.n_iter,
        converged=result.converged,
        eta_reg=eta_reg,
        predicted_tre=TRE
    )