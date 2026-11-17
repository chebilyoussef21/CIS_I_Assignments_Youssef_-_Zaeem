"""The end-to-end registration workflow: design, patch plan, data collection, registration."""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..mesh import Mesh, PatchBox
from ..session import Session, SceneDesign, PointerPlacement
from .report import RegistrationReport

from .icp import coarse_registration, icp, IcpResult
from .uncertainty import predicted_tre
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


def design_scene() -> SceneDesign:
    """
    TODO: implement -- return the same values the notebook's Step 1 cell uses
    (grading calls this instead of the notebook).
    """
    F_tracker_approx = [0,0,0]     # TODO: Frame, camera pose relative to the workspace (looks along its local +z)
    ref_marker_positions = [    # TODO: >= 3 points, reference-body coordinates
    ]
    ptr_marker_positions = [    # TODO: >= 3 points, pointer coordinates
    ]
    p_tip = None                # TODO: vct3, tip in pointer coordinates

    return SceneDesign(F_tracker_approx, ref_marker_positions, ptr_marker_positions, p_tip)


def plan_patches(boxes: List[PatchBox], n_patches: int, samples_per_patch: int) -> List[int]:
    """
    TODO: implement.
    input: the patch boxes, how many patches to use, samples per patch
    output: ordered list of patch ids to call place_pointer with (repeats allowed)
    """
    plan = []
    #we want to choose sample patches with enough separation between centers and ones that are large enough for repeated ptr placement
    spacing = len(boxes) / n_patches
    for i in range(n_patches):
        patch_id_chosen = i*spacing
        for j in range(samples_per_patch):  #repeats ALLOWE so we can put same sample more than once
            plan.append(boxes[patch_id_chosen])
    return plan    

def collect_samples(session: Session, plan: List[int]) -> SampleSet:
    """
    TODO: implement.
    For every patch id: session.place_pointer, session.sample_marker_bodies,
    d_k = F_ref^-1 F_ptr p_tip; skip (and log) samples where a body was invisible.
    input: session, patch plan
    output: SampleSet
    """
    sample_set = SampleSet()
    for patch_id in plan: 
        session.place_pointer(patch_id)
        observations = session.sample_marker_bodies()
        #if one of bodies isn't visible we skip the sample
        F_ref = observations["ref"]
        F_ptr = observations["ptr"]
        if F_ref is None or F_ptr is None: 
            continue
        #ptip*=ptip so no uncertainty
        d_k = F_ref.inv() * F_ptr * session.design.p_tip
        sample_set.d_ref.append(d_k)
    return sample_set

def run_registration(session: Session, mesh: Mesh, boxes: List[PatchBox], **kwargs) -> RegistrationReport:
    """
    TODO: implement -- the headless entry point used for grading:
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
    result = icp(d=samples_by_patch, mesh=mesh, F_init=F_init, max_iter= 100, tol= 1e-4, reject= "mad") #reject attribute optional defaults to 'mad'
    t_simulated = session.elapsed() - t_simulated_start
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
        eta_reg = result.eta_reg,
        predicted_tre = predicted_tre(mesh, result.F_reg, C_reg=):
    )