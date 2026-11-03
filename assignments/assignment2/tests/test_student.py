"""
Unit tests for YOUR components (src/student). The complete tests below fail
with NotImplementedError until you implement the function they exercise;
the ones marked TODO are yours to write -- the handout requires you to
describe your unit tests and report their results.

Run from the assignment2 folder:  python -m pytest tests -q
"""

import numpy as np
import pytest

from src import simulation, vct3, Rot, Frame
from src.student import metrics, registration, workflow, kinematics


def random_theta(rng):
    return np.array([rng.uniform(0, 600), rng.uniform(-300, 300), rng.uniform(0, 600),
                     rng.uniform(-1.3, 1.3), rng.uniform(-1.3, 1.3), rng.uniform(-2.5, 2.5)])


def random_frame(rng, angle=1.0, dist=100.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return Frame(Rot(vct3(axis), angle=rng.uniform(-angle, angle)), vct3(rng.uniform(-dist, dist, 3)))


# ---------------------------------------------------------------- metrics

def test_frame_error_known_values():
    F_a = Frame(Rot(), vct3(1, 2, 3))
    F_b = Frame(Rot('z', 0.1), vct3(1, 2, 3))
    angle, dist = metrics.frame_error(F_a, F_b)
    assert np.isclose(angle, 0.1) and np.isclose(dist, 0.0)
    angle, dist = metrics.frame_error(F_a, Frame(Rot(), vct3(4, 6, 3)))
    assert np.isclose(angle, 0.0) and np.isclose(dist, 5.0)


def test_frame_error_is_invariant_to_common_left_factor():
    rng = np.random.default_rng(1)
    G, F_a, F_b = random_frame(rng), random_frame(rng), random_frame(rng)
    assert np.allclose(metrics.frame_error(F_a, F_b), metrics.frame_error(G * F_a, G * F_b))


def test_summarize_errors():
    s = metrics.summarize_errors([(0.0, 3.0), (0.0, 4.0)])
    assert s["n"] == 2 and np.isclose(s["rms_pos"], np.sqrt(12.5)) and np.isclose(s["max_pos"], 4.0)


# ---------------------------------------------------------------- registration

def test_register_points_recovers_known_transform():
    rng = np.random.default_rng(2)
    F = random_frame(rng)
    a = [vct3(rng.uniform(-50, 50, 3)) for _ in range(6)]
    b = [F * p for p in a]
    F_est = registration.register_points(a, b)
    assert np.allclose(F_est.R.matrix, F.R.matrix, atol=1e-9)
    assert np.allclose(F_est.p.vec, F.p.vec, atol=1e-9)


def test_register_points_rejects_degenerate_input():
    with pytest.raises(ValueError):
        registration.register_points([vct3(0, 0, 0), vct3(1, 0, 0)], [vct3(0, 0, 0), vct3(1, 0, 0)])


def test_average_frames_of_identical_frames():
    rng = np.random.default_rng(3)
    F = random_frame(rng)
    F_avg = registration.average_frames([F] * 5)
    assert np.allclose(F_avg.R.matrix, F.R.matrix, atol=1e-9)
    assert np.allclose(F_avg.p.vec, F.p.vec, atol=1e-9)


def test_average_frames_is_a_proper_rotation():
    rng = np.random.default_rng(4)
    F_avg = registration.average_frames([random_frame(rng, angle=0.2) for _ in range(10)])
    R = F_avg.R.matrix
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)


def test_refine_marker_body_reduces_layout_error():
    """
    Hint: define a body, place it in front of a tracker, measure how far the
    nominal layout is from body.get_actual_marker_positions() (ground truth,
    allowed in a test) up to a rigid transform, refine, and show the
    distance shrank. Also check how much simulated time it cost.
    """

    mb_positions = [vct3(0, 0, 0), vct3(60, 0, 0), vct3(0, 60, 0), vct3(0, 0, 60)]
    body = simulation.define_marker_body("mb", mb_positions)
    trk = simulation.create_tracker("trk", Frame(Rot(), vct3(0, 0, -1600)), seed=5)
    placed = simulation.place_marker_body(body, Frame(Rot('z', 0.3), vct3(100, 50, 0)))
    simulation.add_marker_bodies(trk, [placed])

    # ground truth marker positions in the body frame (allowed in a test)
    actual_positions = [a.p for a in body.get_actual_marker_positions()]

    def layout_error(layout): # rms distance between a layout and the true markers, after the best rigid fit
        F = registration.register_points(layout, actual_positions)
        return np.sqrt(np.mean([(F * a - b).norm() ** 2 for a, b in zip(layout, actual_positions)]))

    # calculate error before refinement
    error_before = layout_error(body.nominal_marker_positions)

    # now we do the refine_marker_body function and time how long it took in simulated time
    t_start = simulation.clock.elapsed
    refined = registration.refine_marker_body(trk, placed, n_readings=30)
    t_cost = simulation.clock.elapsed - t_start

    # calculate error after refinement
    error_after = layout_error(body.nominal_marker_positions)

    # the refined layout is stored on the body and is closer to the truth
    assert all(np.allclose(r.vec, n.vec) for r, n in zip(refined, body.nominal_marker_positions))
    assert error_after < 0.5 * error_before
    # each reading costs delta_t_samp of simulated time
    assert np.isclose(t_cost, 30 * simulation.constants.SAMPLE_DELTA_T_SEC)


# ---------------------------------------------------------------- workflow

def test_estimate_base_frame_on_synthetic_data():
    """With perfect nominal data, the estimate must reproduce F_X exactly."""
    rob = simulation.create_robot("rob", Frame.eye(), seed=0)
    rng = np.random.default_rng(5)
    F_X = random_frame(rng, angle=0.3, dist=300.0)
    data = workflow.CalibrationData()
    for _ in range(10):
        th = random_theta(rng)
        data.thetas.append(th)
        data.frames.append(F_X * rob.fwd_kins(th))
    F_est = workflow.estimate_base_frame(data, rob)
    assert np.allclose(F_est.R.matrix, F_X.R.matrix, atol=1e-9)
    assert np.allclose(F_est.p.vec, F_X.p.vec, atol=1e-6)


def test_planned_poses_are_visible_and_within_limits():
    rob = simulation.create_robot("rob", Frame.eye(), seed=0)
    R_tracker = Rot(matrix=np.column_stack([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
    p_tracker = vct3(2500, 0, 300)
    F_tracker_approx = Frame(R_tracker, p_tracker)
    trk = simulation.create_tracker("trk", F_tracker_approx, seed=0)
    DESIRED_ACCURACY = 0.001 
    mbee_local_positions = [vct3(0, 0, 0), vct3(50, 0, 0), vct3(0, 50, 0), vct3(0, 0, 50)]
    poses = workflow.plan_calibration_poses(rob, trk, mbee_local_positions, DESIRED_ACCURACY) 
    for p in poses:
        assert rob.within_limits(p)
        F_tool = rob.fwd_kins(p)
        marker_positions = [F_tool * m for m in mbee_local_positions]
    assert trk.is_visible(marker_positions)


def test_measure_home_offsets_matches_encoders():
    rob = simulation.create_robot("rob", Frame.eye(), seed=9)
    offsets = workflow.measure_home_offsets(rob, n_homings=20)
    enc = rob.home_robot()
    assert offsets.shape == (6,)
    assert np.allclose(-offsets, enc, atol=4 * simulation.constants.SIGMA_RND.max())


# ---------------------------------------------------------------- kinematics

def test_improved_kinematics_initial_params_match_nominal():
    rob = simulation.create_robot("rob", Frame.eye(), seed=0)
    rng = np.random.default_rng(6)
    F_X = random_frame(rng, angle=0.3, dist=300.0)
    improved = kinematics.ImprovedKinematics(rob, F_X)
    for _ in range(10):
        th = random_theta(rng)
        F_nom, F_imp = rob.fwd_kins(th), improved.fwd_kins(th)
        assert np.allclose(F_nom.R.matrix, F_imp.R.matrix, atol=1e-9)
        assert np.allclose(F_nom.p.vec, F_imp.p.vec, atol=1e-9)
    assert np.allclose(improved.base_frame().p.vec, F_X.p.vec, atol=1e-9)


def test_improved_inv_kins_round_trip():
    rob = simulation.create_robot("rob", Frame.eye(), seed=0)
    improved = kinematics.ImprovedKinematics(rob, Frame.eye())
    improved.params = improved.initial_params() + 1e-3 * np.random.default_rng(7).normal(size=improved.params.shape)
    rng = np.random.default_rng(8)
    for _ in range(10):
        th = random_theta(rng)
        F = improved.fwd_kins(th)
        th_back = improved.inv_kins(F, seed=th)
        F_back = improved.fwd_kins(th_back)
        assert metrics.frame_error(F, F_back)[1] < 1e-6
        assert metrics.frame_error(F, F_back)[0] < 1e-8


def test_fit_recovers_synthetic_parameters():
    """
    Hint: generate CalibrationData from your own model with a known,
    non-zero parameter vector (no noise), fit from zero, and check that
    predict_in_tracker agrees with the generating model on new poses. Note that
    the parameters themselves need not match if your model has a null
    direction -- compare poses, not parameters.
    """
    rng = np.random.default_rng(0)
    N_CALIBRATION_POSES = 50
    N_TEST_POSES = 25
    rob = simulation.create_robot("rob", Frame.eye(), seed=0)
    original_params_kinematics = kinematics.ImprovedKinematics(rob, Frame.eye())
    radian_dist = np.random.default_rng(42).uniform(-0.003, 0.003, size=original_params_kinematics.params.shape)
    translation_dist = np.random.default_rng(43).uniform(-0.1, 0.1, size=original_params_kinematics.params.shape)

    # As different parameters are on different scales we selectively apply values for distribution based on parameter type
    for param in range(len(original_params_kinematics.params)): 
        if param < 3 or 7 <= param < 19:  # Base rotation, axis tilts, and tool rotation
            original_params_kinematics.params[param] = radian_dist[param]
        elif 3 <= param < 6 or 19 <= param < 22:  # Base and tool translations
            original_params_kinematics.params[param] = translation_dist[param]
        elif param == 6:  # Unused parameter due to gauge fixing
            original_params_kinematics.params[param] = 0.0
        else:  # Joint encoder distortion coefficients
            joint = (param - 22) // 2 # get the joint index (0, 1, 2, 3, 4) for the encoder distortion parameters
            if joint < 3:
                original_params_kinematics.params[param] = rng.uniform(-0.01, 0.01)
            else:
                original_params_kinematics.params[param] = rng.uniform(-1e-5, 1e-5)

    calibration_data = workflow.CalibrationData()  # using new parameters to generate synthetic calibration data
    for _ in range(N_CALIBRATION_POSES):
        theta = random_theta(rng)
        observed_frame = original_params_kinematics.predict_in_tracker(theta, original_params_kinematics.params)
        calibration_data.thetas.append(theta.copy())
        calibration_data.frames.append(observed_frame)

    blind = kinematics.ImprovedKinematics(rob, Frame.eye()) # the blind model we will fit to the data, starting from zero parameters
    result = blind.fit(calibration_data)

    for test_theta in [random_theta(rng) for _ in range(N_TEST_POSES)]:  # compare poses from the original and fitted models using frame error
        expected_frame = original_params_kinematics.predict_in_tracker(test_theta, original_params_kinematics.params)
        predicted_frame = blind.predict_in_tracker(test_theta)
        angle_error, position_error = metrics.frame_error(expected_frame, predicted_frame)
        assert angle_error < 1e-5
        assert position_error < 1e-3


