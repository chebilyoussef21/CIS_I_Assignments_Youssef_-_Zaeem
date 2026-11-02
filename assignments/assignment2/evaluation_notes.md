# Assignment 2 — Evaluation and Improvement Notes

## Run 1 (SEED = 1)

### How performance is evaluated

Performance is measured in two notebook steps, and both use the same two functions from `src/student/metrics.py`. `metrics.frame_error(F_a, F_b)` compares two frames and returns two numbers: the angle error in radians and the position error in mm. `metrics.summarize_errors(list)` takes a list of those pairs and returns the count, the rms (typical) error and the max (worst) error for angle and for position.

Step 9 tests forward kinematics, which asks "can the model predict where the robot is?". It takes 729 test poses (a 3×3×3×3×3×3 grid of joint values, all of them visible to the tracker). For each pose it gets the true end-effector pose from the simulator with `rob.actual_ee_pose(theta)`, which is a hidden answer used only for grading. It then asks each model where it thinks the end effector is: the nominal model through `F_X_nom * rob.fwd_kins(theta)`, and ours through `improved.predict_in_tracker(theta)`. Each guess is compared with the truth using `frame_error`. The robot never moves in this step; it is pure prediction.

Step 10 tests inverse kinematics, which asks "can the model send the robot where we want?". It picks 40 of the test poses and uses their true poses as targets. Each model works out which joint values should reach the target with `inv_kins`. The robot then actually moves there with `rob.move_joints`, and where it really ended up (`mbee_placed.actual_placement`) is compared with the target using `frame_error`. Because the robot really moves, this step costs simulated time.

The summary cell prints one row for each of the four cases (forward/inverse × nominal/ours). It shows the rms and max angle error, the rms and max position error, and a "met?" column. "met?" is True only if the worst angle is at most 0.001 rad and the worst position is at most 0.2 mm, so it uses the max, not the rms.

### Results

| case | rms angle (rad) | max angle (rad) | rms pos (mm) | max pos (mm) | met? |
|---|---|---|---|---|---|
| fwd nominal | 3.26e-2 | 5.40e-2 | 7.914 | 16.458 | False |
| fwd improved | 1.89e-3 | 2.68e-3 | 0.183 | 0.370 | False |
| inv nominal | 3.45e-2 | 5.38e-2 | 8.240 | 14.091 | False |
| inv improved | 1.88e-3 | 2.71e-3 | 0.179 | 0.322 | False |

Calibration took 496.7 s for 64 samples (soft budget 900 s). Testing took 1058 s for 40 targets × 2 moves, mostly because the wrist turns at only 0.1 rad/s.

### What the results mean

Before calibration the robot was off by about 8 mm and 0.03 rad. After calibration it is off by about 0.18 mm and 0.002 rad, so the calibration clearly works. It does not meet the targets yet, though. For position, the average (0.18 mm) is under the 0.2 mm target, but the worst case (0.37 mm) is not. For angle, even the average is about twice the 0.001 rad target.

The inverse-kinematics rows have almost the same error as the forward rows. That means the robot's random jitter (`sigma_rnd`) is not what limits us; the remaining error comes from the model itself. The angle error is also spread evenly (rms 1.9e-3, max 2.7e-3), which means it is not caused by a few bad poses but by something in the model that is a little off everywhere.

### Finding the cause: comparing fitted parameters with the truth (Step 8)

To find which part of the model is off, we compared the output of `improved.parameters_report()` with the hidden true values from `rob.ground_truth_parameters()`. The true values are only used for debugging, never in the calibration itself.

The angles between the axes matched the truth almost exactly (for example d2–d3 is 88.31° estimated and 88.31° true, and d4–d5 is 91.81° and 91.81°). The axis vectors themselves looked wrong: axes 2, 3, 5 and 6 were all rotated by about 0.2 rad around x. That is not a real error. Rotating the base about x and rotating those axes back the other way gives exactly the same predictions, so the fit cannot tell them apart. This is the "rotation of the base about d1" the handout asks about. The same rotation explains why the first value of the F_we twist was −0.216 instead of −0.0001; the other F_we values matched. Because of this hidden rotation, the Jacobian rank should have been 33, not the reported 34, which means the rank check's tolerance is too loose. The joint offsets come from homing (`workflow.measure_home_offsets`) and were only slightly different from the truth.

The joint scales were where the real problem showed up. A "joint slope" describes how a joint's encoder error grows as the joint turns. The simulator models each joint's error mostly as a straight line, error(θ) = offset + slope × θ. The offset is how wrong the joint is at θ = 0, and the slope is how much extra error is added for every 1 rad (or 1 mm) the joint moves. For example, with slope = 0.001, at θ = 2 rad the joint is off by an extra 0.001 × 2 = 0.002 rad.

We did not choose the slope ourselves. We chose the shape of the model in `fwd_kins`, which corrects each joint by home_offset + c1 × (θ/scale) + c2 × (θ/scale)². The `least_squares` fit inside `improved.fit` then picks the numbers c1 and c2 (stored in `params[22:]`) that match the data best. The slope is the c1 term divided by the joint's scale.

The 0.0017 value came from this line of the Step 8 output:

```
estimated joint scales : [0.9998 0.9996 0.9996 0.9983 0.9999 0.9999] | true: [-0.0002 -0.0002 -0.0002 -0. 0. -0.]
```

Our `parameters_report` reports each joint scale as 1 + slope, so the slope is the printed value minus 1. For joint 4 that is 0.9983 − 1 = −0.0017. The "true" list is printed differently: it shows the slope directly, and joint 4's true value `-0.` means about 0. Our fit therefore thinks joint 4 drifts by 0.0017 rad per radian turned, when it really does not drift at all. The slope error is the difference, 0.0017. That is about 35 times larger than the largest physically allowed wrist slope (5e-5). Joint 1 matched the truth, and joints 2 and 3 were a little off (−0.0004 against −0.0002) but still within the allowed bound.

To check whether this one error explains the result, we multiplied the slope error by the largest joint 4 angle in the test grid (1.3 rad): 0.0017 × 1.3 ≈ 0.0022 rad. At those poses our model gets the wrist angle wrong by about 0.002 rad, which matches the measured angle error (0.0019 rms, 0.0027 max). So the wrong joint 4 slope is the most likely main cause.

### Why the fit got joint 4 wrong

Joint 4 rotates about the x axis, and the tool arm also points along x, so turning joint 4 spins the calibration body in place without moving its centre. Joint 4's slope can therefore only be seen in the rotation part of the data. In `residuals`, the rotation errors (in rad, around 0.001) and the position errors (in mm, around 0.1) are added together with equal weight. Rotation errors are so much smaller as numbers that the fit almost ignores them, so it was free to give joint 4 a wrong slope. In addition, every calibration pose was read only once, so each orientation reading carries about 1e-3 rad of noise.

### Planned changes for run 2

1. Weight the rotation part of each residual so that rotation errors count as much as position errors.
2. Remove the hidden rotation about x by fixing the x-part of d2's tilt to zero.
3. Make `parameters_report` return the slope instead of 1 + slope, so it matches the truth's format.
4. Tighten the Jacobian rank tolerance and print the smallest singular values.
5. If needed, take several readings at each wrist pose and average them.

## Run 2 (SEED = 1): rotation weighting + hidden x rotation removed

For this run we made two changes. In `residuals`, the rotation part of each residual is now multiplied by 100 (`eta[:3] *= 100.0`), so a rotation error of 0.002 rad counts about as much as a position error of 0.2 mm. In `_corrected_axes`, the x-part of d2's tilt is fixed to zero, which removes the hidden base rotation about x so the fit has one unique answer. The pose plan and the number of samples were not changed.

| case | rms angle (rad) | max angle (rad) | rms pos (mm) | max pos (mm) | met? |
|---|---|---|---|---|---|
| fwd nominal | 3.26e-2 | 5.39e-2 | 7.859 | 16.366 | False |
| fwd improved | 2.23e-4 | 4.14e-4 | 0.088 | 0.162 | True |
| inv nominal | 3.45e-2 | 5.37e-2 | 8.188 | 14.027 | False |
| inv improved | 2.79e-4 | 4.80e-4 | 0.093 | 0.147 | True |

Calibration time stayed at 496.7 s for 64 samples, and testing took 1057.9 s.

Both targets are now met, for forward and inverse kinematics. The worst angle error dropped from 0.0027 rad to 0.0004 rad, which is under the 0.001 rad target by more than half, and the worst position error dropped from 0.37 mm to 0.16 mm. The weighting alone fixed the joint 4 problem: once rotation errors counted in the fit, the fit could no longer give joint 4 a wrong slope. Inverse kinematics is now slightly worse than forward (4.8e-4 against 4.1e-4 rad). That small gap is the robot's random jitter (`sigma_rnd`) on the real move, which now shows because the model error is so small. The nominal rows changed slightly from run 1 because the simulator's random noise is different each run.

## Run 3: seed sweep (seeds 1–10), same code as run 2

To check that the calibration works for any robot, not just seed 1, we added a "Seed sweep" cell at the end of the notebook. For each seed it rebuilds the simulation, runs the full calibration (homing, pose plan, data collection, fit), then runs the Step 9 forward test on the full test grid and the Step 10 inverse test on 40 targets, using the same `metrics.frame_error` and `metrics.summarize_errors` functions as before. The `SEED` value decides the robot's hidden errors (base placement, axis tilts, joint offsets and slopes, tool frame error, tracker distortion), so each seed is effectively a different robot. The table is saved to `seed_sweep.csv`.

| seed | fwd rms ang | fwd max ang | fwd rms pos | fwd max pos | inv max ang | inv max pos | t_cal (s) | met? |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.97e-4 | 8.50e-4 | 0.103 | 0.196 | 9.19e-4 | 0.174 | 397.9 | True |
| 2 | 4.21e-4 | 6.57e-4 | 0.103 | 0.193 | 7.42e-4 | 0.210 | 397.9 | False |
| 3 | 4.89e-4 | 8.36e-4 | 0.126 | 0.209 | 9.21e-4 | 0.174 | 398.0 | False |
| 4 | 3.49e-4 | 6.05e-4 | 0.072 | 0.142 | 6.91e-4 | 0.125 | 398.0 | True |
| 5 | 7.48e-4 | 1.42e-3 | 0.127 | 0.247 | 1.53e-3 | 0.249 | 397.9 | False |
| 6 | 3.78e-4 | 6.22e-4 | 0.126 | 0.230 | 6.86e-4 | 0.200 | 397.9 | False |
| 7 | 3.46e-4 | 5.81e-4 | 0.084 | 0.144 | 6.71e-4 | 0.132 | 397.9 | True |
| 8 | 4.63e-4 | 7.65e-4 | 0.091 | 0.157 | 9.46e-4 | 0.147 | 397.9 | True |
| 9 | 5.90e-4 | 1.09e-3 | 0.079 | 0.155 | 1.05e-3 | 0.159 | 397.9 | False |
| 10 | 4.40e-4 | 5.90e-4 | 0.085 | 0.167 | 8.23e-4 | 0.154 | 397.9 | True |

Five of the ten seeds met both targets. Every failing seed failed by only a small amount: the worst position errors were 0.209 to 0.249 mm against a 0.2 mm target, and the worst angle errors were 1.05e-3 to 1.53e-3 rad against 1e-3 rad. No seed had a large error, which means the model itself is right and the remaining error is mostly measurement noise. Two more signs point to noise. First, the same seed gives different results on different runs: seed 1 had a worst forward angle of 4.1e-4 rad in run 2 but 8.5e-4 rad here, and seed 10 failed in its single notebook run but passed here. Second, the Step 8 parameters for seed 10 matched the truth well. The calibration time in the sweep (about 398 s) is lower than in the notebook (about 497 s) because the sweep skips the Step 6 baseline poses.

The cause is that every calibration pose is read only once, so each reading carries the tracker's full 0.1 mm per-marker noise into the fit. The planned fix for run 4 is to take several tracker readings at each pose and average them. Re-reading without moving costs only 0.03 s per reading, and there is about 500 s of the 900 s budget left.

## Run 4: seed sweep with 10 averaged readings per pose

For this run, `collect_calibration_data` takes all `repeats` readings at each pose and averages them with `average_frames` (before, it kept only the first visible reading). Both the Step 7 cell and the sweep cell now use `repeats=10`. Nothing else changed.

| seed | fwd rms ang | fwd max ang | fwd rms pos | fwd max pos | inv max ang | inv max pos | t_cal (s) | met? |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.55e-4 | 2.26e-4 | 0.063 | 0.091 | 4.49e-4 | 0.105 | 415.2 | True |
| 2 | 1.62e-4 | 2.70e-4 | 0.058 | 0.098 | 4.84e-4 | 0.087 | 415.2 | True |
| 3 | 1.64e-4 | 2.76e-4 | 0.074 | 0.104 | 4.34e-4 | 0.104 | 415.2 | True |
| 4 | 1.71e-4 | 3.15e-4 | 0.075 | 0.102 | 4.15e-4 | 0.101 | 415.2 | True |
| 5 | 1.79e-4 | 3.32e-4 | 0.080 | 0.119 | 4.22e-4 | 0.111 | 415.2 | True |
| 6 | 1.80e-4 | 2.92e-4 | 0.058 | 0.099 | 6.40e-4 | 0.094 | 415.2 | True |
| 7 | 1.96e-4 | 3.17e-4 | 0.091 | 0.140 | 4.88e-4 | 0.115 | 415.2 | True |
| 8 | 1.99e-4 | 3.55e-4 | 0.073 | 0.117 | 5.35e-4 | 0.109 | 415.2 | True |
| 9 | 1.58e-4 | 2.35e-4 | 0.076 | 0.109 | 5.17e-4 | 0.113 | 415.2 | True |
| 10 | 2.22e-4 | 3.55e-4 | 0.082 | 0.116 | 4.80e-4 | 0.108 | 415.2 | True |

All ten seeds now meet both targets, for both forward and inverse kinematics. Across all seeds, the worst forward angle error is 3.6e-4 rad (about a third of the 1e-3 target) and the worst forward position error is 0.140 mm (target 0.2 mm). The worst inverse errors are 6.4e-4 rad and 0.115 mm. Averaging 10 readings cut the forward errors roughly in half compared with run 3, which confirms that the failures in run 3 came from tracker noise and not from the model. The extra readings cost only about 17 s of calibration time (415 s against 398 s), well within the 900 s budget. Inverse kinematics is now noticeably worse than forward on angle (about 4–6e-4 rad against 2–4e-4 rad). That gap is the robot's own random jitter (`sigma_rnd`) on each real move, which no calibration can remove, so it acts as a floor for the inverse result.
