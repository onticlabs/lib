# Franka Duo placement in robot-dextris

The first frame of `alignment_79a476` has a saved Franka FR3 Duo mesh placement in
the recording's calibrated world. This is a visual estimate of one pose, not
measured joint state or a robot trajectory. Other recordings have different arm
poses; the hand-equipped recordings also have different end effectors.

Open `robot-dextris`, select `alignment_79a476`, and set the frame to **0**. In
**4. Display → Hands & workspace**, enable **Show robot**. Advancing to another
frame hides the mesh; returning to frame 0 restores it.

The placement maps the Duo cell's mount-base coordinates into the camera-calibrated
world. Its translation is approximately **(-0.603, 0.004, -0.413) metres** and its
yaw is **3.52 degrees**. The world datum is the calibration board, so the negative
base height is expected. Mesh scale stays at one; both arms use the physical Duo
mount transforms and FR3 joint limits from the sibling robotics repository.

The base was fitted to the centres of the two fixed Franka badges in P03, P04,
P05 and P07, with a soft upright prior. Badge residuals are about 1 pixel at
960×540. This describes agreement with manually selected image points; it is not
a measured physical accuracy. The fourteen arm angles were initialized from
approximate joint/gripper landmarks, then refined against image edges. The arm
and wrist fit remains approximate. Cables, wrist cameras and the actual central
mount housing are absent from the model; grippers are assumed open.

**P06 was excluded from fitting.** Its projections of both stationary badges
disagree by approximately 48 pixels horizontally at 960×540. A separate
checkerboard projection check also showed an inconsistency in that view. The
original camera calibration is unchanged. All overlays use that original
calibration, including its lens distortion, so the P06 mismatch remains visible.

The projection overlays (`all_cameras.jpg`, a `viewer.png` screenshot and one image per
camera) are local render outputs under `artifacts/robot_alignment/robot-dextris/alignment_79a476/`,
which Git ignores; regenerate them with the command below.

The saved [alignment](../packages/ontic-data/src/ontic_data/datasets/robot_dextris_alignments/alignment_79a476.json)
contains the `xyz + wxyz` base pose, arm joint angles in radians, the exact camera
calibration hash, source model revisions and limitations. The loader refuses to
apply it to a different camera calibration or frame. A recording-local
`robot_alignment.json` takes precedence over this packaged fit.

Regenerate the projection images with:

```bash
uv run --no-sync python scripts/render_robot_alignment.py \
  /mnt/fast/mzhobro/trailer-demo/alignment_79a476 \
  packages/ontic-data/src/ontic_data/datasets/robot_dextris_alignments/alignment_79a476.json \
  --output artifacts/robot_alignment/robot-dextris/alignment_79a476
```

The renderer needs OpenCV, MuJoCo and the robotics checkout. It uses EGL by
default; `MUJOCO_GL=osmesa` selects an alternative offscreen backend. The regular
interactive mesh overlay needs no offscreen rendering.

## Teleoperation recordings checked

The archive `teleop_franka_recordings-20260917T163942Z-1-001.zip` was inspected on
2026-09-17 for measured joint angles. Its 183 files comprise 71 MP4 camera videos,
16 recording manifests, 48 encoder logs and 48 encoder progress files. The text
sidecars contain no joint states or references to a robot-state export. The
manifests describe independent UVC camera recordings without hardware
synchronization; encoder progress fields are not robot joint measurements.

Comparing sampled frames from the ZED videos with the dataset gave these results:

| Dataset recording | Archive comparison |
| --- | --- |
| `alignment_79a476` | `robot_calib_video` failed immediately: both existing videos have zero frames and the third camera video is missing. It cannot establish a match. |
| `demo_0750cb` | The September 16 metal-parts recordings around 14:43–14:46 UTC have a similar table setup. An exact take and frame correspondence were not established. |
| `demo_08bb57`, `demo_159de8`, `demo_17d4c1` | These show mounted anthropomorphic hands. No matching take was identified in the archive; `wuji_hand_pick_success_01` instead shows a detached hand being picked up. |

No measured angles were imported. A timestamped robot-state export and a verified
correspondence to the dataset frames are still needed to replace the estimated
pose. Similar objects or filesystem modification times alone do not establish
that correspondence.
