"""ontic_lib: shared, promoted code for Ontic experiments.

Layout follows the function-type split used by PyTorch3D/Kaolin: ``transforms``
(SO(3)/SE(3)/Sim(3) math), ``camera`` (pinhole intrinsics/projection/rays),
``ops`` (batched tensor ops: sampling, serialization, alignment, point clouds),
``depth`` (lifting and metric alignment), ``splats`` (3D Gaussian helpers),
``metrics`` (evaluation protocols), plus training infrastructure at the top
level (``tracking``, ``checkpoint``, ``distributed``).

Conventions (binding for all geometric APIs)
--------------------------------------------
* Tensors are torch-first and preserve arbitrary leading batch dimensions.
* Points are stored as ``(..., 3)`` rows, but transforms use the column-vector
  equation ``x_dst = T_dst_from_src @ x_src``. The equivalent row operation is
  ``x @ R.T + t``.
* A camera pose is camera-to-world. APIs that need world-to-camera say so
  explicitly rather than using the ambiguous name ``extrinsics``.
* Camera coordinates follow OpenCV: +x right, +y down, +z forward.
* Intrinsics may be pixel-space or normalized. Function names and docstrings
  state which representation is required. Normalized pixel centers are
  ``((x + 0.5) / width, (y + 0.5) / height)``.
* Canonical quaternions are real-first ``(w, x, y, z)``. Explicit ``xyzw``
  conversion functions exist for SciPy, RoMa, and legacy checkpoints.
* Depth APIs distinguish camera-z depth from Euclidean ray distance explicitly.
* Ops with both a torch reference and a vendored CUDA kernel take
  ``impl="torch" | "cuda" | "auto"``. The default is ``"auto"`` only where the
  kernel is measured bit-exact against the reference (Morton/Hilbert codes);
  FPS defaults to ``"torch"`` because its kernel can pick different (equally
  spread) indices on exact distance ties. FPS starts at index 0 and greedily
  maximizes squared distance in float32.
"""
