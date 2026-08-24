"""Convention-locked tensor geometry for Ontic projects.

Conventions
-----------
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
"""
