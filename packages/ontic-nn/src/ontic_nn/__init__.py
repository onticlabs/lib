"""ontic_nn: neural building blocks promoted from Ontic experiments.

Subpackages are imported explicitly (``ontic_nn.ptv3``, ``ontic_nn.dinov2``,
``ontic_nn.dpt``, ``ontic_nn.ppt``, ``ontic_nn.layers``); this module imports
nothing so ``import ontic_nn`` needs only the core dependencies. Every module
imports with torch + einops alone; optional accelerators (spconv, flash-attn,
the CUDA extensions under ``ext/``) are imported lazily at construction or
call time and raise an ``ImportError`` naming the extra to install.
"""

__version__ = "0.5.0"
