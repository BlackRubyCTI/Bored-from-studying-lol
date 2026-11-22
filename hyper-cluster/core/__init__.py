"""Hyper-Cluster Omega :: shared core layer.

This package holds everything that must behave identically on the head node,
on every mesh worker, and inside the mobile control plane:

* :mod:`core.config`      -- layered configuration (defaults -> .env -> env vars).
* :mod:`core.logging_setup` -- structured JSON logging with request-scoped context.
* :mod:`core.protocol`    -- the versioned wire protocol spoken across the mesh.
* :mod:`core.credentials` -- multi-account credential vault primitives.
* :mod:`core.providers`   -- canonical registry of every external provider we can drive.
* :mod:`core.errors`      -- typed exception hierarchy.
* :mod:`core.ids`         -- identifier + clock helpers.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
