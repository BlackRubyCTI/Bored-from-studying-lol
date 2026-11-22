"""Mesh worker package.

The worker is a self-benchmarking GPU node in the Hyper-Cluster mesh:

``worker_agent``      the runtime: registration, heartbeat, SSE chat, drain.
``inference_engine``  model loading + streaming (llama.cpp / OpenAI upstream / simulator).
``telemetry``         hardware + latency benchmarking at join time.
``mesh``              Tailscale mesh automation (join, advertise, discover peers).
``bootstrap.sh``      zero-touch provisioning with a pre-compiled binary cache.
"""

from __future__ import annotations

__all__ = ["config", "inference_engine", "mesh", "telemetry", "worker_agent"]
