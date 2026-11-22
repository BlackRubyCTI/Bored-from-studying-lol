"""Head node package: the control plane of Hyper-Cluster Omega.

Modules
-------
``app``            FastAPI application factory.
``serve``          CLI entry point (``python -m head_node.serve``).
``orchestrator``   Request brain: routing, hold/requeue, council, speculation.
``scheduler``      Latency-aware worker selection.
``worker_registry``Authoritative view of the mesh.
``hold_queue``     In-memory hold-and-requeue buffer.
``request_store``  Ledger of request outcomes (deferred work stays recoverable).
``worker_client``  HTTP/SSE client used to talk to workers.
``edge_bridge``    Multi-account bridge to the free inference APIs.
``council``        Mixture-of-Agents deliberation.
``events``         Async event bus feeding the live dashboard.
"""

from __future__ import annotations

__all__ = [
    "app",
    "council",
    "edge_bridge",
    "events",
    "hold_queue",
    "orchestrator",
    "request_store",
    "scheduler",
    "serve",
    "worker_client",
    "worker_registry",
]
