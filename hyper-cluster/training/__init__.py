"""The sleep-time self-evolution engine (Milestone 4).

``training.qlora`` holds the maths, the other modules the pipeline:

* ``synthetic_data`` -- mine the conversations, synthesise harder prompts;
* ``self_play`` -- draft on the mesh, let the critic keep the good answers;
* ``distributed_trainer`` -- federated, checkpointed QLoRA with an auto-merge;
* ``checkpoint_streamer`` -- the training state lives in Drive / the Hub;
* ``prompt_evolver`` -- mutate, benchmark and promote the system prompt;
* ``autopilot_evolver`` -- the idle watchdog and the preemption contract.
"""

from __future__ import annotations

from training.autopilot_evolver import (
    STATE_IDLE,
    STATE_PREEMPTED,
    STATE_TRAINING,
    AutopilotEvolver,
    EvolutionReport,
)
from training.distributed_trainer import (
    Checkpoint,
    DistributedTrainer,
    PreemptedError,
    PreemptionToken,
    TrainingReport,
    TrainingTrace,
)

__all__ = [
    "STATE_IDLE",
    "STATE_PREEMPTED",
    "STATE_TRAINING",
    "AutopilotEvolver",
    "Checkpoint",
    "DistributedTrainer",
    "EvolutionReport",
    "PreemptedError",
    "PreemptionToken",
    "TrainingReport",
    "TrainingTrace",
]
