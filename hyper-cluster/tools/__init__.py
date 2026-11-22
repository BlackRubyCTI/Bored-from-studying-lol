"""Tool-use layer.

``sandbox`` exposes three tool families to the model:

* an isolated Python executor (real rlimits, no network, no process spawning),
* live web search / scraping,
* a local file analyzer.

The head node calls them through :class:`tools.sandbox.ToolExecutor`, and the
model talks to them through the JSON tool-call protocol defined in the same
module.
"""

from __future__ import annotations

__all__ = ["sandbox"]
