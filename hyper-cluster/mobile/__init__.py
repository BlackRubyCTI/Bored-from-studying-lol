"""The Mobile Mission Control (Milestone 5).

* ``mobile.dashboard`` -- the touch-first PWA and its API proxy;
* ``mobile.notifier`` -- the phone alerts (Telegram and webhooks);
* ``mobile.cli``      -- the Termux command line;
* ``mobile.qr``       -- the QR code of the dashboard (used by the installer).
"""

from __future__ import annotations

__all__ = ["cli", "dashboard", "notifier", "qr"]
