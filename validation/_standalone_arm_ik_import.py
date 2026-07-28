# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helper for CPU-only IK validation scripts (Phase 0/1 of
``ik_arm_integration_plan.md``) to import ``g1_locomotion.controllers.arm_ik`` without
triggering the real ``g1_locomotion/__init__.py``.

That package's top-level ``__init__.py`` unconditionally does ``from .tasks import *``,
which pulls in ``isaaclab_tasks`` -> ``isaaclab`` -> ``pxr`` (USD) — only importable from
*inside* Isaac Sim's own launched Kit runtime. ``controllers.arm_ik`` itself has zero
Isaac dependency, so we load it directly by file path instead, stubbing just enough of
the parent-package chain in ``sys.modules`` for its own internal relative import
(``from .weighted_moving_filter import ...``) to resolve.
"""

import importlib.util
import os
import sys
import types

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_ROOT = os.path.join(_REPO_ROOT, "source", "g1_locomotion", "g1_locomotion")


def load_arm_ik():
    """Import and return the ``g1_locomotion.controllers.arm_ik`` module."""
    pkg = types.ModuleType("g1_locomotion")
    pkg.__path__ = [_PKG_ROOT]
    sys.modules.setdefault("g1_locomotion", pkg)

    controllers_pkg = types.ModuleType("g1_locomotion.controllers")
    controllers_pkg.__path__ = [os.path.join(_PKG_ROOT, "controllers")]
    sys.modules.setdefault("g1_locomotion.controllers", controllers_pkg)

    spec = importlib.util.spec_from_file_location(
        "g1_locomotion.controllers.arm_ik",
        os.path.join(_PKG_ROOT, "controllers", "arm_ik.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["g1_locomotion.controllers.arm_ik"] = module
    spec.loader.exec_module(module)
    return module
