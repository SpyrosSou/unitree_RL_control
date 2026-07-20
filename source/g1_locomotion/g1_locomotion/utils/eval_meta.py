# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared run_meta.yaml writer for the validation/ eval scripts (2026-07-15).

Same idea eval_full_demo.py introduced for the integration eval a day earlier: every
eval output directory should be self-describing — until then, mapping a results folder
back to which checkpoint/args produced it required digging through whichever overnight
log happened to invoke the script. One function here instead of a copy per script, for
the same importability reason metrics_wrappers.py documents (safe to import from any
entry point after AppLauncher is up, no argparse/launch side effects of its own).
"""

import datetime
import os
import subprocess

import yaml


def _yaml_safe(value):
    """CLI args can hold anything argparse produced — coerce non-plain types to str so
    yaml.safe_dump never chokes on an exotic value (device handles, Paths, ...)."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    return str(value)


def write_eval_meta(run_dir: str, args, script_file: str) -> str:
    """Write run_meta.yaml into ``run_dir`` recording which script/args/commit produced
    this eval run. ``args`` is the script's parsed argparse namespace — recorded whole
    (AppLauncher args included), completeness beats curation here."""
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(script_file)),
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except OSError:
        git_commit = None

    meta = {
        "script": os.path.abspath(script_file),
        "git_commit": git_commit,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": {k: _yaml_safe(v) for k, v in sorted(vars(args).items())},
    }
    os.makedirs(run_dir, exist_ok=True)
    meta_path = os.path.join(run_dir, "run_meta.yaml")
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"[Eval] Run metadata: {meta_path}")
    return meta_path
