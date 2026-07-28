# Vendored URDF

`g1_29dof.urdf` is a copy of `~/Elm/Code/g1_simulation/ros2_ws/src/g1_navigation/description_files/urdf/g1_29dof.urdf`
(`g1_simulation`, a sibling repo, is the source of truth for this asset — copied here
2026-07-27 so `controllers/arm_ik.py` doesn't depend on a path outside this repo).

Only the kinematic tree is used (`pin.buildModelFromUrdf` — no `package_dirs` passed),
so the mesh files this URDF's `<mesh filename="...">` tags reference are **not** vendored
alongside it and don't need to resolve; loading fails only if the kinematic structure
itself changes upstream. If `g1_simulation`'s copy is updated (joint limits, link
lengths, new joints), re-copy this file and re-run `validation/check_arm_ik_solver.py`.
