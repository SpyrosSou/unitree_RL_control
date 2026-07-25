# G1 29dof Arm Workspace Reachability Check

Arm: `left` — Goal bounds: `{'x': (0.2, 0.42), 'y': (0.08, 0.4), 'z': (0.9, 1.15)}` (starting hypothesis, reused numerically from the 23dof-era 5-DOF task — see g1_arm_env.py's module docstring)
Reachable-workspace samples: 409600 (random joint configs within real hardware limits)
Goal-box grid points checked: 2744 (14^3)

| Tolerance | Coverage (% of goal box within this distance of a reachable point) |
|---|---|
| 2.0cm | 97.5% |
| 5.0cm | 100.0% |
| 10.0cm | 100.0% |
| 15.0cm | 100.0% |

Nearest-reachable-point distance over the grid: mean=0.60cm, median=0.50cm, p90=0.88cm, max=4.55cm

## Per-octant breakdown (split the box at its own x/y/z median)

Which corner/region of the box is hardest to reach, if any — high mean distance
here means that region is systematically far from anything the arm can reach.

| x half | y half | z half | n points | mean dist (cm) | max dist (cm) |
|---|---|---|---|---|---|
| low | low | low | 343 | 0.50 | 1.12 |
| low | low | high | 343 | 0.45 | 0.85 |
| low | high | low | 343 | 0.49 | 1.43 |
| low | high | high | 343 | 0.44 | 0.96 |
| high | low | low | 343 | 0.52 | 1.82 |
| high | low | high | 343 | 0.51 | 1.01 |
| high | high | low | 343 | 1.07 | 3.76 |
| high | high | high | 343 | 0.79 | 4.55 |
