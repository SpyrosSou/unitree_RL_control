# G1 29dof Arm Workspace Reachability Check

Arm: `left` — Goal bounds: `{'x': (0.2, 0.42), 'y': (0.08, 0.4), 'z': (0.9, 1.15)}` (starting hypothesis, reused numerically from the 23dof-era 5-DOF task — see g1_arm_env.py's module docstring)
Reachable-workspace samples: 409600 (random joint configs within real hardware limits)
Goal-box grid points checked: 2744 (14^3)

| Tolerance | Coverage (% of goal box within this distance of a reachable point) |
|---|---|
| 2.0cm | 97.0% |
| 5.0cm | 99.9% |
| 10.0cm | 100.0% |
| 15.0cm | 100.0% |

Nearest-reachable-point distance over the grid: mean=0.61cm, median=0.50cm, p90=0.95cm, max=5.21cm

## Per-octant breakdown (split the box at its own x/y/z median)

Which corner/region of the box is hardest to reach, if any — high mean distance
here means that region is systematically far from anything the arm can reach.

| x half | y half | z half | n points | mean dist (cm) | max dist (cm) |
|---|---|---|---|---|---|
| low | low | low | 343 | 0.49 | 1.02 |
| low | low | high | 343 | 0.45 | 0.94 |
| low | high | low | 343 | 0.48 | 1.23 |
| low | high | high | 343 | 0.43 | 1.02 |
| high | low | low | 343 | 0.55 | 1.62 |
| high | low | high | 343 | 0.50 | 1.04 |
| high | high | low | 343 | 1.19 | 5.21 |
| high | high | high | 343 | 0.82 | 4.54 |
