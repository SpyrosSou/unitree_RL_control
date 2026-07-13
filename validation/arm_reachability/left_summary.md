# Arm Workspace Reachability Check

Arm: `left` — Goal bounds: `{'x': (0.15, 0.42), 'y': (0.05, 0.4), 'z': (0.9, 1.15)}`
Reachable-workspace samples: 409600 (random joint configs within real hardware limits)
Goal-box grid points checked: 2744 (14^3)

| Tolerance | Coverage (% of goal box within this distance of a reachable point) |
|---|---|
| 2.0cm | 65.3% |
| 5.0cm | 84.0% |
| 10.0cm | 98.5% |
| 15.0cm | 100.0% |

Nearest-reachable-point distance over the grid: mean=2.24cm, median=0.76cm, p90=6.56cm, max=11.90cm

## Per-octant breakdown (split the box at its own x/y/z median)

Which corner/region of the box is hardest to reach, if any — high mean distance
here means that region is systematically far from anything the arm can reach.

| x half | y half | z half | n points | mean dist (cm) | max dist (cm) |
|---|---|---|---|---|---|
| low | low | low | 343 | 4.61 | 11.90 |
| low | low | high | 343 | 4.13 | 11.76 |
| low | high | low | 343 | 1.82 | 10.30 |
| low | high | high | 343 | 1.66 | 9.88 |
| high | low | low | 343 | 0.81 | 4.08 |
| high | low | high | 343 | 0.90 | 4.77 |
| high | high | low | 343 | 1.93 | 9.33 |
| high | high | high | 343 | 2.08 | 9.89 |
