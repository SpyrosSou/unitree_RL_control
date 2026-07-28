# G1 29dof Arm IK Coverage Check — right arm

Goal bounds (ground-relative): `{'x': (0.2, 0.42), 'y': (-0.4, -0.08), 'z': (0.9, 1.15)}`
Grid points swept: 2744

| Tolerance | Coverage |
|---|---|
| 2.0cm | 58.5% |
| 5.0cm | 80.0% |
| 10.0cm | 96.9% |
| 15.0cm | 100.0% |

Error: mean=2.89cm, median=1.48cm, p90=7.12cm, max=15.51cm
IPOPT solve success rate: 100.0%
Solutions within joint limits: 100.0%
Solve time: mean=1.89ms, p90=2.23ms, max=3.56ms

Gate (>=95% within 2cm): FAIL (58.5%)
