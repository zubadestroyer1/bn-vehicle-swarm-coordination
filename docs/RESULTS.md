# Initial implementation results

Measured locally on November 12, 2026. This is an implementation check and exploratory comparison, not a final project conclusion. The complete per-episode CSV, summary JSON, training report, plots, and environment record are in [examples/validation](../examples/validation/).

## Reproduction

```bash
python -m swarm train --scenes 700 --epochs 60 --seed 7 --output outputs/training
python -m swarm simulate --duration 120 --seed 11 --output outputs/demo
python -m swarm evaluate --duration 120 --seeds 11 12 13 \
  --scenarios nominal busy noisy no-communication --output outputs/evaluation
python -m swarm benchmark --seed 7 --output outputs/benchmark
python -m unittest discover -s tests -v
```

The model uses separate synthetic scene sets for training, validation, and testing. The evaluation is 60 episodes: five policies, three shared seeds, and four conditions. Nominal total arrival intensity is 0.45 vehicles/s; busy intensity is 0.81 vehicles/s. Each episode runs for 120 s without warm-up or a drain period. The noisy condition increases measurement error, dropout and packet delay. The no-communication condition retains local sensing but sends no requested messages.

## Held-out edge prediction

| Metric | Value |
| --- | ---: |
| Test pairs from independent scenes | 5,742 |
| Positive-label fraction | 46.26% |
| Accuracy at threshold 0.5 | 90.00% |
| ROC AUC | 0.9567 |
| Brier score | 0.07766 |
| Log loss | 0.25943 |
| Expected calibration error, 10 bins | 0.00861 |

These measure synthetic route/occupancy interaction recognition. They do not measure driving safety, causal identification, or delay CPD accuracy. The learning and calibration plot is [training.png](../examples/validation/training.png).

## Nominal traffic

Values are arithmetic means across seeds 11, 12 and 13. Delay includes unfinished demand and upstream queues, so a controller cannot improve this measure merely by completing only easy trips.

| Controller | Accrued delay / arrival (s) | Throughput (vehicles/h) | Stops / arrival | Mean structure update (ms) |
| --- | ---: | ---: | ---: | ---: |
| Fixed signal | 38.86 | 520 | 2.02 | 0 |
| FCFS | 20.51 | 880 | 2.15 | 0 |
| Independent beliefs | 18.50 | 960 | 2.03 | 0 |
| Greedy BIC graph | 18.55 | 950 | 2.01 | 1.683 |
| Neural graph | 17.70 | 980 | 2.01 | 0.116 |

The neural policy has about 54.5% less accrued delay than the conservative fixed-signal reference in this small sample. Its paired mean delay improvement is 0.80 s relative to independent beliefs and 0.84 s relative to greedy BIC. The corresponding normal-approximation 95% half-widths are 1.36 s and 1.41 s: these samples do **not** establish a reliable advantage over either stronger comparison. The fixed-signal baseline is not actuated or optimized.

## Other conditions

| Condition | Signal delay (s) | FCFS delay (s) | Independent delay (s) | Greedy delay (s) | Neural delay (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Busy | 47.13 | 36.65 | 33.86 | 33.83 | 34.09 |
| Noisy | 42.48 | 22.24 | 20.12 | 20.23 | 20.05 |
| No communication | 38.87 | 20.31 | 18.60 | 18.62 | 16.97 |

The neural method is slightly worse than independent and greedy control in the busy condition. Stops are not consistently improved across scenarios. The appropriate result is that the full loop works and neural graph construction is fast; the control-quality hypothesis still needs stronger experiments and model improvements. See [comparison.png](../examples/validation/comparison.png) and the paired differences in [summary.json](../examples/validation/summary.json).

All 60 episodes recorded zero collision pairs **with the shared privileged safety shield enabled**. This is not a learned-safety guarantee. For example, the seed-11 neural replay uses 47 stop-line projections and 2,435 following-distance projections over 120 s. At that horizon it has completed 32 of 61 arrivals; 29 remain unfinished. The shield is materially involved in the result, and the demand is congested.

## Structure-update scaling

One warm-up followed by eight timed updates per method and size, with a 48-observation sliding history and parent cap of three:

| Vehicles | Neural median (ms) | Ordered greedy BIC median (ms) |
| --- | ---: | ---: |
| 8 | 0.106 | 0.741 |
| 16 | 0.157 | 3.013 |
| 32 | 0.248 | 14.071 |
| 64 | 0.629 | 57.582 |

These are timings of these implementations on synthetic snapshots. Neural training cost, inference, planning and environment simulation are excluded, and the methods optimize different objectives. Both examine pairwise candidates; the table does not establish linear-time scaling. Timings vary with hardware and process load. See [scaling.json](../examples/validation/scaling.json).

## Validation status

The regression suite covers route continuity and conflicts, arrival kinematics, uncertainty growth and delayed-observation handling, noisy-OR probabilities, temporal persistence, DAG/parent bounds, BIC parent recovery, held-out learning and checkpoint round trips, collision shielding, car following, stop hysteresis, communication accounting and budgets, empty episodes, seed reproducibility, noisy traffic, and all five policies moving traffic.

Training, simulation, evaluation, and scaling plots were generated and visually inspected. The standalone HTML replay was generated and its JavaScript syntax checked. Browser interaction could not be checked in this environment because its browser policy rejects local-file URLs. The separate static simulation preview is in [preview.png](../examples/validation/preview.png).
