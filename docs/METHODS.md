# Model and experimental method

## Relationship to the proposal

The implementation follows the supplied *Final Project Proposal (CS 238) — Kieran + Aadi + Arthur*, titled *Neurally Updated Bayesian Networks for Autonomous Vehicle Swarm Coordination*. The proposal specifies a research direction rather than a simulator, network factorization, training objective, or solver. The choices below make that direction executable and keep the simplifications visible.

Dynamic Bayesian networks represent temporal processes using factored state variables and conditional dependencies. Our implementation uses this conventional two-slice interpretation, with an additional observation-conditioned graph that can change as vehicles arrive and depart. Background: [Kevin Murphy, Dynamic Bayesian Networks: Representation, Inference and Learning, 2002](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2002/8174.html).

## Simulator and decision process

- **State:** active vehicle identities, approach lanes, scalar route progress, speed, fixed latent maneuver, desired cruise speed, upstream demand, in-flight messages, and reservations. Intent remains fixed within a trip.
- **Actions:** desired speed for each tracked vehicle, an ordered set of proposed crossing grants, and identities to contact. Acceleration and braking are bounded before shielding.
- **Transitions:** route motion every 0.25 s, bounded additive acceleration noise, Poisson demand, deterministic route geometry, and delayed communication. There are no pedestrians, traffic incidents, lane changes, occlusion geometry, or continuous steering decisions.
- **Observations:** noisy route-progress and speed readings within sensor range, categorical maneuver labels, and more precise requested messages. Registration supplies identity and approach; exit detection retires tracks.
- **Reward:** `20 * completed - delay_deficit - 2 * stops - 1000 * new_collision_pairs - 0.02 * attempted_messages` by default. Delay deficit integrates the fraction of free-flow speed lost, plus upstream queue time. Coefficients are configurable.
- **Policy approximation:** an observation-based finite priority scheduler replans every step. It does not solve the complete POMDP, perform policy-gradient learning, or optimize the full infinite-horizon reward.

Each incoming lane has one lane of traffic with shared left/straight/right access. Cubic curves or straight segments connect it to an outgoing lane. A route-conflict matrix tests whether swept paths approach within the vehicle diameter plus a margin; shared incoming and outgoing lanes are always treated as conflicting. Opposing straight routes can run concurrently.

The geometric conflict area is a square with side 18 m. Vehicles use circular footprints of radius 1.7 m for collision checks. Synchronous swept line segments between consecutive positions check minimum separation, including between-step encounters. The approximation follows route chords within each small timestep, not exact continuous curves.

## Beliefs and communication

Each track has a Gaussian belief over `[progress, speed]` with a constant-acceleration prediction using the controller's commanded acceleration. It does not get the realized driver noise or safety-projected velocity. Updates use Gaussian measurement noise and the Joseph-form covariance update.

Delayed observations are extrapolated from measurement time using reported velocity; measurement covariance is propagated and inflated for unknown acceleration. This is an approximation to out-of-sequence filtering, not an exact smoother. Older-than-last-accepted packets are discarded. Position/velocity uncertainty grows during missed observations.

Intent is a three-category belief. Sensor labels are correct with probability 0.78; messages with probability 0.97. The remaining probability is split over the other labels. A small forgetting term prevents a confidently wrong early label from becoming irreversible.

The budget counts attempted messages, including dropped transmissions. A requested packet arrives after `1 + UniformInteger(0, max_message_delay)` simulation ticks, so even a configured zero extra delay has one control-cycle latency. Packet loss and delays are independent of the sensor stream. One-second request cooldowns prevent every-step polling of the same vehicle. Belief-based controllers rank uncertainty, intent entropy, proximity, and graph importance; rule-based controllers request near-intersection heads. This is a heuristic allocation, not an optimized value-of-information calculation.

## Neural training target

Independent synthetic scenes contain 4–12 vehicles, with sampled lane, route, progress, speed, observation age, measurement error, and maneuver confidence. These are sampled states, not logged closed-loop simulator episodes. They include stopped, flowing, stale-message, and ambiguous-intent cases but cannot represent every deployment distribution.

For each unordered pair, the latent label is one when either:

1. The actual routes conflict and estimated free-flight conflict-zone occupancy intervals overlap; or
2. The vehicles share an incoming lane and their headway is shorter than the simulated interaction threshold.

The label uses hidden simulator values **only during offline training/evaluation**. Online model features use track beliefs: progress, speed, ETA differences, approach equality, expected route-conflict probability, covariance, observation age, intent entropy, and progress gap. This is supervised interaction recognition; it is not causal discovery or a proof that an edge implies a causal effect.

The network has 13 standardized inputs, one 32-unit tanh hidden layer, and a sigmoid output. Binary cross-entropy with a small weight penalty is optimized with Adam. Standardization is fitted only on training scenes. Validation loss selects the best checkpoint, with ten-epoch early stopping patience. Testing uses separate scene seeds and is never used to select the checkpoint. Seeds 7, 10007, and 20007 generate the default train/validation/test datasets.

Reported test accuracy, AUC, log loss, Brier score, and calibration refer to the unpruned pair predictor on these synthetic scenes. They are not estimates of closed-loop delay-prediction accuracy or collision risk calibration.

## Changing Bayesian network

The binary node `D_i,t` means vehicle `i` is delayed at time `t`. Vehicles are ordered by belief ETA with ID as a deterministic tie breaker. Edges only run from an earlier to a later node, guaranteeing an acyclic within-slice graph, even when the order changes. Persistent state is keyed by vehicle ID, not its changing position in that ordering.

For the neural method, the MLP estimates an independent Bernoulli edge probability `w_ji`. Values below 0.15 are removed, and at most three incoming edges are retained by default. These pruning choices are fixed heuristics, not calibrated by the held-out test set.

The noisy-OR CPD is:

```text
P(D_i,t = 0 | D_i,t-1, D_parents,t, observations)
  = (1 - leak_i)
    * (1 - 0.65 * D_i,t-1)
    * product_j (1 - 0.75 * w_ji * D_j,t)

leak_i = min(0.20, 0.025 + 0.015 * observation_age_i)
```

The `w_ji` term analytically marginalizes an independent Bernoulli edge variable under this CPD. Temporal self-dependencies distinguish the model from a sequence of unrelated static graphs. The CPD coefficients are hand-set; the learned component is the edge predictor.

Ancestral particles propagate delay through the graph and yield a predictive delay marginal for planning. Soft evidence derived from the current speed belief reweights these particles; the resulting posterior is carried into the next slice. Previous-slice marginals are factorized, losing cross-time correlations. Evidence confidence decays with observation age. Effective sample size is reported because importance weights can collapse in large, highly constrained graphs. This compact filter is an approximation, and the current predictive marginal deliberately precedes local current-speed evidence.

The greedy baseline learns binary delay structure from a sliding window of belief-speed observations. It uses local maximum-likelihood BIC with a `2^k` parameter penalty, the same fixed acyclic ordering, and the same parent cap. Candidate parent sets are compared on a common set of complete observations. New vehicles initially have too little history and receive no learned parents. Only positive associations are retained because the shared noisy-OR CPD cannot express inhibitory effects. Empirical positive conditional-probability differences supply baseline influence weights; these are not Bayesian edge posteriors. This is an ordered greedy learner, not unrestricted add/delete/reverse search.

## Control and the safety boundary

At most one head vehicle per approach is considered. Up to `4! = 24` priority orders are scored. Predicted route occupancy, Gaussian uncertainty, and DBN delay probability determine reservation duration. Waiting age and lane pressure increase the cost of deferring a vehicle. The objective also includes stop cost and a short-horizon throughput preference. The chosen schedule supplies crossing grants and speed targets; belief-based adaptive cruise control limits following speeds.

The configured risk threshold screens expected route-conflict probability from categorical maneuver beliefs. It is **not** the probability of a collision over a future trajectory, and it does not certify a chance constraint.

The shared shield owns true-state reservations at the stop line, blocks conflicting routes and occupied downstream merges, and projects following distances onto a minimum gap. It can clamp position and velocity abruptly. Thus it can override the nominal acceleration bounds, and is stronger than a physically validated emergency controller. It has privileged information that the learned policy does not have. Reporting zero collisions without this qualification would overstate the result. Stop-line intervention events and following-distance projection events are separate metrics.

## Evaluation and metric definitions

Demand is generated once from a seed. Behavior noise, sensors, and messages use independent streams keyed by seed, tick, and vehicle ID. Different policies have the same demand and exogenous random draws for matching vehicle/tick observations; actions change states and which messages are requested. The horizon is rounded up to a whole simulation step and the actual simulated duration is reported.

- **Throughput:** completed exits divided by actual episode duration, expressed in vehicles/hour. No warm-up or drain period is used.
- **Completed delay:** actual elapsed travel time, including upstream waiting, minus constant-cruise ideal travel time. Only completed trips contribute; it is null if none complete.
- **Accrued delay per arrival:** for completed trips, completed delay; for unfinished trips, elapsed time since requested arrival minus progress divided by desired cruise speed, floored at zero. Pending arrivals have zero progress. Sum over all arrivals and divide by arrival count. This includes censored demand, but is still horizon-dependent, not final average trip delay.
- **Stops:** transition below 0.3 m/s after a moving interval at or above 2 m/s. Hysteresis excludes repeated tiny movements near zero. Reported per arrival as well as total.
- **Collision pairs:** distinct vehicle pairs whose swept separation falls below the sum of their radii. A continuing encounter counts once.
- **Safety interventions:** each stop-line projection and each headway projection event is counted; repeated interventions on one car can occur.
- **Communication:** attempted, dropped, delivered, and in-flight packets. Delivered includes packets later discarded because a vehicle already exited. Accounting obeys `attempted = dropped + delivered + in_flight`.
- **Timing:** graph structure update, total graph/belief-model stage, and planning are measured separately. Kinematic filtering and simulator dynamics are included only in episode wall time. The independent and rule-based policies have no structure-learning cost.

Means and standard deviations are taken across episode seeds, not correlated time steps. Intervals are normal-approximation 95% intervals; with three seeds they are exploratory. Paired differences use the same scenario and demand seed. All demand, noise, budget, horizon and other configuration values are saved with the comparison.

The scaling benchmark uses synthetic snapshots with correlated binary histories and warms each method before measuring. Neural training, belief inference, planning, and environment simulation are excluded. Both methods score pairwise candidates, and the labels/objectives differ, so the timing ratio is a comparison of these implementations, not a general algorithmic guarantee. Re-run it in an otherwise idle process for stronger timing evidence.

## Useful next experiments

1. Learn edge labels from closed-loop counterfactual simulator interventions; compare against the current occupancy-overlap proxy.
2. Fit delay CPDs and evaluate next-step delay probability/calibration independently of edge classification.
3. Add more seeds and longer runs with warm-up/drain periods; compare actuated traffic signals and a stronger reservation scheduler.
4. Replace privileged projections with a physically realizable safety controller, logging braking-limit violations and near misses.
5. Use sparse spatial candidate neighborhoods, graph neural embeddings, or a learned ordering for larger swarms.
6. Add decentralized sensing/communication, changing intent, richer vehicle dynamics, and an external simulator such as SUMO when the core hypotheses are established.
