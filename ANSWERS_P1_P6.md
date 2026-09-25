# CS690 Assignment 1 — Answers (Parts 1–7)


## Part 1

The random/untrained policy gets stuck in the local valley at the bottom of the
hill. MountainCar gives a reward of -1 per step and nothing else until the flag
is reached, so there is no gradient of reward pointing out of the valley. With
no intermediate signal to follow, the agent never discovers the
back-and-forth energy-building behavior and stays in the valley until the
200-step timeout.

## Part 2 — Human demonstrations

| statistic | value |
|---|---|
| mean return | -153.3 |
| best return | **-87** |
| worst return | -200 (timeout) |

**Best strategy.** Of the three suggested strategies, *left, then right all the
way up the hill* worked best and is the one used for every demonstration that
follows. Going right first wastes steps because the car starts near the bottom
with no speed, so the right push is fighting gravity from a standstill. Pushing
left first climbs the shorter left hill, converts that height into speed on the
way down, and one sustained right push then carries the car over. The
multi-oscillation strategy (left, right, left, right) also works but costs an
extra swing, and since the reward is -1 per step the extra swing is pure loss.

**Best human score: -87**, i.e. reaching the flag in 87 steps. Two of the 18
episodes still timed out, so the human demonstrator is good but not perfectly
consistent.

## Part 3 — Behavior cloning on a single demonstration

Trained on one demonstration (return -151).

- BC loss: 1.068 → 0.285 over 100 iterations (monotone decrease, no plateau).
- Evaluation over 6 episodes:

| statistic | value |
|---|---|
| average return | -177.5 |
| min return | -200.0 |
| max return | -146.0 |

The loss drops cleanly, but the policy generalizes poorly: two of six
evaluation episodes time out. One trajectory covers too little of the state
space, so at test time the agent leaves the demonstrated distribution and has
no learned behavior to fall back on (compounding covariate shift).

It does reproduce the rough shape of the demonstrated left-then-right strategy
on the episodes it solves, but when a random start puts it in a state the
single demo never visited it falls back to pushing in one direction and gets
stuck oscillating in the valley until the timeout.

## Part 4 — Behavior cloning on five demonstrations

Trained on five demonstrations (returns -192, -160, -169, -162, -92).

- BC loss: 1.130 → 0.523 over 100 iterations.
- Evaluation over 6 episodes:

| statistic | value |
|---|---|
| average return | -145.8 |
| min return | -153.0 |
| max return | -141.0 |

More data gives a *higher* final training loss (harder, more diverse target)
but a much better and far more consistent policy: average return improves from
-177.5 to -145.8, and no episode times out. The spread shrinks from 54 to 12,
which is the practical point — coverage of the state distribution matters more
than fitting a single trajectory tightly.

The agent does copy the demonstrated strategy: it pushes left first, then holds
right through the swing back up, which is the strategy used for all five
demonstrations. Keeping the demonstrations consistent is what makes this
possible — the cloning objective averages over whatever it is shown, so five
demos of *different* strategies would have produced a policy that interpolates
between them rather than committing to either.

## Part 5 — Behavior cloning on two demonstrations

Trained on two demonstrations (returns -200, -116), i.e. one timeout and one
good episode.

- BC loss: 0.933 → 0.334 over 100 iterations (lowest loss of the three runs).
- Evaluation over 6 episodes:

| statistic | value |
|---|---|
| average return | -200.0 |
| min return | -200.0 |
| max return | -200.0 |

Complete failure — every evaluation episode times out, despite the *best*
training loss of all three runs. This is the clearest illustration that BC
training loss is not a proxy for policy performance. Half the training data is
a failed episode in which the car oscillates in the valley, so the cloned
policy faithfully reproduces exactly that: it imitates the stuck behavior.
With only two trajectories there is not enough good data to outvote the bad
one.

**Making BC robust to a minority of bad demonstrations.** Weight or filter the
demonstrations by return before cloning, rather than treating all state-action
pairs as equally worth imitating. Concretely: keep only the demonstrations
whose return is above the median (or above some threshold, e.g. discard any
episode that timed out), or weight each demonstration's contribution to the
cross-entropy loss by its return, so a -200 episode contributes far less than a
-90 one. This is cheap here because MountainCar returns are observed, and it
directly removes the failure mode above. A variant that does not need reward
labels at all is to fit a policy on all the data, roll it out, and iteratively
drop the demonstrations it agrees with least — outlier rejection on the
assumption that bad demos are a minority.

### Takeaway across Parts 3–5

| demos | avg return | timeouts / 6 | final loss |
|---|---|---|---|
| 1 | -177.5 | 2 | 0.285 |
| 2 (one is a failure) | -200.0 | 6 | 0.334 |
| 5 | -145.8 | 0 | 0.523 |

Demonstration *quality* and state-space coverage dominate; lower BC loss can
mean better imitation of a worse policy.

## Part 6 — Changes needed to turn the BC code into BCO(0)

BCO learns from state-only observations: the demonstrations give us
(s_t, s_{t+1}) but the expert's actions a_t are assumed unavailable. Concretely,
`mountain_car_bc.py` would need three changes.

1. **Pre-demonstration phase (new).** Before touching the demonstrations, run a
   uniformly random policy in the environment and record its
   (s_t, a_t, s_{t+1}) transitions — this is what
   `collect_random_interaction_data` provides. These actions *are* known,
   because the agent chose them itself.
2. **Inverse dynamics model (new network).** Add a second network
   M_θ: (s_t, s_{t+1}) → a_t and train it on the random transitions with the
   same cross-entropy classification loss the policy already uses. Its input is
   4-dimensional (the two states, or equivalently s_t concatenated with
   s_{t+1} - s_t) instead of 2-dimensional, and its output is again a 3-way
   distribution over actions. Critically, it is trained *only* on the random
   interaction data — the demonstrations are never used to fit it.
3. **Label, then clone (changed).** Run M_θ on each action-free demonstration
   transition to get a pseudo-label â_t = argmax M_θ(s_t, s_{t+1}), and pass
   those pseudo-labels to the existing `train_policy` in place of the
   ground-truth `acs` tensor. Everything downstream — the policy network, the
   BC loss, the evaluation loop — is unchanged.

The demonstrator's true actions are kept only to *measure* how good the inferred
labels are; they never enter either training objective.

**Why this is BCO(0) and not BCO(α).** The α in BCO(α) is the size of the
post-demonstration phase: after cloning, BCO(α) rolls out the learned policy,
appends its transitions to the inverse-dynamics dataset, re-fits M_θ, and
re-clones, repeating so that M_θ becomes accurate on the states the policy
actually visits. BCO(0) sets α = 0, i.e. it does none of this — a single pass of
random exploration → inverse dynamics → label → clone. That is the whole
difficulty: M_θ is trained on the random policy's state distribution but queried
on the expert's, and Part 7 shows this mismatch is the dominant source of error.

## Part 7 — BCO ablations

Setup: BCO on MountainCar, 734 human demonstration transitions, 20 evaluation episodes
per run, 5 seeds per configuration. `demo_acc` is the inverse-dynamics model's
action-prediction accuracy on the expert states; `success` is the fraction of
evaluation episodes that reach the flag. Raw per-seed rows are in
`results_p7.csv` and `results_p7_coverage.csv`.

Baseline (hidden 64, 2 layers, 1000 inverse-dynamics iters, 5 random episodes,
no action repeat, normalized states): **success 0.36, return -185.2,
demo_acc 0.771**.

### Inverse-dynamics network capacity

| hidden dim | success | return | demo_acc |
|---|---|---|---|
| 2 | 0.31 | -185.8 | 0.529 |
| 4 | **0.86** | **-163.3** | 0.849 |
| 8 | 0.70 | -165.7 | 0.830 |
| 16 | 0.51 | -176.6 | 0.806 |
| 32 | 0.19 | -187.2 | 0.784 |
| 128 | 0.49 | -175.5 | 0.790 |
| 256 | 0.55 | -166.8 | 0.805 |

| layers | success | return | demo_acc |
|---|---|---|---|
| 1 | **0.68** | -166.8 | 0.837 |
| 2 (baseline) | 0.36 | -185.2 | 0.771 |
| 3 | 0.02 | -199.5 | 0.712 |
| 4 | 0.00 | -200.0 | 0.649 |

Smaller is better here. A 4-unit single-hidden-layer model beats the baseline
by a wide margin, and depth actively hurts — at 3–4 layers the policy fails
completely. The inverse dynamics of MountainCar are nearly linear in
(Δposition, Δvelocity), so extra capacity only lets the model overfit the
narrow random-exploration distribution and extrapolate badly onto expert
states. `demo_acc` tracks final success closely, which makes it a usable
diagnostic without running evaluations.

### Inverse-dynamics training iterations

| iters | success | return | demo_acc |
|---|---|---|---|
| 10 | 0.00 | -200.0 | 0.612 |
| 25 | 0.32 | -187.0 | 0.714 |
| 50 | **0.54** | -179.3 | 0.752 |
| 100 | 0.37 | -182.3 | 0.760 |
| 250 | 0.39 | -182.3 | 0.768 |
| 500 | 0.35 | -186.3 | 0.771 |
| 1000 (baseline) | 0.36 | -185.2 | 0.771 |
| 3000 | 0.27 | -187.2 | 0.785 |

Beyond ~50 iterations more training raises `demo_acc` slightly while success
flattens or drops — the model keeps sharpening its fit to the random-policy
transitions, which does not transfer. Ten iterations is simply undertrained.

### Exploration coverage (the dominant factor)

| configuration | success | return | demo_acc |
|---|---|---|---|
| baseline (5 random episodes) | 0.36 | -185.2 | 0.771 |
| unnormalized states | 0.71 | -174.0 | 0.818 |
| 20 random episodes | **0.97** | -149.5 | 0.837 |
| 50 random episodes | 0.74 | -167.7 | 0.834 |
| 200 random episodes | 0.89 | -156.5 | 0.849 |
| action repeat 2 | 0.74 | -153.9 | 0.868 |
| action repeat 4 | 0.88 | -154.2 | 0.885 |
| action repeat 8 | **1.00** | **-140.5** | 0.924 |
| action repeat 16 | 0.89 | -145.8 | 0.940 |
| action repeat 8, unnormalized | 0.90 | -145.7 | 0.824 |

Coverage of the pre-demonstration phase matters far more than any architecture
or optimization choice. Action repeat is the single biggest win: repeating each
random action for 8 steps turns the random walk into a committed
back-and-forth that actually swings the car up the hill, so the
inverse-dynamics data covers the high-velocity states the expert visits.
That alone takes success from 0.36 to 1.00 and return from -185 to -140.5,
matching the expert demonstrations. Simply collecting more random episodes
helps for the same reason but less efficiently — 20 episodes of committed
exploration is worth more than 200 of dithering.

### Conclusion

BCO's bottleneck is not the cloning step or the inverse-dynamics model's
capacity; it is whether the pre-demonstration data covers the state-action
distribution the expert occupies. The two interventions that fixed the run
(temporally correlated exploration, smaller inverse-dynamics model) both reduce
the mismatch between where the model is trained and where it is queried.
