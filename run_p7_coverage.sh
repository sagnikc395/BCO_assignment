#!/bin/bash
# Part 7, second sweep: how much the *coverage* of the random interaction data
# matters, holding the inverse dynamics model fixed at the baseline
# (64 units, 2 hidden layers, 1000 iterations, lr 0.01).
# Two ways to cover more of the track are compared at that fixed model:
#   - more uniformly random episodes (more data, same narrow position range)
#   - holding each random action for k steps (same data budget, much wider range)
# plus an ablation of input standardization at both coverage levels.
# As in run_p7.sh the inverse dynamics model only ever trains on random
# interaction data; the demos are used to measure its accuracy.

set -euo pipefail

PY="uv run mountain_car_bco.py"
RESULTS="results_p7_coverage.csv"
SEEDS="0 1 2 3 4"
NUM_EVALS=20
COMMON="--results_file $RESULTS --num_evals $NUM_EVALS --eval_seed 1000 --no_render --quiet_bc"

# never mix rows from different versions/settings into one file
if [ -f "$RESULTS" ]; then
    echo "$RESULTS already exists; rename or delete it first" >&2
    exit 1
fi

# demos.pkl must already exist (run_p7.sh collects it) so both sweeps are
# scored on the same demonstration transitions
if [ ! -f demos.pkl ]; then
    echo "demos.pkl not found; run ./run_p7.sh first to record the demonstrations" >&2
    exit 1
fi

# run one configuration for every seed:  run <tag> [options...]
run() {
    tag=$1; shift
    for s in $SEEDS; do
        $PY "$@" --seed $s --tag "$tag" $COMMON
    done
}

# baseline coverage (5 random episodes = 1000 transitions), with and without
# input standardization
run norm_on
run norm_off --no_normalize

# more uniformly random data, same narrow position range
for e in 20 50 200; do
    run episodes --num_random_episodes $e
done

# wider position range at the same ~1000-transition budget
for k in 2 4 8 16; do
    run repeat --action_repeat $k
done

# normalization ablation at the wide-coverage setting
run repeat_nonorm --action_repeat 8 --no_normalize

echo "done: results in $RESULTS"
