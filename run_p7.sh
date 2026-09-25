#!/bin/bash
# Part 7: BCO(0) on MountainCar-v0.
# Sweeps the two things Part 7 asks about, network size and training iterations,
# and records inverse dynamics accuracy on the demonstration data.
# The inverse dynamics model is always trained on random interaction data only;
# the demonstrations are used to measure its accuracy, never to train it.
# Every configuration runs with the same 5 seeds; each run appends one row to $RESULTS.

set -euo pipefail

PY="uv run mountain_car_bco.py"
RESULTS="results_p7.csv"
SEEDS="0 1 2 3 4"
NUM_EVALS=20
COMMON="--results_file $RESULTS --num_evals $NUM_EVALS --eval_seed 1000 --no_render --quiet_bc"

# never mix rows from different versions/settings into one file
if [ -f "$RESULTS" ]; then
    echo "$RESULTS already exists; rename or delete it first" >&2
    exit 1
fi

# run one configuration for every seed:  run <tag> [options...]
run() {
    tag=$1; shift
    for s in $SEEDS; do
        $PY "$@" --seed $s --tag "$tag" $COMMON
    done
}

# collect demos once (window opens) if demos.pkl does not exist yet
if [ ! -f demos.pkl ]; then
    $PY --num_demos 5 --tag demo_collection --results_file demo_collection.csv
fi

# defaults: 64 x 2 network, 1000 inverse dynamics iters, lr 0.01, 5 random episodes
run baseline

# network size: width (2 hidden layers)
for h in 2 4 8 16 32 128 256; do
    run width --hidden_dim $h
done

# network size: depth (64 units per layer)
for l in 1 3 4; do
    run depth --num_layers $l
done

# inverse dynamics training iterations
for i in 10 25 50 100 250 500 3000; do
    run iters --num_inv_dyn_iters $i
done

echo "done: results in $RESULTS"
