#!/bin/bash
# BCO(0) inverse dynamics experiments on MountainCar-v0 (reproducible version).
# Every configuration runs with the same 5 seeds; each run appends one row to $RESULTS.
# The inverse dynamics model is always trained on random interaction data only;
# the demonstrations are only used to measure its accuracy.

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

# ---------------------------------------------------------------------------
# 0. collect demos once (window opens) if demos.pkl does not exist yet.
#    This run goes to its own file so it does not become an extra row in $RESULTS.
# ---------------------------------------------------------------------------
if [ ! -f demos.pkl ]; then
    $PY --num_demos 2 --tag demo_collection --results_file demo_collection.csv
fi

# ---------------------------------------------------------------------------
# 1. baseline (defaults: 5 random episodes, 64 x 2 network, 1000 iters, lr 0.01)
# ---------------------------------------------------------------------------
run baseline

# ---------------------------------------------------------------------------
# 2. network width (2 hidden layers, 1000 iters)
# ---------------------------------------------------------------------------
for h in 2 4 8 16 32 128 256; do
    run width --hidden_dim $h
done

# ---------------------------------------------------------------------------
# 3. network depth (64 units)
# ---------------------------------------------------------------------------
for l in 1 3 4; do
    run depth --num_layers $l
done

# ---------------------------------------------------------------------------
# 4. training iterations
# ---------------------------------------------------------------------------
for i in 10 25 50 100 250 500 3000; do
    run iters --num_inv_dyn_iters $i
done

# small network + few iterations: where training accuracy drops below 100%
run small_short --hidden_dim 4 --num_inv_dyn_iters 50
run small_short --hidden_dim 4 --num_inv_dyn_iters 250

# ---------------------------------------------------------------------------
# 5. learning rate
# ---------------------------------------------------------------------------
for lr in 0.001 0.1; do
    run lr --inv_dyn_lr $lr
done

# ---------------------------------------------------------------------------
# 6. input normalization off
# ---------------------------------------------------------------------------
run normalization --no_normalize

# ---------------------------------------------------------------------------
# 7. amount of random data (episodes x 200 steps), unchanged starter sampling
# ---------------------------------------------------------------------------
for e in 20 50 200; do
    run data --num_random_episodes $e
done

# ---------------------------------------------------------------------------
# 8. coverage of random data: hold each random action for k steps
#    (still uniformly random actions, no demo information)
# ---------------------------------------------------------------------------
for k in 4 8 16; do
    run action_repeat --action_repeat $k
done
run action_repeat_more_data --action_repeat 8 --num_random_episodes 50

# ---------------------------------------------------------------------------
# 9. BC iterations on (near) perfect labels, so only BC changes
# ---------------------------------------------------------------------------
for b in 100 500 1000; do
    run bc_iters_clean_labels --action_repeat 4 --num_bc_iters $b
done


echo "done: results in $RESULTS"