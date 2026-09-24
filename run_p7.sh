#!/bin/bash
# BCO(0) inverse dynamics experiments on MountainCar-v0.
# Every configuration is run with 3 seeds; each run appends one row to $RESULTS.
# The inverse dynamics model is always trained on random interaction data only;
# the demonstrations are only used to measure its accuracy.

set -e

PY="uv run mountain_car_bco.py"
RESULTS="results_p7.csv"    # 
SEEDS="2 4 8"

# run one configuration for every seed:  run <tag> [options...]
run() {
    tag=$1; shift
    for s in $SEEDS; do
        $PY "$@" --seed $s --tag "$tag" --results_file $RESULTS --no_render --quiet_bc
    done
}

# ---------------------------------------------------------------------------
# 0. collect demos once (window opens). Skipped automatically if demos.pkl exists.
#    Delete demos.pkl first if you want to record new demonstrations.
# ---------------------------------------------------------------------------
$PY --num_demos 2 --tag baseline --seed 0 --results_file $RESULTS
$PY --tag baseline --seed 1 --results_file $RESULTS --no_render --quiet_bc
$PY --tag baseline --seed 2 --results_file $RESULTS --no_render --quiet_bc

# ---------------------------------------------------------------------------
# 1. network width (2 hidden layers, 1000 iters)
# ---------------------------------------------------------------------------
for h in 2 4 8 16 32 128 256; do
    run width --hidden_dim $h
done

# ---------------------------------------------------------------------------
# 2. network depth (64 units)
# ---------------------------------------------------------------------------
for l in 1 3 4; do
    run depth --num_layers $l
done

# ---------------------------------------------------------------------------
# 3. training iterations
# ---------------------------------------------------------------------------
for i in 10 25 50 100 250 500 3000; do
    run iters --num_inv_dyn_iters $i
done

# small network + few iterations: look for where training accuracy drops below 100%
run small_short --hidden_dim 4 --num_inv_dyn_iters 50
run small_short --hidden_dim 4 --num_inv_dyn_iters 250

# ---------------------------------------------------------------------------
# 4. learning rate
# ---------------------------------------------------------------------------
for lr in 0.001 0.1; do
    run lr --inv_dyn_lr $lr
done

# ---------------------------------------------------------------------------
# 5. input normalization off (recheck the single-seed result)
# ---------------------------------------------------------------------------
run normalization --no_normalize

# ---------------------------------------------------------------------------
# 6. amount of random data (episodes x 200 steps)
# ---------------------------------------------------------------------------
for e in 20 50 200; do
    run data --num_random_episodes $e
done

# ---------------------------------------------------------------------------
# 7. coverage of random data: hold each random action for k steps
#    (still random actions, no demo information). Check random_pos_max.
# ---------------------------------------------------------------------------
for k in 4 8 16; do
    run action_repeat --action_repeat $k
done
run action_repeat_more_data --action_repeat 8 --num_random_episodes 50

# ---------------------------------------------------------------------------
# 8. BC iterations (BC loss was still ~0.4-0.6 at 100 iterations)
# ---------------------------------------------------------------------------
for b in 500 1000; do
    run bc_iters --num_bc_iters $b
done

echo "done: results in $RESULTS"