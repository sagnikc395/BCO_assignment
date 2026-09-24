#!/bin/bash

# collect demos once 
uv run mountain_car_mountain_car_bco.py --num_demos 2 --tag baseline

uv run mountain_car_bco.py --hidden_dim 8 --tag width --no_render --quiet_bc
uv run mountain_car_bco.py --num_layers 3 --tag depth --no_render --quiet_bc
uv run mountain_car_bco.py --num_inv_dyn_iters 50 --tag iters --no_render --quiet_bc
uv run mountain_car_bco.py --inv_dyn_lr 0.001 --tag lr --no_render --quiet_bc
uv run mountain_car_bco.py --num_random_episodes 50 --tag data --no_render --quiet_bc
uv run mountain_car_bco.py --no_normalize --tag normalization --no_render --quiet_bc
uv run mountain_car_bco.py --seed 1 --tag baseline --no_render --quiet_bc
