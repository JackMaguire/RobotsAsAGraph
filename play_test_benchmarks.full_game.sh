#!/bin/bash

mkdir play_test_benchmarks/full 2>/dev/null

{
    for x in {1..500}; do
	start=1 #dummy for consistancy
	for key in 5.3; do
	    model="train/nconv${key}.h5.checkpoint.h5"
	    echo $model $start $((start+22)) $x $(basename $model)
	done
    done
} | xargs -n 5 -P 48 bash -c 'python3 model_play_game.py --model $0 --start_level 1 --stop_level 999 2>/dev/null 1>play_test_benchmarks/full/$4.$3.log'

# nconv2.3.h5.checkpoint.h5 10 67 45 train/nconv2.3.h5.checkpoint.h5
# 4                         3  2  1  0                                          
