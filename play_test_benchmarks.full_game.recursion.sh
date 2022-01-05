#!/bin/bash

input_dirname="train_fewer_resources2"
output_dirname="train_fewer_resources2_benchmarks"

for rd in 7 6 5 4 3 2 1; do
    mkdir -p $output_dirname/full.$rd 2>/dev/null
done

{
    for x in {1..500}; do
	start=1 #dummy for consistancy
	for rd in 7 6 5 4 3 2 1; do
	    for key in 0.1 0.2 0.3 0.4 0.5 1.1 1.2 1.3 1.4 1.5 2.1 2.2 2.3 2.4 2.5; do
		model="$input_dirname/nconv${key}.h5.checkpoint.h5"
		echo $model $rd $((start+22)) $x $(basename $model)
	    done
	done
    done
} | xargs -n 5 -P `grep processor /proc/cpuinfo | wc -l` bash -c 'python3 model_play_game.py --model $0 --recursion_depth $1 --start_level 1 --stop_level 999 2>/dev/null 1>'$output_dirname'/full.$1/$4.$3.log'

# nconv2.3.h5.checkpoint.h5 10 67 2  train/nconv2.3.h5.checkpoint.h5
# 4                         3  2  1  0                                          
