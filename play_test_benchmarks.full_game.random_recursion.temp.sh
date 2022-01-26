#!/bin/bash

input_dirname="train_fewer_resources2"
output_dirname="train_fewer_resources2_benchmarks"

{
    for x in {1..500}; do
	echo model 1 dummy $x rand
    done
} | xargs -n 5 -P `grep processor /proc/cpuinfo | wc -l` bash -c 'python3 random_play_game.py --recursion_depth 0 --start_level 1 --stop_level 999 1>'$output_dirname'/full/$4.$3.log'

# nconv2.3.h5.checkpoint.h5 10 67 2  train/nconv2.3.h5.checkpoint.h5
# 4                         3  2  1  0                                          
