# RobotsAsAGraph

## Setup

### Prepare Data

```sh
$ ls data/
all_data

$ cd data/all_data/
$ gunzip -k *
$ cd ../../

$ g++ prepare_data.cc -o prepare_data --std=c++17 -I extern/RobotsCore/include/ -O3 -Wall -pedantic -Wextra -Wshadow
$ ./prepare_data

$ ls data/
all_data  training_data.txt  validation_data.txt
```

### Prepare PyBind11

```sh
#TODO
```

## Train

```sh
mkdir train
python3 train.py --nconv 3 --model train/model_3.h5

# or...
for y in 1 2 3 4 5; do
    echo $y
    for x in 1 2 3; do
    	python3 train.py --nconv $y --model train/nconv$y.$x.h5 1>train/nconv$y.$x.log 2>train/nconv$y.$x.err &
    done
done 
wait
```

## Evaluate

```sh
# TODO
```

## Derivative Free Refinement

```sh
mkdir 1.1.dfo
mkdir 1.1.dfo/attention
mpirun -np 64 python3 dfo/run_mpi.py --output_dir 1.1.dfo/attention --model train/nconv1.1.h5.checkpoint.h5 --layers xe_net_conv/dense_4/kernel:0 xe_net_conv/dense_4/bias:0 xe_net_conv/dense_5/kernel:0 xe_net_conv/dense_5/bias:0 1>1.1.dfo/attention/log 
```