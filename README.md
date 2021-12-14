# RobotsAsAGraph

## Setup

### Install and Build

```sh
git submodule update --init --recursive

pip3 install pybind11
g++ python_bindings.cc -o robots_core$(python3-config --extension-suffix) -O3 -Wall -Wextra -Iinclude -Iextern/RobotsCore/extern/pybind11/include -std=c++17 -fPIC $(python3 -m pybind11 --includes) -shared -Iextern/RobotsCore/include/

pip3 install nevergrad spektral

sudo apt-get install mpich
pip3 install mpi4py
```

### Prepare Data

```console
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

See `play_test_benchmarks.sh` and `play_test_benchmarks.full_game.sh`

## Derivative Free Refinement

We will be using `dfo/run_mpi.py`
```console
$ python3 dfo/run_mpi.py --help
usage: run_mpi.py [-h] --model MODEL [--opt OPT] [--just_print_layers] [--layers [LAYERS [LAYERS ...]]]
                  [--scale_coeff SCALE_COEFF] --output_dir OUTPUT_DIR

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Where should we save the output model?
  --opt OPT             Which optimizer should we use?
  --just_print_layers
  --layers [LAYERS [LAYERS ...]]
  --scale_coeff SCALE_COEFF
  --output_dir OUTPUT_DIR
```


List layers available to optimize. At the time of writing this, you need to pass a dummy argument to `--output_dir`
```console
$ python3 dfo/run_mpi.py --model train/nconv1.1.h5.checkpoint.h5 --just_print_layers --output_dir .
dense/kernel:0 (10, 16)
dense/bias:0 (16,)
dense_1/kernel:0 (8, 16)
dense_1/bias:0 (16,)
xe_net_conv/dense/kernel:0 (64, 32)
xe_net_conv/dense/bias:0 (32,)
xe_net_conv/dense_1/kernel:0 (32, 32)
xe_net_conv/dense_1/bias:0 (32,)
xe_net_conv/p_re_lu/alpha:0 (32,)
xe_net_conv/dense_2/kernel:0 (80, 16)
xe_net_conv/dense_2/bias:0 (16,)
xe_net_conv/dense_3/kernel:0 (32, 0)
xe_net_conv/dense_3/bias:0 (0,)
xe_net_conv/dense_4/kernel:0 (32, 1)
xe_net_conv/dense_4/bias:0 (1,)
xe_net_conv/dense_5/kernel:0 (32, 1)
xe_net_conv/dense_5/bias:0 (1,)
dense_2/kernel:0 (16, 16)
dense_2/bias:0 (16,)
dense_3/kernel:0 (16, 1)
dense_3/bias:0 (1,)
```

Create destination and run on selected layers
```sh
cd dfo ; for x in ../*.so; do ln -s $x; done ; cd ..

mkdir 1.1.dfo
mkdir 1.1.dfo/attention
nproc=`grep processor /proc/cpuinfo | wc -l`
mpirun -np $nproc python3 dfo/run_mpi.py --output_dir 1.1.dfo/attention --model train/nconv1.1.h5.checkpoint.h5 --layers xe_net_conv/dense_4/kernel:0 xe_net_conv/dense_4/bias:0 xe_net_conv/dense_5/kernel:0 xe_net_conv/dense_5/bias:0 1>1.1.dfo/attention/log 
```