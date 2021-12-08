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