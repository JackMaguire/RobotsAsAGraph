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
# TODO
```