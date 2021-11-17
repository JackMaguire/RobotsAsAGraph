# RobotsAsAGraph

## Setup

```sh
cd data/all_data/
gunzip -k *
cd ../../

g++ prepare_data.cc -o prepare_data --std=c++17 -O3 -Wall -pedantic -Wshadow -Wextra
./prepare_data
```