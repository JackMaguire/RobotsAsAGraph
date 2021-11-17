# RobotsAsAGraph

## Setup

```sh
cd data/all_data/
gunzip -k *
cd ../../

g++ prepare_data.cc -o prepare_data --std=c++17 -I extern/RobotsCore/include/ -O3 -Wall -pedantic -Wextra -Wshadow
./prepare_data
```