#!/bin/bash

mkdir tmp || exit 1

for key in 0.1 0.2 0.3 0.4 0.5 1.1 1.2 1.3 1.4 1.5 2.1 2.2 2.3 2.4 2.5; do
    echo $key > tmp/$key
    grep FINAL nconv${key}.h5.checkpoint.h5.*.log | awk '{print $2 + ($3/11) + int($4 == "MoveResult.YOU_WIN_GAME") }' | sort -gr >> tmp/$key
done

paste -d, tmp/0.1 tmp/0.2 tmp/0.3 tmp/0.4 tmp/0.5 tmp/1.1 tmp/1.2 tmp/1.3 tmp/1.4 tmp/1.5 tmp/2.1 tmp/2.2 tmp/2.3 tmp/2.4 tmp/2.5

rm -r tmp
