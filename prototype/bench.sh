#!/bin/bash
echo "Start benchmark" ; date
START_TIME=$(date)
for file in ./videobench/*
do
    echo $file
    python3 ./main_bench.py --device gpu --source $file
done
END_TIME=$(date)
DIFF=$(( $END_TIME - $START_TIME ))
echo "Обработка gpu заняла $DIFF секунд"
START_TIME=$(date)
for file in ./videobench/*
do
    echo $file
    python3 ./main_bench.py --device cpu --source $file
done
END_TIME=$(date)
DIFF=$(( $END_TIME - $START_TIME ))
echo "Обработка cpu заняла $DIFF секунд"
echo "Done" ; date