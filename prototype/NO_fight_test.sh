#!/bin/bash
echo "Start" ; date
for file in ./NO_fight/*
do
    python3 ./main.py --source $file
done
echo "Done" ; date
