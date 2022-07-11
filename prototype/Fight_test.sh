#!/bin/bash
echo "Start" ; date
for file in ./Fight/*
do
    python3 ./main.py --source $file
done
echo "Done" ; date