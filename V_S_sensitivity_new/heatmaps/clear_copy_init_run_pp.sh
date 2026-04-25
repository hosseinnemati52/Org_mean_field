#!/bin/bash

declare -i N0=1
declare -i N=3

# Create directories
for i in $(seq $N0 $N); do
    mkdir -p "set_$i"
done

# Clear the folder contents
for i in $(seq $N0 $N); do
    rm -rf "set_$i"/*
done

# Copy all contents from source into the run_$i folders
for i in $(seq $N0 $N); do
    cp -r source/* "set_$i"
done

# Run the Python initialization script
#python3 init_cell_number_maker_v2.py

