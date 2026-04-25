#!/bin/bash

for i in {1..100}; do
    old_name="set_$i"
    new_name="set_$((i + 300))"
    
    if [ -d "$old_name" ]; then
        mv "$old_name" "$new_name"
        echo "Renamed $old_name to $new_name"
    else
        echo "$old_name does not exist."
    fi
done

