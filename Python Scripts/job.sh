#!/bin/sh

for random_seed in 1 2 3 4 5
do
    for dataset in PEMS-BAY METR-LA INRIX-SEA
    do
        for missing_rate in 0.1 0.2 0.4
        do
        python Exp_GMN.py -d $dataset -m $missing_rate -o Adam -l 0.001 -r $random_seed -s 0 -t random
        done
    done
done

for random_seed in 1 2 3 4 5
do
    for dataset in PEMS-BAY METR-LA INRIX-SEA
    do
        for missing_rate in 0.1 0.2 0.4
        do
            python Exp_baseline.py -d $dataset -m $missing_rate -o Adam -l 0.001 -r $random_seed -s 0 -t random
        done
    done
done
