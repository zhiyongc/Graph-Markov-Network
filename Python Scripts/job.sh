#!/bin/sh

for random_seed in 1 2 3 4 5
do
    for dataset in PEMS-BAY LOOP-SEA
    do
        for missing_rate in 0.1 0.2 0.4
        do
            python Exp_baseline.py -d $dataset -m $missing_rate -o Adam -l 0.001 -r $random_seed -s 0 -t temporal
        done
    done
done

for random_seed in 1 2 3 4 5
do
    for dataset in PEMS-BAY METR-LA INRIX-SEA
    do
        for missing_rate in 0.1 0.2 0.4
        do
            python Exp_GMN.py -d $dataset -m $missing_rate -o Adam -l 0.001 -r $random_seed -s 0 -t spatial
        done
    done
done

for random_seed in 1 2 3 4 5
do
    for dataset in PEMS-BAY METR-LA INRIX-SEA
    do
        for missing_rate in 0.1 0.2 0.4
        do
            python Exp_baseline.py -d $dataset -m $missing_rate -o Adam -l 0.001 -r $random_seed -s 0 -t spatial
        done
    done
done

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

# python Exp_GMN.py -d METR-LA -m 0.1 -o Adam -l 0.01 -r 1 -s 0

#python Exp_GMN.py METR-LA 0.1 Adam 0.01
#python Exp_GMN.py METR-LA 0.2 Adam 0.01
#python Exp_GMN.py METR-LA 0.4 Adam 0.01

#python Exp_baseline.py METR-LA 0.1 Adam 0.001
#python Exp_baseline.py METR-LA 0.2 Adam 0.001
#python Exp_baseline.py METR-LA 0.4 Adam 0.001

# python Exp_GMN.py PEMS-BAY 0.1 Adam 0.01
#python Exp_GMN.py PEMS-BAY 0.2 Adam 0.01
#python Exp_GMN.py PEMS-BAY 0.4 Adam 0.01

#python Exp_baseline.py PEMS-BAY 0.1 Adam 0.001
#python Exp_baseline.py PEMS-BAY 0.2 Adam 0.001
#python Exp_baseline.py PEMS-BAY 0.4 Adam 0.001

#python Exp_GMN.py LOOP-SEA 0.1 Adam 0.01
#python Exp_GMN.py LOOP-SEA 0.2 Adam 0.01
#python Exp_GMN.py LOOP-SEA 0.4 Adam 0.01

#python Exp_baseline.py LOOP-SEA 0.1 Adam 0.001
#python Exp_baseline.py LOOP-SEA 0.2 Adam 0.001
#python Exp_baseline.py LOOP-SEA 0.4 Adam 0.001

#python Exp_GMN.py INRIX-SEA 0.1 Adam 0.01
#python Exp_GMN.py INRIX-SEA 0.2 Adam 0.01
#python Exp_GMN.py INRIX-SEA 0.4 Adam 0.01

#python Exp_baseline.py INRIX-SEA 0.1 Adam 0.001
#python Exp_baseline.py INRIX-SEA 0.2 Adam 0.001
#python Exp_baseline.py INRIX-SEA 0.4 Adam 0.001



# python Exp_GMN.py METR-LA 0.1
# python Exp_GMN.py METR-LA 0.2
# python Exp_GMN.py METR-LA 0.4

# python Exp_GMN.py LOOP-SEA 0.2
# python Exp_GMN.py LOOP-SEA 0.4
# python Exp_GMN.py LOOP-SEA 0.6

# python Exp_GMN.py PEMS-BAY 0.2 Adam 0.01 0.99
# python Exp_GMN.py PEMS-BAY 0.2 Adam 0.01 0.9
# python Exp_GMN.py PEMS-BAY 0.2 Adam 0.01 0.8
# python Exp_GMN.py PEMS-BAY 0.2 Adam 0.01 0.7
# python Exp_GMN.py PEMS-BAY 0.2 Adam 0.01 0.6
# python Exp_GMN.py PEMS-BAY 0.2 Adam 0.01 0.5
# python Exp_GMN.py PEMS-BAY 0.2 Adam 0.01 0.4
# python Exp_GMN.py PEMS-BAY 0.2 Adam 0.01 0.3
# python Exp_GMN.py PEMS-BAY 0.2 Adam 0.01 0.2
# python Exp_GMN.py PEMS-BAY 0.2 Adam 0.01 0.1



