#!/bin/bash
dims=( 2 3 4 ) #5 6 7 8 ) #TODO CHANGE
randN=300
devN=50
randfolder="conditioning_ratios/rand/newton/dim"
devfolder="conditioning_ratios/dev/newton/dim"
newton=newton
for dim in "${dims[@]}"
do
  nohup python3 -u conditioning_ratios.py rand $newton $randN $dim > $randfolder"$dim"/out.txt 2> $randfolder"$dim"/err.txt &
  echo rand $dim $! >> rand_dev_cond_ratios_pids.txt
  nohup python3 -u conditioning_ratios.py dev  $newton $devN  $dim > $devfolder"$dim"/out.txt  2> $devfolder"$dim"/err.txt &
  echo dev $dim $! >> rand_dev_cond_ratios_pids.txt
done
