#!/bin/bash
dims=( 2 3 4 ) #5 6 7 8 ) #TODO CHANGE
randN=300
devN=25
method=qrpfastnull #TODO CHANGE
randfolder="conditioning_ratios/$method/rand/newton/dim"
devfolder="conditioning_ratios/$method/dev/newton/dim"
newton=newton
for dim in "${dims[@]}"
do
  mkdir -p $randfolder"$dim"
  mkdir -p $devfolder"$dim"
  nohup python3 -u conditioning_ratios.py rand $method $newton $randN $dim > $randfolder"$dim"/out.txt 2> $randfolder"$dim"/err.txt &
  echo rand $dim $! >> rand_dev_cond_ratios_pids.txt
  nohup python3 -u conditioning_ratios.py dev  $method $newton $devN  $dim > $devfolder"$dim"/out.txt  2> $devfolder"$dim"/err.txt &
  echo deva $dim $! >> rand_dev_cond_ratios_pids.txt
done
