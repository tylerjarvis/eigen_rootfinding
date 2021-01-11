#!/bin/bash
precision=100 #decimal digits of precision
minseed=0
maxseed=100 #CHANGE number of samples
multiplicities=( 1 2 3 4 ) #multiplicities of primary root to test
dims=4 #without spaces, e.g. 234-- it can't do more than 10
# all dimensions must be larger than highest multiplicity

alphas=('1e-6'
        '1e-5'
        '1e-4'
        '1e-3'
        '1e-2'
        '1e-1')
# ('1e-6 1e-5'
        # '1e-4 1e-3'
        # '1e-2 1e-1')
# the commented out alpha values below are 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 plus np.logspace(-6,-1)
# ( '1.00000000e-06 1.26485522e-06 1.59985872e-06'
#          '2.02358965e-06 2.55954792e-06 3.23745754e-06'
#          '4.09491506e-06 5.17947468e-06 6.55128557e-06'
#          '8.28642773e-06 1.04811313e-05 1.32571137e-05'
#          '1.67683294e-05 2.12095089e-05 2.68269580e-05'
#          '3.39322177e-05 4.29193426e-05 5.42867544e-05'
#          '6.86648845e-05 8.68511374e-05 1.09854114e-04'
#          '1.38949549e-04 1.75751062e-04 2.22299648e-04'
#          '2.81176870e-04 3.55648031e-04 4.49843267e-04'
#          '5.68986603e-04 7.19685673e-04 9.10298178e-04'
#          '1.15139540e-03 1.45634848e-03 1.84206997e-03'
#          '2.32995181e-03 2.94705170e-03 3.72759372e-03'
#          '4.71486636e-03 5.96362332e-03 7.54312006e-03'
#          '9.54095476e-03 1.20679264e-02 1.52641797e-02'
#          '1.93069773e-02 2.44205309e-02 3.08884360e-02'
#          '3.90693994e-02 4.94171336e-02 6.25055193e-02'
#          '7.90604321e-02 1.00000000e-01 1e-5'
#          '1e-4 1e-3 1e-2' )

for alphavals in "${alphas[@]}"
do
  for multiplicity in "${multiplicities[@]}"
  do
    mkdir -p conditioning_ratios/nearby_roots/multiplicity"$multiplicity"
    nohup python3 -u conditioning_ratios.py $dims $multiplicity $precision $minseed $maxseed $alphavals> conditioning_ratios/nearby_roots/multiplicity"$multiplicity"/"$alphavals".txt 2> conditioning_ratios/nearby_roots/multiplicity"$multiplicity"/err"$alphavals".txt &
    echo $alphavals multiplicity $multiplicity $! >> cond_ratios_pids.txt
  done
done
