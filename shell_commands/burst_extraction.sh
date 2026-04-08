#!/bin/bash

for i in {1..9}
do
  ScanSAR_burst_copy data/S1_data/test2/iw1_vv.slc data/S1_data/test2/iw1_vv.slc.par data/S1_data/test2/iw1_vv.slc.tops_par data/S1_data/test2/iw1_vv_burst$i.slc data/S1_data/test2/iw1_vv_burst$i.slc.par $i 1 - - > /dev/null 2>&1
  echo "burst $i done"
done
