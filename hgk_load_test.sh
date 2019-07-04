#!/bin/bash

for i in $(seq 1000 1000 20000) ; do

  #echo -e "1\n1" | ./pipeline.sh /qualitas_jars/unique_jars/selection  qualitas_jars_out load_test__$i $i
  env dataset=load_test__${i} python test_hgk_parallel_correct.py
  # env dataset=links_${i} python -m pytest test_hgk_parallel_correct.py
done
