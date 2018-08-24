#!/bin/bash

for i in $(seq 1000 1000 20000) ; do

  # echo -e "2\n1" | ./pipeline.sh qualitas-corpus/links  example links_$i $i
  env dataset=links_${i} python -m pytest test_hgk_parallel_correct.py
done
