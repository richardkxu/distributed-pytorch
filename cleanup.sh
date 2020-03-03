#!/bin/bash

HOSTLIST="hal01 hal02 hal03 hal04"

RANK=0
for node in $HOSTLIST; do
  ssh -p $node && pkill -9 python
done
wait
