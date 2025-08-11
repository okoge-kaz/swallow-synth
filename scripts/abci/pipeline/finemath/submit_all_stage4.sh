#!/bin/bash

for ((INDEX=3; INDEX<=127; INDEX++))
do
  qsub -P gch51639 -q R9920251300 -v RTYPE=rt_HG,INDEX=$INDEX scripts/abci/pipeline/finemath/stage4_qa.sh
done
