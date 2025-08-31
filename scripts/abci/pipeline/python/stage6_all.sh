#!/bin/bash

for INDEX in {2..160}; do
  echo "Submitting job for INDEX=$INDEX"
  qsub -P gag51395 -v INDEX=$INDEX scripts/abci/pipeline/python/stage6_format.sh
  sleep 1
done

echo "All jobs submitted successfully!"
