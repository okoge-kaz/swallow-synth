#!/bin/bash

for INDEX in {2..162}; do
  echo "Submitting job for INDEX=$INDEX"
  qsub -P gag51395 -v INDEX=$INDEX scripts/abci/pipeline/python/stage4_score.sh
  sleep 1
done

echo "All jobs submitted successfully!"
