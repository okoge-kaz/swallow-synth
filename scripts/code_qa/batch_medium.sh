#!/bin/bash

# code
for INDEX in {1..60}
do
  qsub -P gch51639 -q R9920251300 -v RTYPE=rt_HG,INDEX=$INDEX scripts/code_qa/swallow-code-v2.sh
done
