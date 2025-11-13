#!/bin/bash

# code
for INDEX in {11..632}
do
  qsub -P gag51395 -v INDEX=$INDEX scripts/nemotron/code/code-en-reasoning-medium.sh
done

# math
for INDEX in {1..681}
do
  qsub -P gag51395 -v INDEX=$INDEX scripts/nemotron/math/math-en-reasoning-medium.sh
done

# stem
for INDEX in {1..999}
do
  qsub -P gag51395 -v INDEX=$INDEX scripts/nemotron/science/science-en-reasoning-medium.sh
done
