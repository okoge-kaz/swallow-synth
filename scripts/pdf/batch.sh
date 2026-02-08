#!/bin/bash

# finepdf (Japanese)
for INDEX in {0..29}
do
  qsub -P gch51639 -q R9920251300 -v RTYPE=rt_HG,INDEX=$INDEX scripts/pdf/pdf_ja.sh
done
