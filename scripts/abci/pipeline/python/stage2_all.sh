#!/bin/bash

# Batch script to run stage2 for indices 3-154
# Usage: ./run_stage2_batch.sh

set -e

SCRIPT_NAME="scripts/abci/pipeline/python/stage2_auto_fix.sh"
START_INDEX=3
END_INDEX=154
JOB_IDS=()

echo "Starting batch submission for stage2 (indices ${START_INDEX}-${END_INDEX})"
echo "Using script: ${SCRIPT_NAME}"
echo "=========================================="

# Check if the stage2 script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: $SCRIPT_NAME not found in current directory"
    exit 1
fi

# Submit jobs for each index
for i in $(seq $START_INDEX $END_INDEX); do
    echo "Submitting job for INDEX=$i"
    
    # Submit job and capture job ID
    JOB_ID=$(qsub -P gag51395 -v INDEX=$i $SCRIPT_NAME)
    JOB_IDS+=($JOB_ID)
    
    echo "  -> Job ID: $JOB_ID"
    
    # Optional: Add a small delay to avoid overwhelming the queue system
    sleep 0.1
done

echo "=========================================="
echo "All jobs submitted successfully!"
echo "Total jobs: $((END_INDEX - START_INDEX + 1))"
echo ""
echo "Job IDs:"
for job_id in "${JOB_IDS[@]}"; do
    echo "  $job_id"
done

echo ""
echo "To monitor job status:"
echo "  qstat -u \$USER"
echo ""
echo "To check specific jobs:"
echo "  qstat ${JOB_IDS[0]} ${JOB_IDS[1]} ..."
echo ""
echo "To cancel all submitted jobs if needed:"
echo "  qdel ${JOB_IDS[@]}"
