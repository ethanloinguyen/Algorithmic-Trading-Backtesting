#!/bin/bash

# Define the rolling windows
WINDOWS=(
  "2010-01-01,2010-12-31"
  "2010-04-01,2011-03-31"
  "2010-07-01,2011-06-30"
  "2010-10-01,2011-09-30"
)

for window in "${WINDOWS[@]}"; do
  # Extract start and end dates
  start=$(echo "$window" | cut -d',' -f1)
  end=$(echo "$window" | cut -d',' -f2)
  
  echo "------------------------------------------------"
  echo "Triggering Workflow: $start -> $end"
  echo "------------------------------------------------"

  if ! gcloud workflows run quarterly_pipeline --location=us-central1 --data="{\"window_start\": \"$start\", \"window_end\": \"$end\"}" --project=capstone-487001; then 
    echo "ERROR: Workflow failed for window $start -> $end. Stopping batch."
    exit 1
  fi

  if [ "$window" != "${WINDOWS[-1]}" ]; then
    echo "Waiting 10 minutes before next window..."
    sleep 600
  fi


  echo "Waiting 10 minutes to avoid resource exhaustion..."
  
done

echo "All quarterly workflows have been submitted!"