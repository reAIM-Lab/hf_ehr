#!/bin/bash

# Display help information
show_help() {
  echo "Usage: src/ehr_foundation_model_benchmark/tools/linear_prob/run_linear_prob_with_few_shots.sh [options]"
  echo
  echo "This script runs the EHR foundation model benchmark across multiple cohorts."
  echo
  echo "Options:"
  echo "  --base_dir DIR       Base directory containing cohort folders"
  echo "  --output_dir DIR     Output directory for results"
  echo "  --meds_dir DIR       Medications directory"
  echo "  --model_name NAME    Model name to use"
  echo "  -h, --help           Display this help message and exit"
  echo
  echo "All options are required."
}

# ALLOWED_TASKS=( "Celiac")
# ALLOWED_TASKS=("AMI" "CLL" "HTN" "Ischemic_Stroke" "MASLD" "Osteoporosis" "Pancreatic_Cancer" "SLE" "T2DM")
# ALLOWED_TASKS=("AMI"  "Ischemic_Stroke")
# Check if no arguments provided
if [ $# -eq 0 ]; then
  show_help
  exit 1
fi

# Process command line arguments
while [ $# -gt 0 ]; do
  case $1 in
    --base_dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --meds_dir)
      MEDS_DIR="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$BASE_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$MEDS_DIR" ] || [ -z "$MODEL_NAME" ]; then
  echo "Error: Missing required arguments."
  show_help
  exit 1
fi

# Verify directories exist
if [ ! -d "$BASE_DIR" ]; then
  echo "Error: Base directory $BASE_DIR does not exist."
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Error: Output directory $OUTPUT_DIR does not exist."
  exit 1
fi

if [ ! -d "$MEDS_DIR" ]; then
  echo "Error: Medications directory $MEDS_DIR does not exist."
  exit 1
fi

# Iterate through each cohort folder
for cohort_dir in "$BASE_DIR"*/; do
  # Extract task name from folder name
  task_name=$(basename "$cohort_dir")

  # if [[ ! " ${ALLOWED_TASKS[*]} " =~ " ${task_name} " ]]; then
  #   # echo "Skipping cohort: $task_name (not in allowed list)"
  #   continue
  # fi
  


  # Skip if not a directory
  if [ ! -d "$cohort_dir" ]; then
    continue
  fi

  echo "Processing cohort: $task_name"

  # Find features_with_label directory
  features_dir=$(find "$cohort_dir" -type d -name "features_with_label" | head -n 1)

  if [ -z "$features_dir" ]; then
    echo "Warning: No features_with_label directory found for $task_name. Skipping..."
    continue
  fi

  echo "Found features directory: $features_dir"

  # Run the fine-tuning command
  # echo "Running benchmark for $task_name..."
  python -u -m finetune_with_linear_prob \
    --features_label_input_dir "$features_dir" \
    --model_name "$MODEL_NAME" \
    --task_name "$task_name" \
    --output_dir "$OUTPUT_DIR" \
    --meds_dir "$MEDS_DIR"

  # Check if finetune command succeeded
  if [ $? -ne 0 ]; then
    echo "Warning: Benchmark failed for $task_name"
    echo "---------------------------------"
    continue
  fi

  # Find all model directories created by the finetune process
  for model_dir in "$OUTPUT_DIR/$task_name/$MODEL_NAME"_*/; do
    # Skip if not a directory
    if [ ! -d "$model_dir" ]; then
      continue
    fi

    model_folder=$(basename "$model_dir")
    echo "Found model folder: $model_folder"

    # Check if results.json already exists
    if [ -f "$model_dir/results.json" ]; then
      echo "Skipping evaluation for $task_name/$model_folder - results.json already exists"
    # Check if test_predictions directory exists
    elif [ -d "$model_dir/test_predictions" ]; then
      echo "Running meds-evaluation-cli for $task_name/$model_folder"

      # Run the meds-evaluation-cli command
      meds-evaluation-cli predictions_path="$model_dir/test_predictions" output_dir="$model_dir"

      # Check if command succeeded
      if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully"
      else
        echo "Warning: Evaluation failed for $task_name/$model_folder"
      fi
    else
      echo "Warning: No test_predictions directory found in $model_dir. Skipping..."
    fi
  done

  echo "Completed $task_name"
  echo "---------------------------------"
done

echo "All tasks completed!"