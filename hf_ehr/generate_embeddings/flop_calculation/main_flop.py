import argparse
import time
import warnings
from pathlib import Path

import polars as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pyl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils_flop import LRModelLightning, load_task_embeddings

warnings.filterwarnings("ignore")

TRAIN_SIZES = [1000, 10000, 100000]

####################################
# 1. Load model and tokenizer
def main(args):
    device = args.device
    # torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)

    base_path = Path(args.input_meds)
    task_name = args.task

    # Load in outcome cohorts with labels and prediction time
    if task_name in ["AMI", "Celiac", "CLL", "HTN", "Ischemic_Stroke", "MASLD", "Osteoporosis", "Pancreatic_Cancer", "SLE", "T2DM"]:
        task_path_train = base_path / f"task_labels/phenotype_sample/{task_name}" / "train.parquet"
        task_path_val = base_path / f"task_labels/phenotype_sample/{task_name}" / "tuning.parquet"
        task_path_test = base_path / f"task_labels/phenotype_sample/{task_name}" / "held_out.parquet"

        train = pl.read_parquet(task_path_train, columns=["subject_id", "prediction_time", "boolean_value"])
        tune = pl.read_parquet(task_path_val, columns=["subject_id", "prediction_time", "boolean_value"])
        test = pl.read_parquet(task_path_test)
    else:
        task_path_train = base_path / f"task_labels/patient_outcomes_sample/{task_name}" / "train.parquet"
        task_path_val = base_path / f"task_labels/patient_outcomes_sample/{task_name}" / "tuning.parquet"
        task_path_test = base_path / f"task_labels/patient_outcomes_sample/{task_name}" / "held_out.parquet"

        train = pl.read_parquet(task_path_train, columns=["subject_id", "prediction_time", "boolean_value"])
        tune = pl.read_parquet(task_path_val, columns=["subject_id", "prediction_time", "boolean_value"])
        test = pl.read_parquet(task_path_test)

    train_path = base_path / "post_transform/data/train"
    tune_path = base_path / "post_transform/data/tuning"
    test_path = base_path / "post_transform/data/held_out"

    print(len(train))
    print(len(tune))
    print(len(test))

    # Load in embeddings and labels for task (model dependent)
    start_time = time.time()

    train_embeddings, train_labels, train_gflops, _, _ = load_task_embeddings(args, train, train_path, device,"train")
    tune_embeddings, tune_labels, tune_gflops, _, _ = load_task_embeddings(args, tune, tune_path, device,"tune")
    test_embeddings, test_labels, test_gflops, _,_ = load_task_embeddings(args, test, test_path, device,"testing")
    
    print(f"Train GFLOPs: {int(sum(train_gflops))}")
    print(f"Tune GFLOPs: {int(sum(tune_gflops))}")
    print(f"Test GFLOPs: {int(sum(test_gflops))}")

    end_time = time.time()
    print(f"Inference Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for Context Clues linear probing"
    )
    parser.add_argument(
        "--input_meds",
        dest="input_meds",
        action="store",
        default="/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped",
        help="Path to input MEDS data"
    )

    parser.add_argument(
        "--task",
        dest="task",
        action="store",
        default="AMI",
        help="Name of task file"
    )

    parser.add_argument(
        "--model",
        dest="model",
        action="store",
        default="mamba-tiny-8192-clmbr",
        help="Model name used for downloading off huggingface"
    )

    parser.add_argument(
        "--model_path",
        dest="model_path",
        action="store",
        default="",
        help="Path to pretrained model"
    )


    parser.add_argument(
        "--model_type",
        dest="model_type",
        action="store",
        default="mamba-ehrshot",
        help="Model name used for saving embeddings and predictions"
    )


    parser.add_argument(
        "--device",
        dest="device",
        action="store",
        default="cpu",
        help="Model name used for saving embeddings and predictions"
    )

    parser.add_argument(
        "--seed",
        dest="seed",
        action="store",
        default=123,
    )

    main(
        parser.parse_args()
    )