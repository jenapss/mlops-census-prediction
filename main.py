# Put the code for your API here.
import os
import json
import requests
import joblib 
from joblib import load
import argparse
from pydantic import BaseModel, Field
import pandas as pd
import sys

import logging
from starter import train_model, slice_score


def go(args):
    logging.basicConfig(level=logging.INFO)
    if args.choice == "train_model":
        logging.info("Train/Test model procedure started")
        train_model.train_model()

    if args.choice == "slice_score":
        logging.info("Slice Score procedure started")
        slice_score.slice_metrics()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ML training pipeline")
    parser.add_argument(
        "--choice",
        type=str,
        choices=["train_model", 'slice_score'],
        default="all",
        help="Pipeline action"
    )

    main_args = parser.parse_args()

    go(main_args)


