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
from fastapi import FastAPI
import logging
from starter import train_model


def go(args):
    logging.basicConfig(level=logging.INFO)
    if args.choice == "train_model":
        logging.info("Train/Test model procedure started")
        train_model.train_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ML training pipeline")
    parser.add_argument(
        "--choice",
        type=str,
        choices=["train_model"],
        default="all",
        help="Pipeline action"
    )

    main_args = parser.parse_args()

    go(main_args)


