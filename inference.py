# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import os
import sys
import json
import torch
import argparse
import pandas as pd
from loguru import logger
from datetime import timedelta

from data.datasets import Zeek
from data.cleaning import ZeekCleaner

from models.batchdb import BatchDBSCAN
from transformers import BertForMaskedLM
from models.sequence_model_deploy import DataProcessor, AnomalyDetector


def main():
    pd.set_option('display.max_columns', None)
    config = get_cl_args()

    # For valid logger levels, https://loguru.readthedocs.io/en/stable/api/logger.html (add method)
    logger.remove()
    logger.add(sys.stderr, level=config['logger'])

    # Throw this warning asap since it impacts just about everything
    if not torch.cuda.is_available():
        logger.warning("No CUDA enabled device detected. Models will run on the CPU")

    # Zeek are our dataset objects. This makes and prepares the data automatically. See data.datasets for more info
    zeek = Zeek(data_path=config['data_path'])
    zeek.remove_empty_connections()
    zeek.remove_duplicate_connections()

    # Protocol preprocessing and encoding. Sequence building as well
    log_seq, learned_vocab = create_log_vocab(config, zeek)
    log_seq = merge_save_sequence(zeek, log_seq)
    logger.success("Sequence successfully built")
    logger.critical(f"\n{log_seq.head()}")

    # Sequence modeling
    seq_processor = DataProcessor(data=log_seq, vocab=learned_vocab, seq_len=10)
    infer_seq_model(seq_processor, config, zeek=zeek)
    logger.success("Model successfully performed inference.")


def get_cl_args():
    """
    Read arguments from the command line.

    Returns:
    --------
    - dict: All of the arguments in name: value format
    """
    parser = argparse.ArgumentParser(description="CyberSentry Anomaly Detection Training Arguments")
    parser.add_argument('--seed', type=int, default=42, required=False, help=('Random seed for any random processes.'))
    parser.add_argument('--logger', type=str, default='ERROR', required=False, help=('Level for the Loguru logger. '
                                                                                     'Must be one of the predefined '
                                                                                     'levels specified by loguru.'))
    parser.add_argument('--data_path', type=str, required=True, help=('Path to the data folder containing all of the '
                                                                      'protocol files.'))
    parser.add_argument('--model_path', type=str, required=True, help=('Path to the base models folder containing all '
                                                                       'of the models. This is where everything that '
                                                                       'is learned is stored.'))
    parser.add_argument('--output_path', type=str, required=True, help=('The path where the model outputs should be'
                                                                        'stored as CSVs.'))

    args = parser.parse_args()
    config = vars(args)

    if not os.path.exists(config['data_path']):
        logger.error("Data path does not exist. Are you certain it was entered correctly?")
        raise ValueError("Data path does not exist.")

    if not os.path.exists(config['model_path']):
        logger.error("Model save path does not exist. Are you certain it was created or entered correctly?")
        raise ValueError("Model save path does not exist.")

    if not os.path.exists(config['output_path']):
        logger.error("Output path does not exist. Are you certain it was created or entered correctly?")
        raise ValueError("Output path does not exist.")

    # Manually create the path configs that include the necessary extension and full path
    config['cleaner_path'] = os.path.join(config['model_path'], 'cleaner.pkl')
    config['vocab_path'] = os.path.join(config['model_path'], 'vocab.json')
    config['transformer_path'] = os.path.join(config['model_path'], 'transformer')
    config['threshold_path'] = os.path.join(config['model_path'], 'threshold.txt')
    config['cluster_prefix'] = 'cluster'

    for key in config.keys():
        if '_path' in key:
            if not os.path.exists(config[key]):
                logger.critical(f"{config[key]} does not exist and is crucial for inference. This means that either:"
                                "\n\t1) The specified model path exists but is incorrect. Is this the wrong folder?"
                                "\n\t2) The file was deleted. Unfortunately, you must in this case retrain the model."
                                "\nPlease understand this error might REQUIRE ENTIRE RETRAINING. Verify after "
                                "retraining that cleaner.pkl, vocab.json, transformer/, & threshold.txt exist!")
                raise RuntimeError(f"{config[key]} does not exist.")

    return config


def merge_save_sequence(zeek, seq):
    """
    Merges the sequence log with all variables required by the transformer pipeline.

    Params:
    -------
    - zeek (Zeek): The dataset
    - seq (pd.DataFrame): Sequence dataframe containing the entire sequence over the lifetime of each connection

    Returns:
    --------
    - pd.DataFrame: An updated sequence dataframe that is merged with additional necessary columns
    """
    seq = seq.merge(zeek.get("conn")[["uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p"]],
                    how="left", on="uid", suffixes=(False, False))
    return seq


def create_log_vocab(config: dict, zeek: Zeek):
    """
    Trains all of the protocol log models and learns a protocol vocabulary.

    Params:
    -------
    - config (dict): Config dictionary created in main
    - zeek (Zeek): The dataset

    Returns:
    --------
    - pd.DataFrame: Sequence dataframe containing the entire sequence over the lifetime of each connection
    - dict: The vocabulary dictionary mapping tokens to more readable human names
    """
    # Load the existing ZeekCleaner, preprocess the data
    cleaner = ZeekCleaner.load(config['cleaner_path'])
    cleaned = cleaner.transform(zeek)

    # These are the logs that have a custom transform
    logger.debug(cleaner.known_transforms)

    logger.debug("Data Cleaned. Beginning Vocab Building.....")

    # This will be the final data
    new_data = {}

    # In case the model directory is off
    modeldir = os.path.normpath(config['model_path'])
    modeldir = os.path.abspath(modeldir)

    # Things shouldn't have NAs by now. If they do, remove them.
    # Get assignments and drop NA vals (that shouldn't have made it this far)
    uids = []
    for key, df in cleaned:
        # Only applies when we need to use the model, otherwise NAs are fine
        model_path = f"{modeldir}/{config['cluster_prefix']}_{key}.pkl"
        if (key not in cleaner.known_transforms) or (not os.path.exists(model_path)):
            continue

        rows_with_null = df[df.isna().any(axis=1)]
        null_uids = rows_with_null['uid'].tolist()
        uids.extend(null_uids)

        # Logging the warning if any NA values were dropped
        if len(null_uids) > 0:
            logger.warning(f"There were {len(null_uids)} NA values in the {key} log. The models cannot "
                           "accept NA values, so they were dropped. Consider checking how they are handled in "
                           "log cleaners.")

    # Logging the critical information with the number of uids collected
    if len(uids) > 0:
        for key, _ in cleaned:
            cleaned.remove(col='uid', values=uids, log=key)

    for key, df in cleaned:
        logger.debug(f"Predicting {key} of shape {df.shape}")
        model_path = f"{modeldir}/{config['cluster_prefix']}_{key}.pkl"

        if (key not in cleaner.known_transforms) or (not os.path.exists(model_path)):
            # We didn't train a model in this case so just assign it 0
            df['cluster'] = 0
        else:
            # This is done here since cluster prefix is always changing
            model = BatchDBSCAN.load(path=model_path)

            # Sometimes we run into issues with the index. Since the value of UID is what is used to join rows,
            # it doesn't actually matter if the index is rearranged. Note: this is a bandaid solution and might
            # cause problems someday!
            results = model.predict(df)['cluster'].astype(int)
            results = results.reset_index(drop=True)
            df.reset_index(drop=True, inplace=True)
            df['cluster'] = results

        new_data[key] = df[['uid', 'ts', 'cluster']]

        logger.debug(f"Finished prediction for {key}")
        logger.trace(f"\n{new_data[key]['cluster'].value_counts()}")

    # Read the vocabulary mapping that was learned during training
    with open(config['vocab_path'], 'r') as file:
        learned_vocab = json.load(file)

    # Return the new object and the new vocabulary
    clustered_data = Zeek(data=new_data)
    clustered_data.sequence(vocab=learned_vocab, known_transforms=cleaner.known_transforms)
    return clustered_data.get('sequence'), learned_vocab


def infer_seq_model(processor, config, zeek: Zeek):
    # Initialize DataProcessor and load data
    processor.load_vocab()
    processor.load_data()
    logger.debug('Sequence data preprocessed successfully.')

    # Load the trained model configuration
    model = BertForMaskedLM.from_pretrained(config['transformer_path'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # We read the threshold from the saved threshold file
    with open(config['threshold_path'], 'r') as file:
        threshold = float(file.read().strip())

    # Make the anomaly detector
    detector = AnomalyDetector(model, processor.test_pad_data, processor.seq_df, processor.test_pad_mask, threshold,
                               processor.tok_to_word, device)
    detector.generate_output()
    anom_uids = list(detector.score_dict.keys())

    # Import original logs to create .csv for each log with filtered results greater than anomaly score threshold
    dtypes = {'ts': 'datetime64[ns]', 'uid': 'object'}
    combined = pd.DataFrame(columns=['ts', 'uid']).astype(dtypes)  # empty DataFrame to append each log to

    # Iterate through log protocol types and retain timestamp and UID for each log
    for key, df in zeek:
        # Only keep the anomalous uids going forward
        df = df[df['uid'].isin(anom_uids)]
        df = df.sort_values(by=['ts'], ascending=True)
        df['ts'] = pd.to_datetime(df['ts'])
        zeek.set(key, df)

        # Append all anomalous log uids and ts to DataFrame
        temp = df[['ts', 'uid']]
        temp = temp.dropna(axis=1, how='all')
        combined = pd.concat([combined, temp], axis=0, ignore_index=True)  # Combine log UIDs with master DataFrame

    # Recreate the sequences but with timestamps instead of tokens
    # Every event timestamp should be unique within a uid with the exception of the conn log and some protocols
    combined = combined.sort_values(by=['ts'], ascending=True)
    combined = combined.groupby('uid')['ts'].agg(list).reset_index()  # Groupby UID and convert ts to list on uid

    # We know the conn log always is ordered first, even in ties, so we subtract 1 from the ts for the position
    # of the conn log. We will revert this back later, but we want to make sure this ts is unique for merging reasons
    def subtract_timedelta_from_first(ts_list):
        ts_list = ts_list[:min(len(ts_list), processor.seq_len)]
        ts_list[0] = ts_list[0] - timedelta(seconds=1)
        return ts_list

    combined['ts'] = combined['ts'].apply(subtract_timedelta_from_first)
    # Use the score_dict created by the Anomaly_Detector class to create align token scores with
    # timestamps belonging to the same uid
    combined['score'] = combined['uid'].map(lambda x: detector.score_dict[x] if x in detector.score_dict else [])
    combined = combined.explode(['ts', 'score'])

    # Iterate through log types
    for key, df in zeek:
        if df.empty:
            continue

        # Prep DataFrame to have the ts sequences
        df['ts'] = df.groupby('uid')['ts'].transform(lambda x: list(x))
        df['ts'] = df['ts'].apply(lambda x: x if isinstance(x, list) else [x])

        # Because the conn log shares ts with other logs at times,
        # Subtract the ts to match previous work and force it to first postion
        if key == "conn":
            df['ts'] = df['ts'].apply(subtract_timedelta_from_first)

        # Merge score column with exploded DataFrame
        df = df.explode('ts')  # Explode the timestamp list into separate rows
        df = df.reset_index(drop=True)
        new_df = pd.merge(df, combined, how='inner', on=['ts', 'uid'])  # Merge score

        # Recover the original ts by adding a second to match input data
        if key == "conn":
            new_df['ts'] = new_df['ts'] + timedelta(seconds=1)

        # Write to DataFrame
        new_csv = f"{key}.csv"
        out_path = os.path.join(config['output_path'], new_csv)
        new_df.to_csv(out_path, index=False)

    # TODO: make this customizable
    new_csv = "summarized_output.csv"
    out_path = os.path.join(config['output_path'], new_csv)
    detector.output.to_csv(out_path, index=False)
    logger.success(f"Model outputs saved to {config['output_path']}")


if __name__ == '__main__':
    main()
