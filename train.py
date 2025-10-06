# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import os
import sys
import json
import torch
import shutil
import argparse
from loguru import logger

from data.datasets import Zeek
from data.cleaning import ZeekCleaner

from transformers import BertConfig
from models.batchdb import BatchDBSCAN
from models.sequence_model_train import DataProcessor, AnomalyDetector, BertTrainer


def main():
    config = get_cl_args()

    # For valid logger levels, https://loguru.readthedocs.io/en/stable/api/logger.html (add method)
    # log_path = os.path.join(config['model_path'], 'output.log')
    logger.remove()
    logger.add(sys.stderr, level=config['logger'])

    # Don't even go further if training is impossible
    if not torch.cuda.is_available():
        logger.error("No CUDA enabled device detected. Transformer requires CUDA.")
        raise RuntimeError("CUDA unavailable.")

    # Zeek are our dataset objects. This makes and prepares the data automatically. See data.datasets for more info
    zeek = Zeek(data_path=config['data_path'], random_state=config['seed'])
    zeek.remove_empty_connections()
    zeek.remove_duplicate_connections()
    zeek.reset_index()

    # Clear the directory beforehand. Necessary to make sure that we don't use old models and throw appropriate errors
    for filename in os.listdir(config['model_path']):
        file_path = os.path.join(config['model_path'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.debug(f'Failed to delete {file_path}. Reason: {e}')

    # Protocol preprocessing and encoding, then sequence building
    log_seq, vocab = train_log_vocab(config, zeek)
    log_seq = merge_save_sequence(zeek, log_seq, vocab, config['vocab_path'])
    logger.success("Sequence successfully built")

    # Sequence modeling
    seq_processor = DataProcessor(seq_df=log_seq, vocab=vocab, seq_len=8)
    train_seq_model(seq_processor, config)
    logger.success("Model successfully trained and prepared for inference.")


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
                                                                       'is learned is stored. MUST BE ITS OWN '
                                                                       'DIRECTORY, AS IT IS COMPLETELY CLEARED.'))
    parser.add_argument('--acceptance_rate', type=str, required=True, help=('Variable that controls how often '
                                                                            'notables are thrown by our pipeline. '
                                                                            'See the documentation for more info.'))

    args = parser.parse_args()
    config = vars(args)

    if not os.path.exists(config['data_path']):
        logger.error("Data path does not exist. Are you certain it was entered correctly?")
        raise ValueError("Data path does not exist.")

    if not os.path.exists(config['model_path']):
        logger.error("Model save path does not exist. Are you certain it was created or entered correctly?")
        raise ValueError("Model save path does not exist.")

    # Manually create the path configs that include the necessary extension and full path
    config['cleaner_path'] = os.path.join(config['model_path'], 'cleaner.pkl')
    config['vocab_path'] = os.path.join(config['model_path'], 'vocab.json')
    config['transformer_path'] = os.path.join(config['model_path'], 'transformer')
    config['threshold_path'] = os.path.join(config['model_path'], 'threshold.txt')
    config['cluster_prefix'] = 'cluster'

    return config


def merge_save_sequence(zeek, seq, vocab, vocab_path):
    """
    Merges the sequence log with all variables required by the transformer pipeline.
    Also saves the vocab dictionary for later usage during inference to ensure consistent mappings.

    Params:
    -------
    - zeek (Zeek): The dataset
    - seq (pd.DataFrame): Sequence dataframe containing the entire sequence over the lifetime of each connection
    - vocab (dict): Mappings of cluster names to token values for sequencing
    - vocab_path (str): Absolute path to where the vocab should be stored as a json

    Returns:
    --------
    - pd.DataFrame: An updated sequence dataframe that is merged with additional necessary columns
    """
    # The transformer ultimately includes UID and all address information in all of its output, so we merge it here
    seq = seq.merge(zeek.get("conn")[["uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p"]],
                    how="left", on="uid", suffixes=(False, False))

    # Writing of the vocab json so that at inference time we have consistent cluster to token mappings
    with open(vocab_path, 'w') as json_file:
        json.dump(vocab, json_file, indent=4)

    return seq


def train_log_vocab(config: dict, zeek: ZeekCleaner):
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
    # Create and fit the cleaner, preprocess the data
    cleaner = ZeekCleaner()
    cleaned = cleaner.fit_transform(zeek)
    cleaner.save(config['cleaner_path'])

    logger.debug("Data Cleaned. Beginning Training.....")

    # This will be the final data
    new_data = {}

    # Now, we loop through all logs and train the models
    for key, df in cleaned:
        logger.debug(f"Training {key} of shape {df.shape}")

        if (len(df) < 200) or (key not in cleaner.known_transforms):
            # When we don't have a transform for it, just use a singular cluster
            # Or in the case that there is too little data to practically cluster
            df['cluster'] = 0
        else:
            min_samples = max(int(0.01 * len(df)), 20)
            min_samples = min(min_samples, 1000)
            model = BatchDBSCAN(eps=0.5, min_samples=min_samples, batch_size=30000)

            # Train and get assignments, drop NA values as well
            orig_shape = len(df)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            new_shape = len(df)
            if orig_shape != new_shape:
                logger.warning(f"There were {orig_shape - new_shape} NA values in the {key} log. The models cannot "
                               "accept NA values, so they were dropped. Consider checking how they are handled in "
                               "log cleaners.")

            # Sometimes we run into issues with the index. Since the value of UID is what is used to join rows,
            # it doesn't actually matter if the index is rearranged. Note: this is a bandaid solution and might
            # cause problems someday!
            results = model.fit_predict(df)['cluster']
            results = results.reset_index(drop=True)
            df['cluster'] = results

            # This function will overwrite saved models at this location!
            path = f"{config['model_path']}/{config['cluster_prefix']}_{key}.pkl"
            model.save(path)

        new_data[key] = df[['uid', 'ts', 'cluster']]
        logger.trace(f"\n{new_data[key]['cluster'].value_counts()}")

    # Return the new object and the new vocabulary
    clustered_data = Zeek(data=new_data)
    vocab = clustered_data.sequence(known_transforms=cleaner.known_transforms)
    return clustered_data.get('sequence'), vocab


def train_seq_model(processor, config):
    """
    Pipeline for training the entire transformer-based sequence model.

    Params:
    -------
    - processor (sequence_model_train.DataProcessor): A data preprocessing object for managing the inputs and making
                                                      them PyTorch compatible.
    - save_path (str): The path where the transformer will be saved to
    - acceptance_rate (float): Ratio that controls the frequency of which notables are created. Represents the
                               proportion of training data that the transformer will deem "anomalous" to choose
                               a baseline threshold for alerting an anomaly.
    """
    # Processor is given the data and the vocab, these method calls just invoke the necessary preprocessing
    processor.load_vocab()
    processor.load_data()
    processor.pad_and_split()
    processor.data_loader(batch_size=32)

    # Define BERT configuration
    bert_config = BertConfig(
        vocab_size=processor.vocab_size,
        hidden_size=768,  # embedding dimension
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=1024,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=12,  # sequence length
        initializer_range=0.02,
        output_hidden_states=True
    )

    # Initialize BertTrainer with defined configuration and data loaders
    trainer = BertTrainer(config=bert_config,
                          train_data_loader=processor.train_data_loader,
                          val_data_loader=processor.val_data_loader,
                          epochs=3)

    # Train the model
    trainer.train()

    # Save the trained model using save_pretrained
    trainer.model.save_pretrained(config['transformer_path'])

    # Initialize AnomalyDetector and get recommended anomaly threshold
    detector = AnomalyDetector(model=trainer.model, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Assuming you have test_data and m_test from your previous setup
    anomaly_scores = detector.generate_scores(processor.test_data, processor.m_test)

    # Find threshold
    _, threshold = detector.return_anomalies(anomaly_scores, float(config['acceptance_rate']))

    logger.trace(f"Chosen Threshold: {threshold}")

    # The threshold is written to a file so it can be accessed during inference
    with open(config['threshold_path'], 'w') as file:
        file.write(str(threshold))


if __name__ == '__main__':
    main()
