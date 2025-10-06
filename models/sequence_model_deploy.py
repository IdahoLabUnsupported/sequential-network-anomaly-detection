# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import pandas as pd
import numpy as np
import json
import random
import math
import itertools
import transformers
from transformers import BertConfig, BertForMaskedLM
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from data.datasets import Zeek
from loguru import logger


class DataProcessor:
    """
    Designed to pull in and combine protocol log vocabulary into a sequence.
    Prepares the sequences for deployment on the masked language model.

    Methods:
    --------
    - load_data: Loads and combines tokenized data
    - load_vocab: Loads log vocabulary and creates dictionary mapping log tokens to log types
    """

    def __init__(self, data: pd.DataFrame, vocab: dict, seq_len: int):
        self.seq_df = data
        self.word_to_tok = vocab
        self.seq_len = seq_len
        self.vocab_size = 0
        self.tok_to_word = {}
        self.test_sequence = []
        self.test_mask = []
        self.test_pad_data = None
        self.test_pad_mask = None

    def load_data(self):
        """
        Designed to combine log tokens into one unpadded array.
        Trims, masks, and pads data using helper functions.
        """

        # Extract log sequences from dataframe
        tok_sequence = self.seq_df['sequence'].copy()
        self.test_sequence = list(tok_sequence.apply(self._prepare_sentence))

        # Convert to tensors and create mask for BERT input
        self.test_sequence, self.test_mask = self._mask_sequences(self.test_sequence)

        # Pad deployment data and mask
        self.test_pad_data = pad_sequence(self.test_sequence, batch_first=True)  # Pad data with 0's
        self.test_pad_mask = pad_sequence(self.test_mask, batch_first=True)

        logger.debug("BERT preprocessing complete")

    def _prepare_sentence(self, seq: list) -> list:
        """
        Limits the sentence to user specified amount of words and adds the SOS and EOS where applicable.
        Want to force the model to predict the EOS token, so we do not add the EOS token to arrays longer than seq_len.

        Params:
        -------
        - seq (list): list of tokens

        Returns:
        --------
        - seq (list): trimmed and reserved tokenized list of tokens.
        """
        total_len = self.seq_len + 2  # Adds room for reserved BOS and EOS tokens
        seq.insert(0, 2)  # Insert SOS token at the start
        if len(seq) < total_len:
            seq.append(3)  # Append the EOS token
        else:
            seq = seq[:total_len]  # Trim it if we don't use an EOS token
        return seq

    def _mask_sequences(self, sequences):
        """
        Adding reserved tokens for start and end of sequence, masking values in accordance with
        BERT guidelines, generating labels and attention mask to guide training.

        Params:
        --------
        - sequences (list of lists): array of tokenized sequences

        Returns:
        --------
        - masked_sequence, attn_mask (tuple): trimmed sequence of tokens
        """
        seq = [torch.tensor(seq) for seq in sequences]  # Convert to tensors
        mask = [torch.ones(len(seq)) for seq in sequences]  # Create mask of ones to reflect unpadded tokens
        return seq, mask

    def load_vocab(self):
        """
        Loads vocabulary and adds special tokens to dictionary.
        """
        # Add special tokens to the vocabulary dictionary
        special_tokens = {
            '[IGN]': -100,
            'pad': 0,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        self.word_to_tok.update(special_tokens)

        # Create reverse dictionary for interperability
        self.tok_to_word = {v: k for k, v in self.word_to_tok.items()}

        self.vocab_size = len(self.word_to_tok.keys()) - 1


class AnomalyDetector:
    """
    Deploys model on deployment data, producing outputs in the form of protocol logs
    and a summarized .csv of sequences that are deemed anomalous.

    Methods:
    --------
    - generate_scores: Generates anomaly scores using average of geometric means of token probabilites
                       and token anomalies (1-probability)
    """
    def __init__(self, model, test_data, test_df, mask, threshold, tok_to_word, device='cpu'):
        self.model = model
        self.device = device
        self.test_data = test_data
        self.test_df = test_df
        self.mask = mask
        self.threshold = threshold
        self.tok_to_word = tok_to_word
        self.output = None
        self.score_dict = dict()

    def _generate_scores(self, data: torch.Tensor, mask: torch.Tensor, batch_size: int = 4096):
        """
        Generate an anomaly score by averaging the predicted probabilities for known tokens in sequence.

        Params:
        -------
        - data (Tensor): Tensor of sequences
        - mask (Tensor): Tensor of masked items
        - batch_size (int): Size of batch to vectorize on

        Returns:
        --------
        - anomaly_scores (Numeric): Score from 0 to 1 representing how anomalous the sequence is in comparison others
        - anomaly_tokens (numpy.array): Score from 0 to 1 for each token in deployment data
        """
        y_test = data.clone()  # Generate true values
        self.seq_len = data.shape[1]
        probability_score = np.ones(len(data))  # Placeholder for probability scores geometric mean calculation
        anomaly_score = np.ones(len(data))  # Placeholder for anomaly scores geometric mean calculation
        anomaly_tokens = np.zeros(data.shape)  # Placeholder for each token

        # Iterate through batch
        for start in range(0, len(data), batch_size):
            # Data prep
            end = min(start + batch_size, len(data))
            X_batch = data[start:end].clone()
            mask_batch = mask[start:end]
            y_batch = y_test[start:end]

            tokens_size = end - start  # Get size of batch for token score matrix creation
            batch_tokens = np.zeros((tokens_size, self.seq_len))  # Create prlaceholder matrix for token scores

            # Iterate through sequences and mask each token individually
            # Anomaly scores must be generated and unchanged at and after the end of sequence token
            for j in range(1, self.seq_len):
                obs_mask = mask_batch[:, j]
                label = y_batch[:, j]

                X_batch[:, j] = 4  # Mask each value in sequence

                # Generate predicted output for tokens
                with torch.no_grad():
                    output = self.model(X_batch.to(self.device), mask_batch.to(self.device))

                probs = F.softmax(output.logits, dim=-1)  # Calculate probabilities
                masked_probs = probs.squeeze()[:, j, :]

                # Generate probabilities and anomalies
                probabilities = masked_probs[torch.arange(masked_probs.size(0)), label].cpu().numpy()
                anomalies = 1 - probabilities

                # Handles excess padding (multiply by one so the geometric mean won't be affected by padding)
                probabilities[obs_mask == 0] = 1
                anomalies[obs_mask == 0] = 1

                batch_tokens[:, j] = 1 - probabilities  # Populate token anomaly matrix with anomaly scores per token

                # Build geometric mean
                probability_score[start:end] *= probabilities
                anomaly_score[start:end] *= anomalies

            anomaly_tokens[start:end] = batch_tokens  # Populate anomaly_tokens matrix with anomaly scores

        # Get number of items in sequence for geometric mean
        unmasked_sum = mask.sum(dim=1).cpu().numpy()

        # Generate anomaly scores
        probability_score **= (1 / (unmasked_sum - 1))  # Subtract 1 because the BOS token isn't counted
        probability_score_inverted = 1 - probability_score  # Invert to get anomaly score
        anomaly_score **= (1 / (unmasked_sum - 1))
        anomaly_scores = (probability_score_inverted + anomaly_score) / 2  # Average geometric means to avoid bias

        # Generate anomaly_tokens
        anomaly_tokens = np.round(anomaly_tokens[:, 1:-1], 3)

        logger.debug("Anomaly scores generated for deployment data.")

        return anomaly_scores, anomaly_tokens

    def generate_output(self):
        """
        Generates anomaly scores and produces output in the form of summarized .csv containing
        token scores, sequence scores, and event information.
        """
        # Generate outputs for both sequence and token anomaly scores and filter on threshold
        anomalies, scores = self._generate_scores(self.test_data, self.mask)
        inds = np.argwhere(anomalies > self.threshold).flatten()

        # Create output DF combining event information, sequence anomaly scores, and individual token anomaly scores
        self.output = self.test_df.iloc[inds][['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'sequence']]
        self.output['score'] = anomalies[inds]

        # Create columns for each token's anomaly score
        new_cols = []
        for i in range(scores.shape[1]):
            new_cols.append(f"log{i+1}")

        self.output[new_cols] = scores[inds]

        logger.debug(f"{len(self.output)} anomalies found")

        # Trim output to only tokens the model sees
        self.output['sequence'] = self.output['sequence'].apply(self._format_history)
        self.output.reset_index(drop=True, inplace=True)

        logger.debug("Summarized output created")

        # Create dictionary containing anomaly scores for each protocol type for follow-on output requested by analysts
        mask = self.mask[inds]  # Filter mask output
        scores = scores[inds]  # Filter scores output

        for index, row in self.output.iterrows():
            mask_ind = int(sum(mask[index]) - 2)  # Subtract to get rid of reserve tokens
            legit_values = scores[index][:mask_ind].tolist()  # Generate list of tokens
            self.score_dict[row.uid] = legit_values  # Assign tokens to uid in Dictionary

        logger.debug("Protocol score dictionary created")

    def _format_history(self, row: tuple) -> tuple:
        """
        Trimming history column to user specified number of items

        Params:
        -------
        - row (tuple): sequence of logs

        Returns:
        --------
        - row (tuple): trimmed sequence of logs
        """
        if len(row) < self.seq_len:
            row = row[1:len(row) - 1]
        else:
            row = row[1:self.seq_len - 1]  # Get rid of reserve tokens
        return tuple(self.tok_to_word[x] for x in row)
