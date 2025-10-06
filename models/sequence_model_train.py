# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import sys
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
from sklearn.metrics import f1_score
from loguru import logger


class DataProcessor:
    """
    Designed to pull in and combine protocol log vocabulary  into a sequence.
    Prepares the sequences for training on the masked language model.

    Methods:
    --------
    - load_data: Loads and combines tokenized data
    - load_vocab: Loads log vocabulary and creates dictionary mapping log tokens to log types
    - pad_and_split: masks, pads, and splits data
    - data_loader: creates data loader dictionary for input into BERT model
    """
    def __init__(self, seq_df, vocab, seq_len):
        self.seq_df = seq_df
        self.vocab_size = 0
        self.word_to_tok = vocab
        self.seq_len = seq_len
        self.tok_to_word = {}
        self.unpadded_sequence = []
        self.test_sequence = []
        self.X_train = []
        self.X_val = []
        self.y_train = []
        self.y_val = []
        self.m_train = []
        self.m_val = []
        self.test_data = []
        self.m_test = []
        self.train_data_loader = []
        self.val_data_loader = []

    def load_data(self):
        """
        Designed to combine log tokens into one unpadded array.
        Creates a parallel test array to use for thresholding later (because the array is masked differently).
        """
        tok_sequence = self.seq_df['sequence'].copy()

        tok_sequence = tok_sequence.apply(self._prepare_sentence)

        # unpadded_sequence goes onto become the training sequences, test_sequence becomes the test data
        self.unpadded_sequence = list(tok_sequence.copy())
        self.test_sequence = self.test_sequence = [torch.tensor(seq) for seq in tok_sequence.copy()]

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
        logger.debug("Tokenizer dictionaries created")

        # Set vocab size
        self.vocab_size = len(self.word_to_tok.keys()) - 1

    def _prepare_sentence(self, seq: list) -> list:
        """
        Limits the sentence to user specified amount of words and adds the SOS and EOS where applicable.
        Want to force the model to predict the EOS token, so we do not add the EOS token to arrays longer than seq_len.

        Params:
        -------
        - seq (list): list of tokens

        Returns:
        --------
        - seq (list): trimmed and reserved tokenized list of tokens
        """
        total_len = self.seq_len + 2
        seq.insert(0, 2)  # Insert SOS token at the start
        if len(seq) < total_len:
            seq.append(3)  # Append the EOS token
        else:
            seq = seq[:total_len]  # Trim it if we don't use an EOS token
        return seq

    def pad_and_split(self):
        """
        Trim, pad, and split sequences into train, val, and test sets.
        Mask sequences according BERT's Masked Language Model specifications.
        """
        # Mask training sequence and output BERT input arrays (masked data, labels, and binary mask)
        unpadded_masked_sequence, training_labels, training_mask = self._mask_sequences(self.unpadded_sequence)

        # Pad training and test sequences
        training_padded_sequence = pad_sequence(unpadded_masked_sequence, batch_first=True)
        # -100 is Bert's ignore token
        training_padded_labels = pad_sequence(training_labels, batch_first=True, padding_value=-100)
        training_attention_mask = pad_sequence(training_mask, batch_first=True)

        testing_padded_data = pad_sequence(self.test_sequence, batch_first=True)

        # Shuffle training and test sequences
        num_rows = len(training_padded_sequence)
        shuffled_inds = np.random.permutation(num_rows)

        shuffled_training_data = training_padded_sequence[shuffled_inds]
        shuffled_training_labels = training_padded_labels[shuffled_inds]
        shuffled_training_mask = training_attention_mask[shuffled_inds]

        shuffled_testing_data = testing_padded_data[shuffled_inds]
        self.seq_df = self.seq_df.iloc[shuffled_inds].reset_index(drop=True)

        # Split training, validation, testing (thresholding) data
        split1 = int(num_rows * 0.7)
        split2 = int(num_rows * 0.9)

        self.X_train = shuffled_training_data[:split1]
        self.X_val = shuffled_training_data[split1:split2]

        self.y_train = shuffled_training_labels[:split1]
        self.y_val = shuffled_training_labels[split1:split2]

        self.m_train = shuffled_training_mask[:split1]
        self.m_val = shuffled_training_mask[split1:split2]

        self.test_data = shuffled_testing_data[split2:]
        self.m_test = shuffled_training_mask[split2:]

        self.seq_df = self.seq_df.iloc[split2:].reset_index(drop=True)
        logger.debug("Training, validation, and thresholding data created")

    def _mask_sequences(self, sequences: list) -> tuple:
        """
        Adding reserved tokens for start and end of sequence, masking values in accordance with
        BERT guidelines, generating labels and attention mask to guide training.

        Params:
        --------
        - sequences (list of lists): array of tokenized sequences

        Returns:
        --------
        - masked_sequence, labels, attn_mask (tuple): (masked sequence of tokens, labels for masked items, binary mask)
        """
        labels = []
        attn_mask = []
        masked_sequence = []

        # Add designated start of sequence and end of sequence tokens
        for seq in sequences:
            # Generate placeholders for labels and attn_mask
            masked = [-100] * len(seq)  # Reserved token for model to ignore (labels creation)
            attn = [1] * len(seq)  # Attention mask (attn_mask creation)

            # Generate masking logic
            i = 0.25
            num_masked = math.ceil(i * len(seq))
            replace = random.uniform(0, 1)  # Placeholder used to guide what the replacement value is in training
            ind = random.sample(range(1, len(seq)), num_masked)  # index(s) of tokens to be replaced in training

            # Handle masking
            if replace <= 0.8:  # 80% of chosen tokens replaced by reserved mask token (4)
                for i in ind:
                    masked[i] = seq[i]  # Preserve the true value of the token before changing (labels update)
                    seq[i] = 4  # Mask token

            elif replace <= 0.9:  # 10% of chosen tokens replaced by another token in corpus
                for i in ind:
                    masked[i] = seq[i]  # Preserve the true value of the token before changing (labels update)
                    # Create list of tokens available for replacement
                    all_vals = [v for v in self.word_to_tok.values() if v not in {0, 1, 2, 4, -100}]
                    seq[i] = random.choice(all_vals)  # Pick random value to replace token with

            else:  # 10% of chosen tokens left unchanged
                for i in ind:
                    masked[i] = seq[i]  # labels updata
                pass

            # Update labels, attn_mask, and masked_sequence
            labels.append(torch.tensor(masked))  # Masked sequence with only training tokens not ignored
            attn_mask.append(torch.tensor(attn))  # Binary sequence used to tell the model which values aren't padded
            masked_sequence.append(torch.tensor(seq))  # Needed for follow-on padding operations
        return masked_sequence, labels, attn_mask

    def data_loader(self, batch_size: int = 32):
        """
        Creates data loaders for trainv and alidation sets.

        Params:
        --------
        batch_size (int): default size is 32
        """
        self.train_data_loader = self._data_loader(self.X_train, self.y_train, self.m_train, batch_size)
        self.val_data_loader = self._data_loader(self.X_val, self.y_val, self.m_val, batch_size)
        logger.debug("Data Loader complete")

    def _data_loader(self, data: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, batch_size: int) -> list:
        """
        Generating a data loader object per BERT guidelines.

        Params:
        -------
        - data (Tensor): Tensor of sequences
        - labels (Tensor): Tensor of labels
        - mask (Tensor): Tensor of masked items
        - batch_size (int): Number of items in batch

        Returns:
        --------
        - data_loader (list of dictionaries): Batched list of dictionaries containing inputs, labels, and mask
        """
        data_loader = []
        num_batches = math.ceil(len(data) / batch_size)  # Use ceil to account for last batch

        # Generate batched dictionaries for each of the BERT inputs
        for i in range(num_batches):
            beg = i * batch_size
            end = min((i + 1) * batch_size, len(data))  # Ensure we don’t exceed data length
            batch = {
                'input_ids': data[beg:end],
                'mlm_labels': labels[beg:end],
                'attention_mask': mask[beg:end]
            }
            data_loader.append(batch)

        return data_loader


class BertTrainer:
    """
    Uses BERT to train a masked language model on sequenced inputs.

    Methods:
    --------
    - train: Trains model on inputs
    - _evaluate: Evaluates model performance using F1 score as metric
    """
    def __init__(self, config, train_data_loader, val_data_loader, epochs=5, learning_rate=1e-5, device=None):
        self.config = config
        self.model = BertForMaskedLM(config)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.num_epochs = epochs
        self.learning_rate = learning_rate
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        """
        Uses BERT to train a masked language model on sequenced inputs.
        """
        for epoch in range(self.num_epochs):
            self.model.train()
            for batch in self.train_data_loader:
                # Data prep
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['mlm_labels'].to(self.device)

                # Forward
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     labels=labels)
                # Backward
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

            logger.debug(f"Epoch: {epoch}, Training Loss: {loss.item()}")
            self._evaluate(epoch)
        logger.debug("Model training complete")

    def _evaluate(self, epoch):
        """
        Uses BERT on subset of data to evaluate model performance using F1 score.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_f1 = 0

        with torch.no_grad():
            for batch in self.val_data_loader:
                # Data prep
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['mlm_labels'].to(self.device)

                # Forward
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     labels=labels)

                # Predictions using argmax
                preds = torch.argmax(outputs[1], axis=2)

                masked_inds = labels != -100  # Excluding non-masked tokens
                masked_vals = preds[masked_inds]  # Getting predicted values
                masked_labels = labels[masked_inds]  # Getting true values

                # Calculating performance metrics
                total_val_loss += outputs.loss.item()
                total_val_f1 += f1_score(masked_labels.cpu().numpy(), masked_vals.cpu().numpy(), average='macro')

        average_val_loss = total_val_loss / len(self.val_data_loader)
        average_val_f1 = total_val_f1 / len(self.val_data_loader)

        logger.debug(f"Epoch: {epoch}, Validation Loss: {average_val_loss}")
        logger.debug(f"Epoch: {epoch}, Validation F1: {average_val_f1} \n")


class AnomalyDetector:
    """
    Calculates threshold for follow-on deployment operations by using sequence anomaly score and accepted anomaly rate.

    Methods:
    --------
    - generate_scores: Generates anomaly scores using average of geometric means of token probabilites and
                       token anomalies (1-probability)
    - return_anomalies: Finds and logs recommended threshold based on user input
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def generate_scores(self, data: torch.Tensor, mask: torch.Tensor, batch_size: int = 8192):
        """
        Generate an anomaly score by averaging the predicted probabilities for known tokens in sequence

        Params:
        --------
        - data (Tensor): Tensor of sequences
        - mask (Tensor): Tensor of masked items
        - batch_size (int): Batch size for processing

        Returns:
        - anomaly_scores (numpy.array): Array of scores representing how anomalous each sequence is
        """
        y_test = data.clone()  # Generate true values
        seq_len = data.shape[1]
        probability_score = np.ones(len(data))  # Placeholder for probability scores geometric mean calculation
        anomaly_score = np.ones(len(data))  # Placeholder for anomaly scores geometric mean calculation

        # Iterate through batch
        for start in range(0, len(data), batch_size):
            # Data prep
            end = min(start + batch_size, len(data))
            X_batch = data[start:end].clone()
            mask_batch = mask[start:end]
            y_batch = y_test[start:end]

            # Iterate through sequences and mask each token individually
            # Anomaly scores must be generated and unchanged after the end of sequence token
            for j in range(1, seq_len):
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

                # Handle excess padding (multiply by one so the geometric mean won't be affected by padding)
                probabilities[obs_mask == 0] = 1
                anomalies[obs_mask == 0] = 1

                # Build geometric mean
                probability_score[start:end] *= probabilities
                anomaly_score[start:end] *= anomalies

        # Get number of items in sequence for geometric mean
        unmasked_sum = mask.sum(dim=1).cpu().numpy()

        # Calculate geometric mean
        probability_score **= (1 / (unmasked_sum - 1))  # Subtract 1 because the BOS token isn't counted
        probability_score_inverted = 1 - probability_score  # Invert to get anomaly score
        anomaly_score **= (1 / (unmasked_sum - 1))
        anomaly_scores = (probability_score_inverted + anomaly_score) / 2  # Average geometric means to avoid bias
        logger.debug("Anomaly scores calculated on portion of training data")

        return anomaly_scores

    def return_anomalies(self, scores: np.array, percentage: float = 0.001):
        """
        Find anomaly threshold based on defined acceptable ratio.

        Params:
        -------
        - scores (np.array): array of anomaly scores
        - percentage (float): percentage threshold for  accepted anomalies

        Returns:
        --------
        - inds (list): list of anomaly indexes
        - threshold (numeric): threshold used to segregate anomalies, -1 if no good threshold is found
        """
        if len(scores) == 0:
            logger.error("length of output scores is 0: input dataframe likely empty")
            raise RuntimeError("length of scores is 0, future ZeroDivisionError")

        # Attempt to use a step of 0.01 for threshold recommendation
        for thresh in np.arange(0, 1, 0.01):
            total = len(scores)
            inds = np.argwhere(scores > thresh)
            if (len(inds) / total) < percentage:
                logger.success(f"Model trained and validated. Recommended anomaly threshold: {thresh}")
                return inds, thresh

        logger.debug("Accepted Anomaly Rate produced anomaly threshold greater than 0.99")

        for thresh in np.arange(0.99, 1, 0.001):
            total = len(scores)
            inds = np.argwhere(scores > thresh)
            if (len(inds) / total) < percentage:
                logger.warning("Recommended anomaly threshold is abnormally high")
                logger.success(f"Model trained and validated. Recommended anomaly threshold: {thresh}")
                return inds, thresh

        logger.error('A valid threshold value was not found. This means that, in your training set, there were too '
                     'many observations that were very out of distribution. This could happen for two reasons. '
                     '\n\t1) Your training is too small, and thus the transformer failed to learn a meaningful '
                     'distribution.'
                     '\n\t2) There is a large presence of unique anomalous data in the training set. That is, at least '
                     'acceptance rate * len(training set) number of (probably unique) anomalies. '
                     'Are you certain the training set is pure? Perhaps there ')
        raise RuntimeError()
