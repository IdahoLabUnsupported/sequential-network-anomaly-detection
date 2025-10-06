# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import os
import pickle
from loguru import logger

from data.datasets import Zeek
from configs.cleaner_assignments import transforms


class ZeekCleaner:
    """
    ZeekCleaner is designed to abstract away all data preprocessing to one simple fit_transform function. It is also
    completely compatible with Zeek objects.

    It uses LogCleaners (see data.log_cleaners) to preprocess specific types of log files, and each preprocessor is
    tailored to that log.

    Methods:
    - fit: fits a log cleaner (for things like standard scalers) so that they are consistent between train/testing
    - transform: applies the learned transforms to new data
    - fit_transform: sequentially applies fit and then transform
    - save: saves the object for use during inference
    - load: load a saved object to use during inference
    """
    def __init__(self):
        self.transforms = transforms
        self.known_transforms = {k for k, v in self.transforms.items() if str(v) != "NoCleaner"}

    def fit(self, data: Zeek):
        """
        Fits all of the individual cleaners. Highly recommend that you pass in ONLY training data.

        Params:
        -------
        - data (Zeek): The Zeek object we want to fit to
        """
        for log, df in data:
            logger.trace(f"Fitting {log}.")
            if log not in self.transforms:
                logger.warning(f"No transform associated with {log}, skipped. Update this in log_cleaners.py.")
                continue

            cleaner = self.transforms[log]
            cleaner.fit(df)
            logger.debug(f"Fit the {log} cleaner.")

    def transform(self, data: Zeek) -> Zeek:
        """
        Applies known transforms to logs.

        Params:
        -------
        - data (Zeek): The Zeek object that needs to be processed

        Returns:
        --------
        - Zeek: A new Zeek object with processed dataframes
        """
        processed = Zeek()

        for log, df in data:
            logger.trace(f"Transforming {log}.")
            if log not in self.transforms:
                # No need to re-warn the user in transform
                continue

            cleaner = self.transforms[log]

            if not cleaner.fitted():
                logger.warning(f"The transform for {log} was not learned. "
                               "It likely only appears in the test set. Skipped.")
                continue

            processed_df = cleaner.transform(df.copy())
            processed.set(log, processed_df)
            logger.debug(f"Transformed {log}.")

        # Let the Zeek object know it is PyTorch ready
        processed._set_processed(True)
        return processed

    def fit_transform(self, data: Zeek) -> Zeek:
        """
        Fits all the log specific cleaners and returns a transformed set of logs.

        Params:
        -------
        - data (Zeek): The Zeek object that needs to be processed

        Returns:
        --------
        - Zeek: A new Zeek object with processed dataframes
        """
        self.fit(data)
        return self.transform(data)

    def save(self, path: str):
        """
        Save a ZeekCleaner for later use.

        Params:
        -------
        - path (str): Location of the pickle file

        Raises:
        -------
        - ValueError: Path provided is not a pickle file
        """
        _, ext = os.path.splitext(path)
        if ext != ".pkl":
            logger.error(f"ZeekCleaner objects need to be saved as a pickle file, but a {ext} was provided.")
            raise ValueError("ZeekCleaner objects be saved to a pickle file.")

        path = os.path.normpath(path)
        path = os.path.abspath(path)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str) -> 'ZeekCleaner':
        """
        Read an existing ZeekCleaner from a pkl file.

        Params:
        -------
        - path (str): Location of the pickle file

        Returns:
        --------
        - ZeekCleaner: A new ZeekCleaner instance with all of your previous settings

        Raises:
        -------
        - ValueError: Path provided is not a pickle file
        """
        _, ext = os.path.splitext(path)
        ext = ext.lower()

        if ext != ".pkl":
            logger.error(f"ZeekCleaner objects need to be read from a pickle file, but a {ext} was provided.")
            raise ValueError("Provided file must be a pickle file.")

        try:
            with open(path, 'rb') as file:
                new_cleaner = pickle.load(file)
        except Exception as e:
            logger.critical(f"{path} could not be read as a pkl file but had a pickle extension. Perhaps it is "
                            "corrupted? If so, this may REQUIRE RETRAINING or DEBUGGING of ZeekCleaner.save.")
            raise e

        return new_cleaner
