# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED


import os
import pickle
import pandas as pd
from typing import Tuple
from loguru import logger

from zat.log_to_dataframe import LogToDataFrame

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Zeek():
    """
    Designed to handle arbitrary kinds of Zeek logs and provides general utils in conjunction with ZeekCleaner.
    Only requirement is the existence of a conn log. All other logs are optional.

    Iterable and will return a key value pair from data.

    Attributes:
    -----------
    - data (dict): Dictionary whose keys are log file names and values are dataframes associated with that log
    - keys (list): All keys in the data dictionary
    - random_state (int): Random state used in all operations related to this object

    Methods:
    --------
    - get: Gets the df associated with a log type
    - set: Creates or overwrites a df associated with a log type
    - delete_log: Deletes the passed log file altogether
    - n_connections: Returns the number of rows in the conn log
    - n_logs: Gets the number of unique log types stored
    - read: Import data from files in a provided directory
    - train_test_split: Train/test split the data and receive two new Zeek objects
    - remove: Removes all connections with a column matching certain criterion
    - remove_duplicate_connections: Removes all references to UIDs that appear twice, keeping the first
    - remove_empty_connections: Removes all unsuccessful and 0 data connections
    - reset_index: Resets the index of each DataFrame with drop=True
    - keep_n_connections: Will filter the data to only have n unique connections
    - sort: Sorts a given log's dataframe by a specified column
    - sequence: Creates the sequence df under the key 'sequence'
    - to_torch_loader: Turns preprocessed dataframes into PyTorch dataloaders
    - save: Save the current zeek object to a specified file
    - load: Load an existing Zeek object from a pickle
    """
    def __init__(self, data: dict = None, data_path: str = None, random_state: int = 42):
        """
        Params:
        -------
        - data (dict): If provided, sets the data equal to this dictionary
        - data_path (str): If provided, loads the data from this path. Requires UIDs and limits files.
                           For more flexibility, call .read() yourself
        - random_state (int): Random state for all data related processes
        """
        self.random_state = random_state
        self.data = data if data else {}
        self._processed = False
        self.vocab = None

        # These are the kinds of files that currently support being read
        self.ACCEPTED_EXT = ['.log', '.parquet', '.csv']

        if data_path:
            self.read(data_path, require_uid=True, limit_files=True)

    def get(self, log: str) -> pd.DataFrame:
        """
        Gets the dataframe associated with the logfile name passed in.

        Params:
        -------
        - log (str): Name of the log file. i.e. conn, dns, etc.

        Returns:
        --------
        - pd.DataFrame: DataFrame corresponding to this log file

        Raises:
        -------
        - RuntimeError: Instance has not yet been populated with data
        - ValueError: Key not in Zeek instance
        """
        keys = list(self.data.keys())
        if keys is None:
            logger.error("The Zeek dataset object was never populated with data, but data reading is being "
                         "attempted anyway. The object needs to be populated with a successful .read() call before "
                         "data can be read. (Zeek.get)")
            raise RuntimeError("Object instantiated, but not populated. Please call Zeek.read() to load data"
                               "or manually populate with Zeek.set().")

        if log not in self.data:
            logger.error("Nonexistent data is attempted to be read. This only will occur if that log file was "
                         "deemed critical (like a conn log). (Zeek.get)")
            raise ValueError(f"{log} not found in Zeek instance.")

        return self.data[log]

    def set(self, key: str, df: pd.DataFrame):
        """
        Updates the data dictionary with this new (or overwrites an existing) key/value pair.

        Params:
        -------
        - key (str): Name of the log file. i.e. conn, dns, etc.
        - df (DataFrame): df associated with the logfile name
        """
        self.data[key] = df

    def delete_log(self, key: str):
        """
        Deletes the data dictionary at this key.

        Params:
        -------
        - key (str): Name of the log file. i.e. conn, dns, etc.
        """
        if key not in self.data:
            logger.warning(f"Attempted to delete nonexistent entry ({key}) from a Zeek dataset. (Zeek.delete_log)")

        self.data.pop(key, None)
        self.keys = self.data.keys()

    def n_connections(self) -> int:
        """
        Gets the number of unique connections in the dataset via the size of the connection log
        """
        if 'conn' in self.data:
            return len(self.get('conn'))
        return 0

    def n_logs(self) -> int:
        """
        Get the number of unique log types in the dataset.
        """
        return len(self.data.keys())

    def read(self, base_path: str, require_uid: bool = True, limit_files: bool = True, known_logs: list = None):
        """
        Creates the data dictionary given a file path to a directory containing log files.
        All log files in the given directory and subdirectories will be parsed. Assumes naming schema
        <log type>.extension.

        Supports .log, .csv, & .parquet.

        Params:
        -------
        - base_path (str): Base directory path to traverse. Will traverse subdirectories too
        - require_uid (bool): When true, only processes log files that contain a UID column
        - limit_files (bool): When true, only processes log files found on the Zeek cheatsheet
        - known_logs (list): If limit_files, pass this optionally to choose what log files are looked at

        Raises:
        -------
        - RuntimeError: No parquet or log files were found in the specified path
        """
        # These are the "known" log types per the zeek cheatsheet. Known services was added for utility.
        if limit_files and not known_logs:
            known_logs = ['conn', 'dhcp', 'dns', 'dpd', 'files', 'ftp', 'http', 'kerberos',
                          'mysql', 'radius', 'sip', 'ssh', 'ssl', 'syslog', 'weird',
                          'x509', 'dce_rpc', 'nltm', 'rdp', 'smb_files', 'smb_mapping',
                          'known_services']

        # In case the filepath is off
        base_path = os.path.normpath(base_path)
        base_path = os.path.abspath(base_path)

        # Have to do os.walk because sometimes the files are grouped by day in separate folders
        found = False
        for root, dirs, files in os.walk(base_path):
            for file in files:
                # Get the file and the extension
                path = os.path.join(root, file)
                log_type, extension = os.path.splitext(file)
                log_type = log_type.lower()
                extension = extension.lower()

                # Make sure it is a log file that we want
                if limit_files and log_type not in known_logs:
                    continue

                # Handle it accordingly, can add more to this like csv
                if extension == ".log":
                    found = True
                    self._read_log_file(path, log_type, require_uid)
                elif extension == ".parquet":
                    found = True
                    self._read_parquet_file(path, log_type, require_uid)
                elif extension == ".csv":
                    found = True
                    self._read_csv_file(path, log_type, require_uid)

        # If the conn log wasn't read, this can be a problem
        if self.n_connections() == 0:
            logger.warning("conn log is either empty or never read. Please ensure it's named 'conn.<ext>' and "
                           "that it is populated.")

        if not found:
            logger.error(f"No data of type ({self.ACCEPTED_EXT}) was found at the provided data source. (Zeek.read)")
            raise RuntimeError("No valid files found in the directory")

    def train_test_split(self, test_ips: list = None, n_test: int = None, ratio: float = 0.1,
                         shuffle: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Train and test split the data, returning new Zeek objects that are only the train or test data.

        Params:
        -------
        - test_ips (list): All traffic associated with these IPs will be only in the test set. This might exceed user
                           provided test set sizes
        - n_test (int): Amount of connections to put in the test set if malicious ips is not given
        - ratio (float): The train/test split ratio. If ratio is 0.1, 10% of the data is test data
        - shuffle (bool): If true, will shuffle the dataframe, including the test set!

        Returns:
        --------
        - Zeek: Training data
        - Zeek: Test data
        """
        # Need to make a copy since we return new Zeek objects
        train = self.data.copy()
        test = {}

        if (n_test and n_test > self.n_connections()) or (ratio and ratio >= 1):
            n_test = self.n_connections()
            logger.warning("Test size bigger than dataset, train will be type None. Make sure ratio < 1 and "
                           "n_test < # of UIDs. (Zeek.train_test_split)")

        # Everything is done in terms of n_test
        if n_test is None:
            n_test = int(self.n_connections() * ratio)

        # We want all the malicious traffic (that we at least know of) in the test set
        if test_ips:
            malicious_conn = self.get("conn")
            malicious_mask = malicious_conn["id.orig_h"].isin(test_ips) | malicious_conn["id.resp_h"].isin(test_ips)
            malicious_conn = malicious_conn[malicious_mask]
            malicious_uids = malicious_conn["uid"]

            for key, df in train.items():
                attacker_mask = df["uid"].isin(malicious_uids)
                test[key] = df[attacker_mask]
                train[key] = df[~attacker_mask]

            n_test -= len(test["conn"])

        # Sample the remaining from other examples if we haven't already hit the threshold
        if n_test > 0:
            if shuffle:
                sampled_uids = train["conn"]["uid"].sample(n=n_test, replace=False, random_state=self.random_state)
            else:
                sampled_uids = train["conn"]["uid"]
                sampled_uids = sampled_uids[:n_test]

            # Make the train and test sets with samples UIDs iff n_test > 0 since otherwise
            # nothing is sampled
            for key, df in train.items():
                uid_mask = df['uid'].isin(sampled_uids)
                if not uid_mask.any():
                    continue

                if key in test:
                    test[key] = pd.concat([test[key], df[uid_mask]], axis=0, ignore_index=True)
                else:
                    test[key] = df[uid_mask].copy()

                train[key] = df[~uid_mask]

        # Remove any dataframes from the train and test set that are empty
        keys = list(self.data.keys())
        for key in keys:
            if key in test and test[key].empty:
                test.pop(key)
            if train[key].empty:
                train.pop(key)

        # Make them new Zeek objects
        train = Zeek(data=train, random_state=self.random_state)
        test = Zeek(data=test, random_state=self.random_state)
        return train, test

    def remove(self, col: str = None, values: list = None, mask: list = None, log: str = "conn"):
        """
        Removes all connections containing the provided values in the provided column.
        MUST specify either a column value pair or a mask

        Params:
        -------
        - log (str): Log file we are searching, defaults to conn
        - cols (str): Column that we wish to filter on
        - mask (list): A pandas boolean mask that can be used to remove certain UIDs
        - values (list): Values that should be removed
        """
        df = self.get(log)

        # Warn a user if the column doesn't exist
        if (col is not None) and (col not in df.columns):
            logger.warning(f"{col} not found in {log}. Is the column spelled correctly and are you searching "
                           "the correct log file?")
            return

        # Make the mask with values and col if we don't have it
        if (mask is None) and (values is None):
            logger.warning("Removal of items in a Zeek dataset is attempted without providing any values or mask.")
            return
        if values is not None:
            mask = df[col].isin(values)
        to_remove = df[mask]['uid']

        # If nothing is removed, it could be an accident
        if len(to_remove) == 0:
            logger.info(f"No entries removed for column {col}. Are you certain you have the correct values?")

        for key, df in self.data.items():
            mask = df['uid'].isin(to_remove)
            self.set(key, df[~mask])

    def remove_duplicate_connections(self):
        """
        Removes all repeated UIDs from the conn log ONLY. Just keeps the first occurrence of the duplicate.
        """
        conn = self.get('conn')
        conn = conn.drop_duplicates(subset='uid', keep='first')
        self.set('conn', conn)

    def remove_empty_connections(self):
        """
        Removes all UIDs that are only involved in the conn log and no other logs.
        Also removes empty rows.
        """
        long_uids = set()
        for log, df in self.data.items():
            if log == 'conn':
                continue
            long_uids |= set(df['uid'])

        filtered = self.get('conn')
        filtered = filtered[filtered['uid'].isin(long_uids)]
        filtered = filtered[~filtered['uid'].isna()]
        self.set('conn', filtered)

    def reset_index(self):
        """
        Resets teh index of all dataframes in the Zeek object.
        """
        for df in self.data.values():
            df.reset_index(drop=True, inplace=True)

    def keep_n_connections(self, n: int, allowed_states: list = None, shuffle: bool = False):
        """
        Keeps only n UIDs from the data

        Params:
        -------
        - n (int): Number of unique connections to keep
        - allowed_states (list): List of conn_state values to allow to be kept
        - shuffle (bool): If true, takes n random UIDs
        """
        # Filter for only the allowed states
        conn = self.get("conn")
        if allowed_states:
            allowed_mask = conn["conn_state"].isin(allowed_states)
            conn = conn[allowed_mask]
            if len(conn) < n:
                logger.warning(f"Less than {n} observations met the allowed state criterion for keeping n connections. "
                               f"You will have less than {n} connection in the resulting dataset.")

        # n needs to be an int
        n = int(min(n, len(conn)))

        # Keep only the first n uids
        uids = conn["uid"]
        if shuffle:
            uids = uids.sample(n=n, random_state=self.random_state).reset_index(drop=True)
        else:
            uids = uids[:n]

        # Filter all the dfs
        for key, df in self.data.items():
            mask = df["uid"].isin(uids)
            self.data[key] = df[mask]

    def sort(self, log: str = "conn", col: str = "ts", ascending: bool = True):
        """
        Sorts the provided log by the provided column.

        Params:
        -------
        - log (str): Log to be sorted
        - col (str): Column to sort by
        - ascending (bool): Sort ascending if true, otherwise descending
        """
        df = self.get(log)
        df = df.sort_values(by=col, ascending=ascending, inplace=True)

    def sequence(self, vocab: dict = None, seq_name: str = None, known_transforms: list = None) -> dict:
        """
        Sequences the data so that it is a time series per UID.

        Params:
        -------
        - vocab (dict): Mappings of words to token numbers
        - seq_name (str): What key to store the sequence under. If left alone, defaults to 'sequence'
        - known_transforms (list): The keys for which there is a defined transform. Ensures that an outlier bucket
                                   is added even if there are no outliers during training.

        Returns:
        --------
        - dict: A new vocabulary mapping, or just the passed in vocab if it was supplied
        """
        # Dictionary for lookup. Make it if it doesn't exist
        if not vocab:
            vocab = self._make_initial_vocab(known_transforms)

        # This is the foundation for the sequence dataframe
        combined = pd.DataFrame(columns=['ts', 'uid', 'source'])
        dtypes = {'ts': 'datetime64[ns]', 'uid': 'object', 'source': 'int64'}
        combined = combined.astype(dtypes)

        # Make a source column
        for key, df in self.data.items():
            logger.trace(f"Sequencing {key} of shape {df.shape}")
            df_copy = df.copy()
            # If the key doesn't exist in vocab, it's unkown
            if not any(existing_key.startswith(key) for existing_key in vocab):
                df_copy['source'] = vocab["UNK"]  # Unknown token
            elif 'cluster' in df.columns:
                # This drops anything that, for whatever reason, wasn't assigned a cluster.
                # This was a problem during testing because, by virtue of how BatchDB works, when the baseline
                # data was in the test set it would not be assigned a cluster and thus give a NaN value here.
                # df_copy['cluster'] = df_copy['cluster'].fillna(-1)

                df_copy.loc[:, 'temp'] = key + df_copy['cluster'].astype(int).astype(str)
                df_copy.loc[:, 'source'] = df_copy['temp'].map(vocab)
                df_copy.drop('temp', axis=1, inplace=True)

            else:
                df_copy['source'] = vocab[key]
            combined = pd.concat([combined, df_copy[['ts', 'uid', 'source']]], axis=0)

        combined = combined.sort_values(by=['uid', 'ts', 'source'], ascending=[True, True, False])
        sequenced = combined.groupby('uid').apply(lambda x: pd.Series({
            'sequence': x['source'].tolist(),
            'ts': x['ts'].tolist()[0]
        })).reset_index()

        name = seq_name if seq_name else 'sequence'
        self.data[name] = sequenced

        self.vocab = vocab
        return vocab

    def _make_initial_vocab(self, known_transforms) -> dict:
        vocab = {"UNK": 1}
        counter = 5

        for key, df in self.data.items():
            if key == 'conn':
                continue
            if 'cluster' in df.columns:
                vals = df['cluster'].unique()
                has_negative_one = False  # flag to check if -1 is present
                for val in vals:
                    vocab[f"{key}{val}"] = counter
                    counter += 1
                    if val == -1:
                        has_negative_one = True
                if not has_negative_one and key in known_transforms:
                    vocab[f"{key}-1"] = counter  # add the key-1 if not present
                    counter += 1
            else:
                vocab[key] = counter
                counter += 1

        # Make sure we do the conn log last per Garrett's request
        conn = self.get('conn')
        if 'cluster' in conn.columns:
            vals = conn['cluster'].unique()
            has_negative_one = False  # flag to check if -1 is present
            for val in vals:
                vocab[f"conn{val}"] = counter
                counter += 1
                if val == -1:
                    has_negative_one = True
            if not has_negative_one:
                vocab["conn-1"] = counter  # add the conn-1 if not present
                counter += 1
        else:
            vocab['conn'] = counter
            counter += 1

        return vocab

    def to_torch_loader(self, log: str, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """
        Converts the specified log file to a model ready DataLoader. num_workers can be increased if the data reading
        takes lots of time; highly advised that num_workers <= 6.

        Params:
        -------
        - log (str): Specific log file to be prepared
        - batch_size (int): Batch size of the dataloader
        - shuffle (bool): If true, shuffles the data to give random indices in the batches
        - num_workers (int): Number of subprocesses to use for dataloading. If 0, data is loaded in the main process

        Returns:
        --------
        - DataLoader: DataLoader with the data from the particular log file specified by 'log'

        Raises:
        -------
        - ValueError: The specified log is not found
        """
        if log not in self.data:
            logger.error(f"Attempt to convert nonexistent {log} log into a torch dataset/ (Zeek.to_torch_loader)")
            raise ValueError(f"{log} not found in Zeek instance.")

        if not self._processed:
            logger.warning("Data preprocessing for this Zeek object not known to have occurred. Models might be "
                           "unable to handle the input. (Zeek.to_torch_loader)")

        df = self.get(log)
        dataset = ZeekTorch(df)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def save(self, path: str):
        """
        Save a Zeek dataset as a pickle file for later use.

        Params:
        -------
        - path (str): Pickle file to write to

        Raises:
        -------
        - ValueError: Path provided is not a pickle file
        """
        _, ext = os.path.splitext(path)
        if ext != ".pkl":
            logger.error(f"ZeekCleaner objects are expected to be saved as a pickle file, but {ext} type provided. "
                         "(Zeek.save)")
            raise ValueError("ZeekCleaner objects be saved to a pickle file.")

        path = os.path.normpath(path)
        path = os.path.abspath(path)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str) -> 'Zeek':
        """
        Read an existing Zeek dataset from a pkl file.

        Params:
        -------
        - path (str): Location of the pickle file

        Returns:
        --------
        - Zeek: A new Zeek instance with all of your previous settings

        Raises:
        -------
        - ValueError: Path provided is not a pickle file
        """
        _, ext = os.path.splitext(path)
        if ext != ".pkl":
            logger.error(f"ZeekCleaner objects are expected to be read from a pickle file, but {ext} type provided. "
                         "(Zeek.load)")
            raise ValueError(f"Provided file must be a pickle file, not {ext}.")

        with open(path, 'rb') as file:
            new_zeek = pickle.load(file)

        return new_zeek

    def _read_log_file(self, file_path: str, log: str, require_uid: bool):
        """
        Adds the log file at the path to the dictionary, or concats it if it already exists.

        Params:
        -------
        - file_path (str): Path to the log file
        - log (str): Name of the logfile for indexing in the dictionary
        - require_uid (bool): When true, only processes log files that contain a UID column
        """
        log_to_df = LogToDataFrame()
        df = log_to_df.create_dataframe(file_path).reset_index()
        if (len(df) == 0) or (require_uid and "uid" not in df.columns):
            return

        if log in self.data:
            # Categories cause the concatenation to throw warnings if they don't exactly match
            # For now, make them objects. They can be categorized later
            self._convert_category_to_object(df)
            self.data[log] = pd.concat([self.data[log], df], axis=0, ignore_index=True)
        else:
            self.data[log] = df

    def _read_parquet_file(self, file_path: str, log: str, require_uid: bool):
        """
        Adds the parquet file at the path to the dictionary, or concats it if it already exists

        Params:
        -------
        - file_path (str): Path to the parquet file
        - log (str): Name of the logfile for indexing in the dictionary
        - require_uid (bool): When true, only processes log files that contain a UID column
        """
        df = pd.read_parquet(file_path)

        if (len(df) == 0) or (require_uid and "uid" not in df.columns):
            return

        if log in self.data:
            self._convert_category_to_string(df)
            self.data[log] = pd.concat([self.data[log], df], axis=0, ignore_index=True)
        else:
            self.data[log] = df

    def _read_csv_file(self, file_path: str, log: str, require_uid: bool):
        """
        Adds the csv file at the path to the dictionary, or concats it if it already exists

        Params:
        -------
        - file_path (str): Path to the csv file
        - log (str): Name of the logfile for indexing in the dictionary
        - require_uid (bool): When true, only processes log files that contain a UID column
        """
        df = pd.read_csv(file_path)

        if (len(df) == 0) or (require_uid and "uid" not in df.columns):
            return

        if log in self.data:
            self._convert_category_to_string(df)
            self.data[log] = pd.concat([self.data[log], df], axis=0, ignore_index=True)
        else:
            self.data[log] = df

    def _convert_category_to_object(self, df: pd.DataFrame):
        """
        An issue arose where columns deemed as categories could not be concatenated due to differing category types.
        This converts all categories to strings.
        """
        for column in df.select_dtypes(include=['category']).columns:
            df[column] = df[column].astype('str')

    def _set_processed(self, processed: bool):
        """
        Internal method to change the value of processed. A user should never manually do this.
        """
        self._processed = processed

    # Making the class iterable
    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        """
        Returns (key, df)
        """
        keys = list(self.data.keys())
        if self.counter < len(keys):
            key = keys[self.counter]
            self.counter += 1
            return key, self.data[key]
        raise StopIteration

    # For pritning
    def __str__(self):
        return f"Zeek Object w/ {self.n_connections()} Logs"


class ZeekTorch(Dataset):
    """
    PyTorch dataset used by Zeek class to prepare a dataloader.
    Will only return the input X. Assumes reconstruction-based loss.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Params:
        -------
        - df (DataFrame): DataFrame to be used as model inputs
        """
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Only returns X. Does NOT return a target of any kind.
        """
        row = self.df.iloc[idx]
        row = row.astype(float)
        return torch.tensor(row.values, dtype=torch.float32)
