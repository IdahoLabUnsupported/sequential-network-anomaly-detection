# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

from loguru import logger


class LogCleaner(ABC):
    def __init__(self):
        """
        Some parent level methods are needed.
        """
        self._fitted = False
        self._type = "Log"

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """
        Fit methods learn the transform from data.

        MUST set self._fitted = True at the end of the method.
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform returns a new Zeek object with all transformed dataframes.
        """
        pass

    def _enusre_existence_of_columns(self, data: pd.DataFrame, req_cols: list) -> list:
        """
        Sometimes columns don't exist in the data from a partner site. We need to handle this accordingly.
        """
        available = []
        for col in req_cols:
            if col in data.columns:
                available.append(col)
        return available

    def fitted(self):
        """
        Returns true if this model has been fitted.
        """
        return self._fitted

    def __str__(self):
        """
        Returns the cleaner type.
        """
        return f"{self._type}Cleaner"


class ConnCleaner(LogCleaner):
    """
    Converts conn logs per the rationale specified in notes/features/conn.md.
    """
    def __init__(self):
        super().__init__()
        self._type = "Conn"
        self.SCALED_COLS = ['duration', 'byte_ratio', 'packet_ratio', 'orig_pkts', 'resp_pkts',
                            'orig_pkt_size', 'resp_pkt_size']
        self.transformer = None

    def _make_ratio_vars(self, data: pd.DataFrame):
        """
        Internal method used to make all ratio variables
        """
        df = data.copy()

        # Duration is sometimes a timedelta64[ns], so need to make it a float
        if pd.api.types.is_timedelta64_ns_dtype(df['duration']):
            df['duration'] = df['duration'].dt.total_seconds().astype('float64')

        # Fill the NA values here
        TO_FILL = ['duration', 'orig_pkts', 'resp_pkts', 'orig_bytes', 'resp_bytes']
        df[TO_FILL] = df[TO_FILL].fillna(0)

        # Adding 1 fixes the log(0) is undefined and divide by 0
        df['byte_ratio'] = np.log(df['orig_bytes'] / (df['resp_bytes'] + 1) + 1)
        df['packet_ratio'] = np.log(df['orig_pkts'] / (df['resp_pkts'] + 1) + 1)
        df['orig_pkt_size'] = np.log(df['orig_bytes'] / (df['orig_pkts'] + 1.0e-9) + 1)
        df['resp_pkt_size'] = np.log(df['resp_bytes'] / (df['resp_pkts'] + 1.0e-9) + 1)

        return df

    def fit(self, data: pd.DataFrame):
        self.SCALED_COLS = self._enusre_existence_of_columns(data, self.SCALED_COLS)
        df = self._make_ratio_vars(data)
        transformers = []

        # Transform the appropriate columns
        transformers.append(('scale', MinMaxScaler(), self.SCALED_COLS))
        transformers.append(('onehot', OneHotEncoder(), ['proto']))

        # Create and return the column transformer
        self.transformer = ColumnTransformer(transformers, remainder='drop')
        self.transformer.fit(df)
        self._fitted = True

    def transform(self, data: pd.DataFrame):
        df = self._make_ratio_vars(data)

        if not self.transformer:
            raise RuntimeError("Transform not yet learned. Did you fit all cleaner objects?")

        # Making the good_history var
        acceptable_states = ['SF', 'RSTO', 'RSTR']
        df['good_history'] = df['conn_state'].isin(acceptable_states)

        # Binarizing variables
        df['local_resp'] = df['local_resp'].astype('bool')
        df['local_orig'] = df['local_orig'].astype('bool')
        df['in_tunnel'] = df['tunnel_parents'].notna()

        # These will be put back into the data
        TO_KEEP = ['ts', 'uid', 'good_history', 'local_resp', 'local_orig', 'in_tunnel']

        # Transform the data
        augmented_df = self.transformer.transform(df)

        # Fix issue with sparse matrices
        if issparse(augmented_df):
            augmented_df = augmented_df.toarray()

        # Extract the column names from transformers within the ColumnTransformer
        ohe_cols = list(self.transformer.named_transformers_['onehot'].get_feature_names_out())
        col_names = self.SCALED_COLS + ohe_cols
        augmented_df = pd.DataFrame(augmented_df, index=df.index, columns=col_names)
        augmented_df[ohe_cols] = augmented_df[ohe_cols].astype(bool)

        # Rea-add the columns
        recovered_cols = df[TO_KEEP]
        return pd.concat([recovered_cols, augmented_df], axis=1)


class DNSCleaner(LogCleaner):
    """
    Converts DNS logs per the rationale specified in notes/features/dns.md.
    """
    def __init__(self):
        super().__init__()
        self._type = "DNS"
        self._fitted = True  # There is nothing to fit since no scaling happens

    def fit(self, data: pd.DataFrame):
        return

    def transform(self, data: pd.DataFrame):
        df = data.copy()

        # Is the port a known DNS port
        ports = [53, 5355, 5353]
        df['safe_port'] = df['id.resp_p'].isin(ports)

        # Do we get an unusual response code
        fine_codes = [0.0, 1.0, 3.0]
        df['weird_resp'] = ~df['rcode'].isin(fine_codes)

        # Binarizing variables
        df['AA'] = df['AA'].astype('bool')
        df['TC'] = df['TC'].astype('bool')

        # Was the query resolved
        df['resolved'] = df['answers'].notna()

        # These will be put back into the data
        TO_KEEP = ['ts', 'uid', 'safe_port', 'weird_resp', 'AA', 'TC', 'resolved']

        return df[TO_KEEP]


class HTTPCleaner(LogCleaner):
    """
    Converts HTTP logs per the rationale specified in notes/features/http.md.
    """
    def __init__(self):
        super().__init__()
        self._type = "HTTP"
        self.SCALED_COLS = ['request_body_len', 'response_body_len']
        self.transformer = None

    def suspicious_user_agent(self, ua):
        if pd.isna(ua):
            return False
        SUS_AGENTS = ['curl', 'wget', 'powershell', 'python-urllib']
        return any(agent in ua.lower() for agent in SUS_AGENTS)

    def fit(self, data: pd.DataFrame):
        self.SCALED_COLS = self._enusre_existence_of_columns(data, self.SCALED_COLS)

        transformers = []

        # Fill the NA values here
        data[self.SCALED_COLS] = data[self.SCALED_COLS].fillna(0)

        # Transform the appropriate columns
        transformers.append(('scale', MinMaxScaler(), self.SCALED_COLS))

        # Create and return the column transformer
        self.transformer = ColumnTransformer(transformers, remainder='drop')
        self.transformer.fit(data)
        self._fitted = True

    def transform(self, data: pd.DataFrame):
        df = data.copy()

        # Fill the NA values here
        df[self.SCALED_COLS] = df[self.SCALED_COLS].fillna(0)

        if not self.transformer:
            raise RuntimeError("Transform not yet learned. Did you fit all cleaner objects?")

        # Known Port Variable
        ports = [80, 8080, 8088, 8000]
        df['safe_port'] = df['id.resp_p'].isin(ports)

        # Are files being sent?
        non_file = ["[text/html]", "[text/plain]"]
        df['file_sent'] = ~df['resp_mime_types'].isin(non_file)

        # Successful req/resp
        codes = [200.0, 404.0]
        df['successful'] = df['status_code'].isin(codes)

        # Powershell/shell user agents can be suspicious and are a sign of automated web requests
        df['weird_user_agent'] = df['user_agent'].apply(self.suspicious_user_agent)

        # Proxied connection
        df['proxied'] = df['proxied'].notna()

        # These will be put back into the data
        TO_KEEP = ['ts', 'uid', 'safe_port', 'file_sent', 'successful', 'weird_user_agent', 'proxied']

        # Transform the data
        augmented_df = self.transformer.transform(df)

        # Fix issue with sparse matrices
        if issparse(augmented_df):
            augmented_df = augmented_df.toarray()

        # Extract the column names from transformers within the ColumnTransformer
        col_names = self.SCALED_COLS
        augmented_df = pd.DataFrame(augmented_df, index=df.index, columns=col_names)

        # Rea-add the columns
        recovered_cols = df[TO_KEEP]
        return pd.concat([recovered_cols, augmented_df], axis=1)


class FilesCleaner(LogCleaner):
    """
    Converts Files logs per the rationale specified in notes/features/files.md.
    """
    def __init__(self):
        super().__init__()
        self._type = "Files"
        self.SCALED_COLS = ['total_bytes', 'entropy', 'extracted_bytes']
        self.transformer = None

    def suspicious_mime(self, mime):
        if pd.isna(mime):
            return False
        # This can be updated at will
        SUS_MIME = ['sh', 'exe']
        return any(sus in mime for sus in SUS_MIME)

    def fit(self, data: pd.DataFrame):
        self.SCALED_COLS = self._enusre_existence_of_columns(data, self.SCALED_COLS)
        transformers = []

        # Transform the appropriate columns
        transformers.append(('scale', MinMaxScaler(), self.SCALED_COLS))

        # Create and return the column transformer
        self.transformer = ColumnTransformer(transformers, remainder='drop')
        self.transformer.fit(data)
        self._fitted = True

    def transform(self, data: pd.DataFrame):
        df = data.copy()

        if not self.transformer:
            raise RuntimeError("Transform not yet learned. Did you fit all cleaner objects?")

        # Are files being sent?
        df['sus_ext'] = df['mime_type'].apply(self.suspicious_mime)

        # These will be put back into the data
        TO_KEEP = ['ts', 'uid', 'is_orig', 'sus_ext']

        # Transform the data
        augmented_df = self.transformer.transform(df)

        # Fix issue with sparse matrices
        if issparse(augmented_df):
            augmented_df = augmented_df.toarray()

        # Extract the column names from transformers within the ColumnTransformer
        col_names = self.SCALED_COLS
        augmented_df = pd.DataFrame(augmented_df, index=df.index, columns=col_names)

        # Rea-add the columns
        recovered_cols = df[TO_KEEP]
        return pd.concat([recovered_cols, augmented_df], axis=1)


class SSLCleaner(LogCleaner):
    """
    Converts SSL logs per the rationale specified in notes/features/ssl.md.
    """
    def __init__(self):
        super().__init__()
        # Someday, these known versions will need to be updated
        self.KNOWN_VERSIONS = ["TLSv12", "TLSv13"]
        self._type = "SSL"
        self._fitted = True

    def fit(self, data: pd.DataFrame):
        return

    def transform(self, data: pd.DataFrame):
        df = data.copy()
        df.fillna('-1', inplace=True)

        # Known Port Variable
        df['safe_port'] = df['id.resp_p'] == 443

        # Making these columns T/F
        if not pd.api.types.is_bool_dtype(df['established']):
            df['established'] = df['established'].map({'T': True, 'F': False, '-1': False})
        if not pd.api.types.is_bool_dtype(df['resumed']):
            df['resumed'] = df['resumed'].map({'T': True, 'F': False, '-1': False})

        # Are we using a current version
        df['odd_version'] = ~df['version'].isin(self.KNOWN_VERSIONS)

        # Should be using Elliptic Curve Diffie Hellman and Galois/Counter Mode
        df['elliptic_curve'] = df['cipher'].str.contains('ECDH') & df['cipher'].str.contains('GCM')

        # These will be put back into the data
        TO_KEEP = ['ts', 'uid', 'safe_port', 'established', 'resumed', 'odd_version', 'elliptic_curve']
        return df[TO_KEEP]


class KerberosCleaner(LogCleaner):
    """
    Converts Kerberos logs per the rationale specified in notes/features/kerberos.md.
    """
    def __init__(self):
        super().__init__()
        # Someday, these known versions will need to be updated
        self.WEIRD_ERRORS = ["KDC_ERR_S_PRINCIPAL_UNKNOWN", "KDC_ERR_C_PRINCIPAL_UNKNOWN", "KRB_AP_ERR_MODIFIED"]
        self._type = "Kerberos"
        self._fitted = True

    def fit(self, data: pd.DataFrame):
        return

    def transform(self, data: pd.DataFrame):
        df = data.copy()

        # Was our failure one of the weird errors
        df['sus_failure'] = df['error_msg'].isin(self.WEIRD_ERRORS)

        TO_KEEP = ['ts', 'uid', 'sus_failure']
        return df[TO_KEEP]


class WeirdCleaner(LogCleaner):
    """
    Converts weird logs per the rationale specified in notes/features/weird.md.
    """
    def __init__(self):
        super().__init__()
        self._type = "Weird"
        self._fitted = True  # There is nothing to fit

    def fit(self, data: pd.DataFrame):
        return

    def transform(self, data: pd.DataFrame):
        return data


class GeneralCleaner(LogCleaner):
    """
    This is a placeholder cleaner for log files that do not yet have a devoted cleaner.
    """
    def __init__(self):
        super().__init__()
        self._type = "General"
        self.transformer = None

    def fit(self, data: pd.DataFrame):
        df = data.copy()

        # Convert objects to categories
        for column in df.select_dtypes(include=['object']).columns:
            # This definitelyyyyy needs changing
            try:
                # Attempt to convert the column to a 'category' type
                df[column] = df[column].astype('category')
            except TypeError:
                # If a TypeError is raised, drop the column
                df.drop(column, axis=1, inplace=True)

        # Define a column transformer
        transformers = []

        # Standard scale UInt64 columns, filling NA values with 0
        uint64_columns = df.select_dtypes(include=['UInt64']).columns
        if uint64_columns.any():
            df[uint64_columns] = df[uint64_columns].fillna(0)
            transformers.append(('scale_uint64', StandardScaler(), uint64_columns))

        # If there are more than lim categories, just discard it for now (like MAC address)
        lim = 30
        cat_columns = df.select_dtypes(include=['category']).columns
        low_cardinality_cats = [col for col in cat_columns if len(df[col].cat.categories) <= lim]
        high_cardinality_cats = [col for col in cat_columns if len(df[col].cat.categories) > lim]
        df.drop(columns=high_cardinality_cats, inplace=True)

        if low_cardinality_cats:
            transformers.append(('onehot', OneHotEncoder(handle_unknown="ignore"), low_cardinality_cats))

        # Convert timedelta64[ns] to float (seconds) and standard scale
        timedelta_columns = df.select_dtypes(include=['timedelta64[ns]']).columns
        for col in timedelta_columns:
            df[col] = df[col].dt.total_seconds()
        if timedelta_columns.any():
            transformers.append(('scale_timedelta', StandardScaler(), timedelta_columns))

        # Fit and save the transform pipeline
        column_transformer = ColumnTransformer(transformers, remainder='drop')
        column_transformer.fit(df)
        self.transformer = column_transformer

        self._fitted = True

    def transform(self, data: pd.DataFrame):
        df = data.copy()

        if not self.transformer:
            raise RuntimeError("Transform not yet learned. Did you fit all cleaner objects?")

        # Convert objects to categories
        for column in df.select_dtypes(include=['object']).columns:
            # This definitelyyyyy needs changing
            try:
                # Attempt to convert the column to a 'category' type
                df[column] = df[column].astype('category')
            except TypeError:
                # If a TypeError is raised, drop the column
                df.drop(column, axis=1, inplace=True)

        # Filling NA values with 0
        uint64_columns = df.select_dtypes(include=['UInt64']).columns
        if uint64_columns.any():
            df[uint64_columns] = df[uint64_columns].fillna(0)

        # Convert timedelta64[ns] to float (seconds)
        timedelta_columns = df.select_dtypes(include=['timedelta64[ns]']).columns
        for col in timedelta_columns:
            df[col] = df[col].dt.total_seconds()

        # Apply Transforms
        pipeline = self.transformer
        transformed_data = pipeline.transform(df)

        if issparse(transformed_data):
            transformed_data = transformed_data.toarray()

        # Get the column names
        onehot_columns = pipeline.named_transformers_['onehot'].get_feature_names_out()
        uint64_columns = df.select_dtypes(include=['UInt64']).columns

        # Combine column names from transformers
        transformed_columns = list(onehot_columns) + list(uint64_columns) + list(timedelta_columns)

        # Create the transformed DataFrame with the correct index and columns
        transformed_df = pd.DataFrame(transformed_data, index=df.index, columns=transformed_columns)

        # Remove NaN vals
        transformed_df.fillna(0, inplace=True)

        # Readd UID and TS
        transformed_df = pd.concat([df[['ts', 'uid']], transformed_df], axis=1)

        return transformed_df


class NoCleaner(LogCleaner):
    """
    This is a special class that skips all preprocessing and makes all of the logs of this type end up in one cluster.
    """
    def __init__(self):
        super().__init__()
        self._type = "No"
        self._fitted = True  # There is nothing to fit

    def fit(self, data: pd.DataFrame):
        return

    def transform(self, data: pd.DataFrame):
        return data
