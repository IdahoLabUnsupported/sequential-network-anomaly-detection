# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import os
import pickle
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import DBSCAN


class BatchDBSCAN():
    """
    BatchedDBSCAN is something I came up with to massively speed up runtime on large datasets.
    The idea is pretty simple but makes a very key assumption that CLUSTERS ARE WELL SEPARATED.
    If this is not the case, I do not recommend using this.

    Essentially, batch the data into chunks (after shuffling). Perform an initial DBSCAN on one batch, and save the
    resulting clusters as assignments. These will now define the clusters. Then, train more DBSCANs on the remaining
    batches, but include the initial assignments in the batch. Then, using the assignments, map new clusters to the
    original clusters so that the values are consistent.

    Again, if the data are not well separated, this can cause problems. If the data are well separated, the performance
    is typically identical to just running DBSCAN on the original dataset.

    Also, this has the added benefit of being able to have a 'predict' method.

    Time Complexity: O(n^2) -> O(m^2 * (n/m)); where m << n is the batch size.

    Attributes:
    -----------
    - eps (float)
    - min_samples (int)
    - batch_size (int)
    - random_state (int)

    Methods:
    --------
    - fit_predict: Fit the model on the new data and return the assignments.
    - predict: Predict on new data.
    """
    def __init__(self, eps: float = 1.0, min_samples: int = 1000, batch_size: int = 2e5, random_state: int = 42):
        """
        Initialize a BatchDBSCAN object just like a regular DBSCAN object.

        Params:
        -------
        - eps (float): The maximum allowable distance between a point and all of it's neighbor to be put into the
                       same cluster
        - min_samples (int): How many samples need to be in a cluster before it is deemed as such
        - batch_size (int): This controls the maximum number of points DBSCAN'd togetherand has the impact
                            on execution time
        - random_state (int): Random seed for all shuffling of data during batching
        """
        self.eps = eps
        self.min_samples = min_samples
        self.batch_size = batch_size
        self.random_state = random_state
        # TODO: custom distance metric?
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.baseline_data = pd.DataFrame()
        self._fitted = False  # Needed for fit_predict

    def _initial_fit(self, data: pd.DataFrame) -> np.ndarray:
        """
        Just performs the initial fit. This is seaprated into its own method for readability.
        """
        self._fitted = True
        return self.model.fit_predict(data)  # Note that this calls model.fit_predict, which is sklearn's DBSCAN

    def _sample_from_clusters(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Gets min_samples random points from each cluster to save as the defining cluster points.
        """
        clusters = data['cluster'].unique()
        all_samples = pd.DataFrame()

        for cluster in clusters:
            if cluster == -1:
                continue
            cluster_data = data[data['cluster'] == cluster]
            samples = cluster_data.sample(n=self.min_samples, random_state=self.random_state)
            all_samples = pd.concat([all_samples, samples], axis=0)

        return all_samples

    def fit_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the DBSCAN to the data and return the cluster assignments.

        Params:
        -------
        - data (pd.DataFrame): The data

        Returns:
        --------
        - pd.DataFrame(columns=['uid', 'cluster']): The cluster assignments, with the UIDs kept
        """
        # Only batch if we need to
        if len(data) > self.batch_size:
            assignments = pd.DataFrame(columns=['uid', 'cluster'])

            # Shuffling increases the probability we get all clusters well represented in the baseline data
            shuffled_data = data.sample(frac=1, replace=False, random_state=self.random_state)
            shuffled_data['cluster'] = -2  # -2 is just a placeholder I used during debugging

            for start in range(0, len(shuffled_data), self.batch_size):
                end = min(start + self.batch_size, len(shuffled_data))
                batch = shuffled_data.iloc[start:end]
                model_data = batch.drop(columns=['ts', 'uid', 'cluster'])

                # The first time around, we have to create the baseline
                if not self._fitted:
                    self.baseline_data = batch.copy()
                    batch.loc[batch.index, 'cluster'] = self._initial_fit(model_data)
                    self.baseline_data = self._sample_from_clusters(batch)
                    assignments = pd.concat([assignments, batch[['uid', 'cluster']]], axis=0)
                else:
                    pred = self.predict(batch)
                    assignments = pd.concat([assignments, pred], axis=0)

            return assignments

        else:
            # For the small columns, just fit.
            model_data = data.drop(columns=['ts', 'uid'])
            data['cluster'] = self._initial_fit(model_data)
            self.baseline_data = self._sample_from_clusters(data)
            return data[['uid', 'cluster']]

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Uses the learned baseline clusters to predict where new points should be.

        Params:
        -------
        - data (pd.DataFrame): New data to cluster with the existing model

        Returns:
        --------
        - pd.DataFrame: The cluster assignments, with the UIDs kept
        """
        # Obviously can't run if there is no baseline data
        if not self._fitted:
            logger.error("A BatchDBSCAN object was called for prediction without being fit. Please call "
                         "BatchDBSCAN.fit_predict first.")
            raise RuntimeError("BatchDBSCAN not yet fitted.")

        # We need to batch in this case
        if len(data) > self.batch_size:
            assignments = pd.DataFrame(columns=['uid', 'cluster']).astype({'uid': object, 'cluster': int})
            for start in range(0, len(data), self.batch_size):
                end = min(start + self.batch_size, len(data))
                batch = data.iloc[start:end]
                # Readding the baseline data so that the cluster numbers are consistent
                batch = pd.concat([batch, self.baseline_data], axis=0)
                pred = self._predict_helper(batch)
                assignments = pd.concat([assignments, pred], axis=0)
            return assignments
        else:
            # Same rationale as before. Need to readd
            agg_data = pd.concat([data, self.baseline_data], axis=0, ignore_index=True)
            return self._predict_helper(agg_data)

    def _predict_helper(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function that does the bulk of the predict logic.
        """
        # Need to remove these columns for sklearn
        model_data = data.drop(columns=['ts', 'uid', 'cluster'])
        pred = self.model.fit_predict(model_data)
        data.loc[:, 'cluster'] = pred

        # Maps the new clusters to the original ones so that the cluster numbers are consistent across DBSCAN fits
        cluster_mapping = {}
        filtered = data[data['uid'].isin(self.baseline_data['uid'])]
        for cluster in np.unique(self.baseline_data['cluster']):
            # TODO: Does this really make sense?
            filtered_agg = filtered[filtered['cluster'] == cluster]
            filtered_base = self.baseline_data[self.baseline_data['uid'].isin(filtered_agg['uid'])]
            # This happens when a cluster doesn't exist
            if len(filtered_base) == 0:
                logger.trace(f"Cluster {cluster} not present in batch.")
                continue
            else:
                mode = filtered_base['cluster'].mode().iloc[0]
            cluster_mapping[cluster] = mode

        # Don't put repeat baseline data back into the new data
        # The baseline_data was appended to the end so it can be safely discarded (no row switching ops performed)
        no_baseline = data.iloc[:-len(self.baseline_data)]
        no_baseline.loc[:, 'cluster'] = no_baseline['cluster'].apply(lambda x: cluster_mapping.get(x, -1))

        return no_baseline[['uid', 'cluster']]

    def save(self, path: str):
        """
        Save a BatchDBSCAN for later use.

        Params:
        -------
        - path (str): Location of the pickle file

        Raises:
        -------
        - ValueError: Path provided is not a pickle file
        """
        _, ext = os.path.splitext(path)
        if ext != ".pkl":
            logger.error(f"BatchDBSCAN objects must be saved as a pickle file, but {ext} was provided.")
            raise ValueError("BatchDBSCAN objects be saved to a pickle file.")

        path = os.path.normpath(path)
        path = os.path.abspath(path)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str) -> 'BatchDBSCAN':
        """
        Read an existing BatchDBSCAN from a pkl file.

        Params:
        -------
        - path (str): Location of the pickle file

        Returns:
        --------
        - BatchDBSCAN: A new BatchDBSCAN instance with all of your previous settings

        Raises:
        -------
        - ValueError: Path provided is not a pickle file
        """
        _, ext = os.path.splitext(path)
        if ext != ".pkl":
            logger.error(f"BatchDBSCAN objects must be read from a pickle file, but {ext} was provided.")
            raise ValueError(f"Provided file must be a pickle file, not {ext}.")

        with open(path, 'rb') as file:
            model = pickle.load(file)

        return model
