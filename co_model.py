# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:54:47 2018

@author: Kit
"""
import numpy as np
import pandas as pd

def cluster(X, max_num_clusters=3, exact_num_clusters=None):
    '''Applies clustering on reduced data, 
    i.e. data where power is greater than threshold.

    Parameters
    ----------
    X : pd.Series or single-column pd.DataFrame
    max_num_clusters : int

    Returns
    -------
    centroids : ndarray of int32s
        Power in different states of an appliance, sorted
    '''
    # Find where power consumption is greater than 10
    data = _transform_data(X)

    # Find clusters
    centroids = _apply_clustering(data, max_num_clusters, exact_num_clusters)
    centroids = np.append(centroids, 0)  # add 'off' state
    centroids = np.round(centroids).astype(np.int32)
    centroids = np.unique(centroids)  # np.unique also sorts
    # TODO: Merge similar clusters
    return centroids

def _transform_data(data):
    '''Subsamples if needed and converts to column vector (which is what
    scikit-learn requires).

    Parameters
    ----------
    data : pd.Series or single column pd.DataFrame

    Returns
    -------
    data_above_thresh : ndarray
        column vector
    '''

    MAX_NUMBER_OF_SAMPLES = 2000
    MIN_NUMBER_OF_SAMPLES = 20
    DATA_THRESHOLD = 10

    data_above_thresh = data[data > DATA_THRESHOLD].dropna().values
    n_samples = len(data_above_thresh)
    if n_samples < MIN_NUMBER_OF_SAMPLES:
        return np.zeros((MAX_NUMBER_OF_SAMPLES, 1))
    elif n_samples > MAX_NUMBER_OF_SAMPLES:
        # Randomly subsample (we don't want to smoothly downsample
        # because that is likely to change the values)
        random_indices = np.random.randint(0, n_samples, MAX_NUMBER_OF_SAMPLES)
        resampled = data_above_thresh[random_indices]
        return resampled.reshape(MAX_NUMBER_OF_SAMPLES, 1)
    else:
        return data_above_thresh.reshape(n_samples, 1)

def _apply_clustering_n_clusters(X, n_clusters):
    """
    :param X: ndarray
    :param n_clusters: exact number of clusters to use
    :return:
    """
    from sklearn.cluster import KMeans
    k_means = KMeans(init='k-means++', n_clusters=n_clusters)
    k_means.fit(X)
    return k_means.labels_, k_means.cluster_centers_


def _apply_clustering(X, max_num_clusters, exact_num_clusters=None):
    '''
    Parameters
    ----------
    X : ndarray
    max_num_clusters : int

    Returns
    -------
    centroids : list of numbers
        List of power in different states of an appliance
    '''
    # If we import sklearn at the top of the file then it makes autodoc fail

    from sklearn import metrics

    # sklearn produces lots of DepreciationWarnings with PyTables
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Finds whether 2 or 3 gives better Silhouellete coefficient
    # Whichever is higher serves as the number of clusters for that
    # appliance
    num_clus = -1
    sh = -1
    k_means_labels = {}
    k_means_cluster_centers = {}
    k_means_labels_unique = {}

    # If the exact number of clusters are specified, then use that
    if exact_num_clusters is not None:
        labels, centers = _apply_clustering_n_clusters(X, exact_num_clusters)
        return centers.flatten()

    # Exact number of clusters are not specified, use the cluster validity measures
    # to find the optimal number
    for n_clusters in range(1, max_num_clusters):

        try:
            labels, centers = _apply_clustering_n_clusters(X, n_clusters)
            k_means_labels[n_clusters] = labels
            k_means_cluster_centers[n_clusters] = centers
            k_means_labels_unique[n_clusters] = np.unique(labels)
            try:
                sh_n = metrics.silhouette_score(
                    X, k_means_labels[n_clusters], metric='euclidean')

                if sh_n > sh:
                    sh = sh_n
                    num_clus = n_clusters
            except Exception:
                num_clus = n_clusters
        except Exception:
            if num_clus > -1:
                return k_means_cluster_centers[num_clus]
            else:
                return np.array([0])

    return k_means_cluster_centers[num_clus].flatten()

def find_nearest(known_array, test_array):
    """Find closest value in `known_array` for each element in `test_array`.

    Parameters
    ----------
    known_array : numpy array
        consisting of scalar values only; shape: (m, 1)
    test_array : numpy array
        consisting of scalar values only; shape: (n, 1)

    Returns
    -------
    indices : numpy array; shape: (n, 1)
        For each value in `test_array` finds the index of the closest value
        in `known_array`.
    residuals : numpy array; shape: (n, 1)
        For each value in `test_array` finds the difference from the closest
        value in `known_array`.
    """
    # from http://stackoverflow.com/a/20785149/732596

    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted]

    idx1 = np.searchsorted(known_array_sorted, test_array)
    idx2 = np.clip(idx1 - 1, 0, len(known_array_sorted)-1)
    idx3 = np.clip(idx1,     0, len(known_array_sorted)-1)

    diff1 = known_array_sorted[idx3] - test_array
    diff2 = test_array - known_array_sorted[idx2]

    indices = index_sorted[np.where(diff1 <= diff2, idx3, idx2)]
    residuals = test_array - known_array[indices]
    return indices, residuals


class CO():
    def __init__(self, debug=False):
        self.model = []
        self.state_combinations = None
        self.MIN_CHUNK_LENGTH = 100
        self.MODEL_NAME = 'CO'
        self.debug = debug
        if self.debug : print("[CO Initialised]")

    def train(self, df, appliance_list):
        """Train using 1D CO. Places the learnt model in the `model` attribute.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object
        num_states_dict : dict
        **load_kwargs : keyword arguments passed to `meter.power_series()`

        Notes
        -----
        * only uses first chunk for each meter (TODO: handle all chunks).
        """

        if self.model:
            raise RuntimeError(
                "This implementation of Combinatorial Optimisation"
                " does not support multiple calls to `train`.")

        max_num_clusters = 2

        for i, meter in enumerate(appliance_list):
            if self.debug : print(" [train] Training model for submeter", meter)
            meter_data = df[meter]
            
            num_on_states = None
            self.train_on_chunk(meter_data, meter, max_num_clusters, num_on_states)

    def train_on_chunk(self, chunk, meter, max_num_clusters, num_on_states):
        # Check if we've already trained on this meter
        meters_in_model = [d['training_metadata'] for d in self.model]
        if meter in meters_in_model:
            raise RuntimeError(
                "Meter {} is already in model!"
                "  Can't train twice on the same meter!"
                .format(meter))

        states = cluster(chunk, max_num_clusters, num_on_states)
        self.model.append({
            'states': states,
            'training_metadata': meter})
        print(" [train_on_chunk] Done training!")
        
    def disaggregate(self, mains, output_datastore,**load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : nilmtk.ElecMeter or nilmtk.MeterGroup
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        sample_period : number, optional
            The desired sample period in seconds.  Set to 60 by default.
        sections : TimeFrameGroup, optional
            Set to mains.good_sections() by default.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''
        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            # Check that chunk is sensible size
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue

            # Record metadata
            timeframes.append(chunk.timeframe)
            measurement = chunk.name

            appliance_powers = self.disaggregate_chunk(chunk)

            for i, model in enumerate(self.model):
                appliance_power = appliance_powers.iloc[:, i]
                if len(appliance_power) == 0:
                    continue
                data_is_available = True
                cols = pd.MultiIndex.from_tuples([chunk.name])
                meter_instance = model['training_metadata'].instance()
                df = pd.DataFrame(
                    appliance_power.values, index=appliance_power.index,
                    columns=cols)
                key = '{}/elec/meter{}'.format(building_path, meter_instance)
                output_datastore.append(key, df)

            # Copy mains data to disag output
            mains_df = pd.DataFrame(chunk, columns=cols)
            output_datastore.append(key=mains_data_location, value=mains_df)

        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[d['training_metadata'] for d in self.model]
            )
    
        
    def disaggregate_chunk(self, mains):
        """In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series


        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        """
        if not self.model:
            raise RuntimeError(
                "The model needs to be instantiated before"
                " calling `disaggregate`.  The model"
                " can be instantiated by running `train`.")

        if len(mains) < self.MIN_CHUNK_LENGTH:
            raise RuntimeError("Chunk is too short.")

        # sklearn produces lots of DepreciationWarnings with PyTables
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Because CombinatorialOptimisation could have been trained using
        # either train() or train_on_chunk(), we must
        # set state_combinations here.
        self._set_state_combinations_if_necessary()

        """
         # Add vampire power to the model
        if vampire_power is None:
            vampire_power = get_vampire_power(mains)
        if vampire_power > 0:
            print("Including vampire_power = {} watts to model..."
                  .format(vampire_power))
            n_rows = self.state_combinations.shape[0]
            vampire_power_array = np.zeros((n_rows, 1)) + vampire_power
            state_combinations = np.hstack(
                (self.state_combinations, vampire_power_array))
        else:
            state_combinations = self.state_combinations
        """


        state_combinations = self.state_combinations

        summed_power_of_each_combination = np.sum(state_combinations, axis=1)
        # summed_power_of_each_combination is now an array where each
        # value is the total power demand for each combination of states.

        # Start disaggregation
        indices_of_state_combinations, residual_power = find_nearest(
            summed_power_of_each_combination, mains.values)

        appliance_powers_dict = {}
        for i, model in enumerate(self.model):
            print("Estimating power demand for '{}'"
                  .format(model['training_metadata']))
            predicted_power = state_combinations[
                indices_of_state_combinations, i].flatten()
            column = pd.Series(predicted_power, index=mains.index, name=i)
            appliance_powers_dict[self.model[i]['training_metadata']] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict, dtype='float32')
        return appliance_powers        
    def _set_state_combinations_if_necessary(self):
        """Get centroids"""
        # If we import sklearn at the top of the file then auto doc fails.
        if (self.state_combinations is None or
                self.state_combinations.shape[1] != len(self.model)):
            from sklearn.utils.extmath import cartesian
            centroids = [model['states'] for model in self.model]
            self.state_combinations = cartesian(centroids)