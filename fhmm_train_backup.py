# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:17:30 2018

@author: Kit
"""
import numpy as np
from hmmlearn import hmm
from collections import OrderedDict
from six import iteritems
import itertools
from copy import deepcopy

debug = False
SEED = 42

# Fix the seed for repeatibility of experiments
np.random.seed(SEED)

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
                if debug : print("[FHMM][_apply_clustering] n_cluster = ", n_clusters, ", silhouette score =", sh_n)
    
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

def sort_startprob(mapping, startprob):
    """ Sort the startprob according to power means; as returned by mapping
    """
    num_elements = len(startprob)
    new_startprob = np.zeros(num_elements)
    for i in range(len(startprob)):
        new_startprob[i] = startprob[mapping[i]]
    return new_startprob


def sort_covars(mapping, covars):
    new_covars = np.zeros_like(covars)
    for i in range(len(covars)):
        new_covars[i] = covars[mapping[i]]
    return new_covars


def sort_transition_matrix(mapping, A):
    """Sorts the transition matrix according to increasing order of
    power means; as returned by mapping

    Parameters
    ----------
    mapping :
    A : numpy.array of shape (k, k)
        transition matrix
    """
    num_elements = len(A)
    A_new = np.zeros((num_elements, num_elements))
    for i in range(num_elements):
        for j in range(num_elements):
            A_new[i, j] = A[mapping[i], mapping[j]]
    return A_new


def sort_learnt_parameters(startprob, means, covars, transmat):
    mapping = return_sorting_mapping(means)
    means_new = np.sort(means, axis=0)
    startprob_new = sort_startprob(mapping, startprob)
    covars_new = sort_covars(mapping, covars)
    transmat_new = sort_transition_matrix(mapping, transmat)
    assert np.shape(means_new) == np.shape(means)
    assert np.shape(startprob_new) == np.shape(startprob)
    assert np.shape(transmat_new) == np.shape(transmat)

    return [startprob_new, means_new, covars_new, transmat_new]

def return_sorting_mapping(means):
    means_copy = deepcopy(means)
    means_copy = np.sort(means_copy, axis=0)

    # Finding mapping
    mapping = {}
    for i, val in enumerate(means_copy):
        mapping[i] = np.where(val == means)[0][0]
    return mapping

def compute_A_fhmm(list_A):
    """
    Parameters
    -----------
    list_pi : List of PI's of individual learnt HMMs

    Returns
    --------
    result : Combined Pi for the FHMM
    """
    result = list_A[0]
    for i in range(len(list_A) - 1):
        result = np.kron(result, list_A[i + 1])
    return result


def compute_means_fhmm(list_means):
    """
    Returns
    -------
    [mu, cov]
    """
    if True : print("[FHMM][compute_means_fhmm] list_means=", list_means, "\n*list_means=", *list_means)

    states_combination = list_means
    num_combinations = len(states_combination)
    means_stacked = np.array([sum(x) for x in states_combination])
    means = np.reshape(means_stacked, (num_combinations, 1))
    cov = np.tile(5 * np.identity(1), (num_combinations, 1, 1))
    return [means, cov]


def compute_pi_fhmm(list_pi):
    """
    Parameters
    -----------
    list_pi : List of PI's of individual learnt HMMs

    Returns
    -------
    result : Combined Pi for the FHMM
    """
    result = list_pi[0]
    for i in range(len(list_pi) - 1):
        result = np.kron(result, list_pi[i + 1])
    return result

def create_combined_hmm(model):

    list_pi = model.startprob_
    list_A = model.transmat_
    list_means = model.means_.flatten().tolist()
    if debug : print("[FHMM][create_combined_hmm] list_pi=", list_pi, "\nlist_A=", list_A, "\nlist_means=", list_means)

    pi_combined = compute_pi_fhmm(list_pi)
    A_combined = compute_A_fhmm(list_A)
    if debug : print("[FHMM][create_combined_hmm] pi_combined=", pi_combined, "\nA_combined=", A_combined, "\nlist_means=", list_means)

    [mean_combined, cov_combined] = compute_means_fhmm(list_means)

    combined_model = hmm.GaussianHMM(n_components=len(pi_combined), covariance_type='full')
    combined_model.startprob_ = pi_combined
    combined_model.transmat_ = A_combined
    combined_model.covars_ = cov_combined
    combined_model.means_ = mean_combined
    
    return combined_model

class FHMM():
    def __init__(self, debug=False):
        self.debug = debug
            
    def train(self, appliance_power_series):
        max_num_clusters = 3
        
    #    X = appliance_power_series.values.reshape((-1, 1))
        appliance_power_series = appliance_power_series.dropna()
        X = appliance_power_series.values.reshape((-1, 1))
            
        if not len(X):
            print("Submeter '{}' has no samples, skipping...", appliance_power_series.name)
            exit
                
        assert X.ndim == 2
        self.X = X
        
        states = cluster(appliance_power_series, max_num_clusters)
        num_total_states = len(states)
        if self.debug : print("[FHMM][train] num_total_states =", num_total_states, ", state =", states)
        
        if self.debug : print("[FHMM][train] Training model for", appliance_power_series.name)
        learnt_model = hmm.GaussianHMM(num_total_states, "full")

        # Fit
        learnt_model.fit(X)
        
        #self.meters = []
        new_learnt_models = OrderedDict()
        
        startprob, means, covars, transmat = sort_learnt_parameters(
                learnt_model.startprob_, learnt_model.means_,
                learnt_model.covars_, learnt_model.transmat_)
                
        new_learnt_models = hmm.GaussianHMM(startprob.size, "full")
        new_learnt_models.startprob_ = startprob
        new_learnt_models.transmat_ = transmat
        new_learnt_models.means_ = means
        new_learnt_models.covars_ = covars
        
        if self.debug : print("[FHMM][train] \nstart_prob_ =", startprob, "\ntransmat_ =", transmat, "\nmeans_=", means)

        # UGLY! But works.
        #self.meters.append(meter)
        
        learnt_model_combined = create_combined_hmm(new_learnt_models)
        self.individual = new_learnt_models
        self.model = learnt_model_combined        

    def disaggregate_chunk(self, test_mains):
        """Disaggregate the test data according to the model learnt previously

        Performs 1D FHMM disaggregation.

        For now assuming there is no missing data at this stage.
        """
        # See v0.1 code
        # for ideas of how to handle missing data in this code if needs be.

        # Array of learnt states
        learnt_states_array = []
        test_mains = test_mains.dropna()
        length = len(test_mains.index)
        temp = test_mains.values.reshape(length, 1)
        learnt_states_array.append(self.model.predict(temp))
        print(self.model.predict(temp))
        # Model
        means = OrderedDict()
'''
        for elec_meter, model in self.model.items():
            if self.debug : print("[FHMM][disaggregate_chunk] elec_meter =", elec_meter)

            means[elec_meter] = (
                model.means_.round().astype(int).flatten().tolist())
            means[elec_meter].sort()




        decoded_power_array = []
        decoded_states_array = []

        for learnt_states in learnt_states_array:
            [decoded_states, decoded_power] = decode_hmm(
                len(learnt_states), means, means.keys(), learnt_states)
            decoded_states_array.append(decoded_states)
            decoded_power_array.append(decoded_power)

        prediction = pd.DataFrame(
            decoded_power_array[0], index=test_mains.index)

        return prediction
'''     