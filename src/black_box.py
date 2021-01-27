from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import sobol_seq
import numpy as np
import pandas as pd



def RF_experiment(X, y, RF_kwargs):
    
    # Build sklearn model
    clf = RandomForestClassifier(**RF_kwargs)
    
    # evaluate classifier with cross_validation
    scores = cross_validate(clf, X, y,
                            scoring=["f1", "roc_auc"],
                            cv=3,
                            n_jobs=3)

    # mean scores for all folds
    for k, v in scores.items():
        scores[k] = np.mean(v)

    return scores


def random_sample(sp_json, n=1):
    """ Random configuration sampling

    Args:
        sp_json (dict): parameters describtion
        n (int, optional): number of configurations to generate. Defaults to 1.

    Returns:
        pandas.DataFrame: configurations samples
    """    
    conf = []
    for param in sp_json:
        p_type = param['type']
        if p_type == 'category':
            val = np.random.choice(param['categories'], n)
            conf.append(val)
        if p_type == 'float':
            low = param['bounds'][0]
            high = param['bounds'][1]
            # val = sobol_seq.i4_sobol_generate(1, n).flatten()*(high-low) + low
            val = np.random.rand(n, 1).flatten()*(high-low) + low
            conf.append(val)
        if p_type == 'integer':
            low = param['bounds'][0]
            high = param['bounds'][1]
            val = np.random.choice(range(low, high+1), n)
            conf.append(val)

    return pd.DataFrame(list(zip(*conf)), columns=[p['name'] for p in sp_json])


# https://ascpt.onlinelibrary.wiley.com/doi/pdf/10.1002/psp4.6
def psd_sample(sp_json, n=1, kind='sobol'):
    """ Pseudo-random configuration sampling

    Args:
        sp_json (dict): parameters describtion
        n (int, optional): number of configurations to generate. Defaults to 1.
        kind (str, optional): identifying the general kind of pseudo-random samples generator. Sobol('sobol') or Latin Hypercube sampling('lh'). Defaults to 'sobol'.

    Raises:
        ValueError: wronf value for pseudo-samples generation

    Returns:
        pandas.DataFrame: configurations samples
    """    
    dim = len(sp_json)
    if kind == 'sobol':
        squere = sobol_seq.i4_sobol_generate(dim, n)
    elif kind == 'lh':
        squere = np.random.uniform(size=[n, dim])
        for i in range(0, dim):
            squere[:, i] = (np.argsort(squere[:, i])+0.5)/n
    else: 
        raise ValueError(
            "`kind` property could be `sobol` or `lh`. Received: ", kind)

    conf = []
    for desc, sob_vec in zip(sp_json, squere.T):
        p_type = desc['type']
        if p_type == 'category':
            idx = np.round(sob_vec*(len(desc['categories'])-1)).astype(int)
            val = [desc['categories'][i] for i in idx]
            conf.append(val)
        if p_type == 'float':
            low = desc['bounds'][0]
            high = desc['bounds'][1]
            val = low + sob_vec*(high-low)
            conf.append(val)
        if p_type == 'integer':
            low = desc['bounds'][0]
            high = desc['bounds'][1]
            val = (low + sob_vec*(high-low)).astype(int)
            conf.append(val)

    return pd.DataFrame(list(zip(*conf)), columns=[p['name'] for p in sp_json])
