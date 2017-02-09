from __future__ import division
import numpy as np

# Define helper function from stackoverflow:
# http://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ Very close to np.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: np.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with np.percentile.
    :return: np.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


# Should now have all the quantile-values for each agen group. 
# Now just need to calculate the functions of quantiles:
def goh(q):
    # take a dict of quantile results, where keys are the strings:
    # "lo", "med_lo", "medium", "med_hi", "hi"
    # ...and values are the data at those percentiles
    
    # Assert that correct length:
    assert len(q) == 5
    
    # Assert that the keys are correct:
    set1 = set(q.keys())
    set2 = set(["lo", "med_lo", "medium", "med_hi", "hi"])
    assert set1 == set2, "set1: "+str(set1)+" set2: "+str(set2)
    
    # Run the formulas
    return [(q["hi"] - q["lo"]) / (q["med_hi"] - q["med_lo"]), 
            (q["hi"] - 2*q["medium"] + q["lo"]) / (q["hi"] - q["lo"]),
             q["med_hi"] - q["med_lo"],
             q["medium"]]


# Define quantile names:
# Let's use the weighted_quantile code to create measures of 
# the quantiles for particular ... quantiles
quantiles = [5, 25, 50, 75, 95]
quantiles.sort() # Just in case haven't yet

quantile_names = ["lo", "med_lo", "medium", "med_hi", "hi"]

assert len(quantiles) == len(quantile_names)

float_quantiles = [q / 100.0 for q in quantiles]

