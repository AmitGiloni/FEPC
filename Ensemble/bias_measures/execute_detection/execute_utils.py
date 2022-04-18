import pandas as pd
import numpy as np
import os


def get_possible_measurements(labels, score_column, d_constraints):
    measurements = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'measurement_mapping.csv'))
    if labels is None:
        measurements = measurements[measurements['Labels'] != 'V']
    if score_column is None:
        measurements = measurements[measurements['Score_column'] != 'V']
    if d_constraints is None:
        measurements = measurements[measurements['d_constraints'] != 'V']
    return list(measurements['Name'])


def display_measurements(measurements):
    temp_dfs = []
    for prot in measurements.keys():
        prot_measurements = measurements.get(prot)
        temp_df = pd.DataFrame(prot_measurements, columns=[prot], index=['Is the model bias clear?', 'Bias Score'])
        temp_dfs.append(temp_df)
    results = pd.concat(temp_dfs, axis=1)
    return results


def get_pred(data, model):
    try:
        return pd.DataFrame(data=np.asarray(model.predict(np.asarray(data))).reshape(-1, 1), index=data.index,
                            columns=['pred'])
    except:
        return pd.DataFrame(data=np.asarray(model.predict(data)).reshape(-1, 1), index=data.index,
                            columns=['pred'])


def reverse_onehot(data, pf):
    revers = pd.DataFrame()
    data[data.astype(float) <= 0] = 0
    data[data.astype(float) > 0] = 1
    tempdf = data.loc[:, :].replace(1, pd.Series(data.columns, data.columns))
    revers[pf] = (tempdf != 0).idxmax(1)
    return revers
