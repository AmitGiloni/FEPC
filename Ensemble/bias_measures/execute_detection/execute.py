from scipy import stats

from Submission.Ensemble.bias_measures import measurements
# from bias.Original.Detection.execute_detection.execute_utils import get_possible_measurements, display_measurements, get_pred, \
#     reverse_onehot
import pandas as pd
import numpy as np
import sklearn

from Submission.Ensemble.bias_measures.execute_detection.execute_utils import get_possible_measurements, get_pred, \
    reverse_onehot, display_measurements


def measurements_wrapper(data=None, data_encoding='None', labels=None, possible_labels=None,
                         pf_name='Protected', pf_columns=None, pf_columns_map=None, model=None,
                         score_column=None, positive_classes=None, hr_threshold=None, d_constraints=None, prot_treshold_cut=None):
    """
    The function lists all the input variables which are needed in order to perform bias detection (based on statistical
    measurements inequality).
    :param data: Required, type: ndarray/dataframe. The inputs records. Note that this input must have enough records
    to allow possible representation for every examined protected feature value. Note that this data can be training
    data or test set data. However, in order to fully estimate the bias it is recommended to examine the training data.
    If the test data is used, than it must correctly represent the data distribution which is used for inducing the ML
    model.
    :param data_encoding: Required, type: string. The data encoding type: ‘onehot’ for one-hot encoding, ‘label’ for
    label encoding, ‘none’ is the data is not encoded.
    :param labels: Optional, type: ndarray/dataframe. The data true labels. If None, only the unsupervised measurements
    will apply.
    :param possible_labels: Optional, list of objects (the object type should be according to the “labels”
    types: string/int/float). List of possible classes. The labels may not include all the possibilities. In this case,
    the bias is examined only according to the supplied classes.
    :param pf_name: Required, type: string. The name of the protected feature.
    :param pf_columns: Required, type: list of int/string. The protected feature columns names. If the data_encoding is
    ‘onehot’, pf_columns will contain the names of the columns that represent all the protected feature values. If the
    data_encoding is ‘label’ or ‘none’, pf_columns will contain a list with one column name. If data is a ndarray,
    pd_columns values will be the protected feature columns numbers.
    :param pf_columns_map: Optional, type: list of string. Mapping of the protected feature column index to its name.
    If not empty, the pf_columns_map values will replace the pf_columns values in the mid-process outputs.
    :param model: Required, type: object. The examined model. The model object will have a predict function that when
    called with data returns the posterior probabilities of each class for each record.
    :param score_column: Optional, type: int/string. Some measurements require risk score. The risk score column name.
    If data is a ndarray, score_column value will be the score column number. If data is a dataframe, score_column value
    will be the score column name. If None, only the suitable measurements will apply.
    :param positive_classes: Optional, type: list of string/int. The options for the positive classes. If None, the
    algorithm will consider all the possible classes as options for the positive class.
    :param hr_threshold: Optional, type: float. Some measurements require risk score threshold. Need to be in the risk
    score range. If None, the hr_threshold will be set to the risk score average.
    :param d_constraints: Optional, Dict. The domain constraints to be considered in. Contains the column data and the value used
            in the format {'column': <the data>, 'value': <the value>}. For example {'column': Dataframe[0,1,2,1,6], 'value': 1}.
            If None, only the suitable measurements will apply.
    :param prot_treshold_cut: Optional, type: float. If the protected feature is numeric, the prot_treshold_cut value
    will use to categorize the protected feature for the examination.
    :return: dist of elements for the report.
    """
    # in this implementation the model gets ndarray

    if isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        measurements_list = get_possible_measurements(labels, score_column, d_constraints)

        # create the variables for measurement applying
        risk_scores = []
        y_pred = get_pred(data, model)
        if prot_treshold_cut is None:
            protected = data[pf_columns].dropna()
            protected = protected.astype(str)
        else:
            protected = data[pf_columns].dropna().apply(lambda x: x <= prot_treshold_cut)
        if data_encoding == 'onehot':
            protected = reverse_onehot(protected, pf_name)
        if possible_labels is None and labels is not None:
            possible_labels = list(labels[labels.columns[0]].unique())
        if score_column is not None:
            risk_scores = pd.DataFrame(data[score_column], columns=[score_column])
            if hr_threshold is None:
                hr_threshold = data[score_column].mean()
        results_dict = {}
        dfs_results = {}
        df_results = pd.DataFrame()
        ##################################
        print_fisher_for_odds_ratio(y_pred=y_pred.copy(), protected=protected.copy())
        ##################################
        for positive in positive_classes:
            for measure_name in measurements_list:
                measure = getattr(measurements, measure_name.lower())
                if labels is not None:
                    m_labels = labels.copy()
                else:
                    m_labels = None
                if isinstance(risk_scores,list):
                    m_risk_scores = None
                else:
                    m_risk_scores = risk_scores.copy()
                boolean_score, score = measure(y_pred=y_pred.copy(), protected=protected.copy(), positive=positive,
                                            risk_scores=m_risk_scores,
                                            hr_threshold=hr_threshold, y_true=m_labels, possible_labels=possible_labels,
                                            d_constraints=d_constraints)
                results_dict.update({measure_name: [boolean_score, score]})
            df_results = display_measurements(results_dict)
            dfs_results.update({positive:df_results})

        if len(positive_classes) > 1:
            for positive in dfs_results.keys():
                for col in df_results:
                    df_results[col]['Bias Score'] = min(df_results[col]['Bias Score'], dfs_results.get(positive)[col]['Bias Score'])
                    if df_results[col]['Bias Score'] is True and df_results[col]['Bias Score'] != 1:
                        df_results[col]['Is the model bias clear'] = False
        return df_results
    else:
        return "DataTypeError: The data parameter should be pandas.DataFrame or numpy.ndarray."


def print_fisher_for_odds_ratio(y_pred, protected):
    p_values = np.unique(protected.values).tolist()
    y_values = np.unique(y_pred.values).tolist()
    cm_values = []
    if len(p_values)<=2 and len(y_values)<=2:
        for p in p_values:
            v_list = []
            for y in y_values:
                value_records = protected[protected[protected.columns[0]] == p]
                value_pred = pd.DataFrame(y_pred.loc[value_records.index])
                pred_records = value_pred[value_pred[value_pred.columns[0]] == y]
                v_list.append(len(pred_records))
            cm_values.append(v_list)
        oddsratio, pvalue = stats.fisher_exact(cm_values)
        print("Feature: ",protected.columns[0]," Fisher p-value: ",pvalue)