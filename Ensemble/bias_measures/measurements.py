import pandas as pd
from Submission.Ensemble.bias_measures.measurements_utils import get_value_records, get_not_value_records, min_max_norm, \
    cm_value_n_value, tpr, tnr, lr, er, var_norm, treat, precision, mutual_information_mi, enp, acc, fpr, fnr


def demographic_parity(y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the demographic parity measurement for a
    single protected feature.
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    parities_per_value = []
    protected_values = protected[protected.columns[0]].unique()  # Get the protected feature possible values
    for value in protected_values:
        value_records, value_pred = get_value_records(value, protected, y_pred)
        value_positive = pd.DataFrame(value_pred.loc[value_pred[value_pred.columns[0]] == positive])
        value_positive_odds = float(value_positive.shape[0] / value_records.shape[0])
        not_value_records, not_value_pred = get_not_value_records(value, protected, y_pred)
        not_value_positive = pd.DataFrame(not_value_pred.loc[not_value_pred[not_value_pred.columns[0]] == positive])
        not_value_odds = float(not_value_positive.shape[0] / not_value_records.shape[0])
        parity = float(abs(not_value_odds - value_positive_odds))
        parities_per_value.append(parity)

    if len(set(parities_per_value)) <= 1 and parities_per_value[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        score = float(1-max(parities_per_value))
    return boolean_score, score


def disparate_impact(y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the disparate impact measurement.
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    disparate_per_value = []
    protected_values = protected[protected.columns[0]].unique()  # Get the protected feature possible values
    for value in protected_values:
        value_records, value_pred = get_value_records(value, protected, y_pred)
        value_positive = pd.DataFrame(value_pred.loc[value_pred[value_pred.columns[0]] == positive])
        value_odds = float(value_positive.shape[0] / value_records.shape[0])
        not_value_records, not_value_pred = get_not_value_records(value, protected, y_pred)
        not_value_positive = pd.DataFrame(not_value_pred.loc[not_value_pred[not_value_pred.columns[0]] == positive])
        not_value_odds = float(not_value_positive.shape[0] / not_value_records.shape[0])
        if value_odds != 0:
            disparate = float(not_value_odds / value_odds)
        else:
            disparate = 0
        disparate_per_value.append(disparate)

    if min(disparate_per_value) <= 0.8:
        boolean_score = False
        minimum = 0
        maximum = 0.8
        score = min_max_norm(minimum, maximum, disparate_per_value)
    else:
        boolean_score = True
        score = 1
    return boolean_score, score


def sensitivity(y_true, y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the sensitivity measurement.
    :param y_true: Dataframe with 1 column. The true labels (indexes are important)
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    sensitive_per_value = []
    protected_values = protected[protected.columns[0]].unique()  # Get the protected feature possible values
    possible_labels = [positive, -1]
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [-1 if x != positive else x for x in y_true[y_true.columns[0]]]
    y_pred[y_pred.columns[0]] = [1 if x == positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [1 if x == positive else x for x in y_true[y_true.columns[0]]]
    for value in protected_values:
        vtn, vfp, vfn, vtp, ntn, nfp, nfn, ntp = cm_value_n_value(value, protected, y_pred, y_true, [-1,1])
        sensitive = float(abs(tpr(ntn, nfp, nfn, ntp) - tpr(vtn, vfp, vfn, vtp)))
        sensitive_per_value.append(sensitive)

    if len(set(sensitive_per_value)) <= 1 and sensitive_per_value[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        score = float(1-max(sensitive_per_value))
    return boolean_score, score


def specificity(y_true, y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the sensitivity measurement.
    :param y_true: Dataframe with 1 column. The true labels (indexes are important)
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    specific_per_value = []
    protected_values = protected[protected.columns[0]].unique()  # Get the protected feature possible values
    possible_labels = [positive, -1]
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [-1 if x != positive else x for x in y_true[y_true.columns[0]]]
    y_pred[y_pred.columns[0]] = [1 if x == positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [1 if x == positive else x for x in y_true[y_true.columns[0]]]
    for value in protected_values:
        vtn, vfp, vfn, vtp, ntn, nfp, nfn, ntp = cm_value_n_value(value, protected, y_pred, y_true, [-1, 1])
        specific = float(abs(tnr(ntn, nfp, nfn, ntp) - tnr(vtn, vfp, vfn, vtp)))
        specific_per_value.append(specific)

    if len(set(specific_per_value)) <= 1 and specific_per_value[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        score = float(1-max(specific_per_value))
    return boolean_score, score


def likelihood_ratio(y_true, y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the sensitivity measurement.
    :param y_true: Dataframe with 1 column. The true labels (indexes are important)
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    likelihood_per_value = []
    protected_values = protected[protected.columns[0]].unique()  # Get the protected feature possible values
    possible_labels = [positive, -1]
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [-1 if x != positive else x for x in y_true[y_true.columns[0]]]
    y_pred[y_pred.columns[0]] = [1 if x == positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [1 if x == positive else x for x in y_true[y_true.columns[0]]]
    for value in protected_values:
        vtn, vfp, vfn, vtp, ntn, nfp, nfn, ntp = cm_value_n_value(value, protected, y_pred, y_true, [-1, 1])
        likelihood = float(abs(lr(ntn, nfp, nfn, ntp) - lr(vtn, vfp, vfn, vtp)))
        likelihood_per_value.append(likelihood)

    if len(set(likelihood_per_value)) <= 1 and likelihood_per_value[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        maximum = float(y_pred.shape[0] / 2)
        minimum = 0
        normalized = min_max_norm(minimum, maximum, [max(likelihood_per_value)])
        score = float(1 - normalized)
    return boolean_score, score


def balance_error_r(y_true, y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the sensitivity measurement.
    :param y_true: Dataframe with 1 column. The true labels (indexes are important)
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    ber_per_value = []
    protected_values = protected[protected.columns[0]].unique()  # Get the protected feature possible values
    possible_labels = [positive, -1]
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [-1 if x != positive else x for x in y_true[y_true.columns[0]]]
    y_pred[y_pred.columns[0]] = [1 if x == positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [1 if x == positive else x for x in y_true[y_true.columns[0]]]
    for value in protected_values:
        vtn, vfp, vfn, vtp, ntn, nfp, nfn, ntp = cm_value_n_value(value, protected, y_pred, y_true, [-1, 1])
        ber = float(abs(er(ntn, nfp, nfn, ntp) - er(vtn, vfp, vfn, vtp)))
        ber_per_value.append(ber)
    if len(set(ber_per_value)) <= 1 and ber_per_value[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        maximum = float(y_pred.shape[0] /2)
        minimum = 0
        normalized = min_max_norm(minimum, maximum, [max(ber_per_value)])
        score = float(1-normalized)
    return boolean_score, score


def equalized_odds(y_true, y_pred, protected, positive, possible_labels, **kwargs):
    """
    The method calculates the existence and a score for the equalized odds measurement.
    :param y_true: Dataframe with 1 column. The true labels (indexes are important)
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :param possible_labels: List. The possible labels (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    is_fulfill = []
    norm_var = []
    odds_per_label = []
    protected_values = protected[protected.columns[0]].unique()
    for label in possible_labels:
        for value in protected_values:
            value_records = pd.DataFrame(protected.loc[protected[protected.columns[0]] == value])
            value_true = pd.DataFrame(y_true.loc[value_records.index])
            value_true_for_label = pd.DataFrame(value_true.loc[value_true[value_true.columns[0]] == label])
            value_pred_for_label = pd.DataFrame(y_pred.loc[value_true_for_label.index])
            protected_label_positive = pd.DataFrame(
                value_pred_for_label.loc[value_pred_for_label[value_pred_for_label.columns[0]] == positive])
            if value_true_for_label.shape[0] != 0:
                odds = float(protected_label_positive.shape[0] / value_pred_for_label.shape[0])
            else:
                odds = 0
            odds_per_label.append(odds)
        if len(set(odds_per_label)) == 1:
            is_fulfill.append('True')
        else:
            is_fulfill.append('False')
        norm_var.append(var_norm(odds_per_label))
        odds_per_label = []
    if 'False' in is_fulfill:
        boolean_score = False
        score = min(norm_var)
    else:
        boolean_score = True
        score = 1
    return boolean_score, score


def equalized_opp(y_true, y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the equalized opportunity measurement.
    :param y_true: Dataframe with 1 column. The true labels (indexes are important)
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    odds_per_label = []
    protected_values = protected[protected.columns[0]].unique()
    for value in protected_values:
        value_records = pd.DataFrame(protected.loc[protected[protected.columns[0]] == value])
        value_true = pd.DataFrame(y_true.loc[value_records.index])
        value_true_for_label = pd.DataFrame(value_true.loc[value_true[value_true.columns[0]] == positive])
        value_pred_for_label = pd.DataFrame(y_pred.loc[value_true_for_label.index])
        protected_label_positive = pd.DataFrame(
            value_pred_for_label.loc[value_pred_for_label[value_pred_for_label.columns[0]] == positive])
        if value_true_for_label.shape[0] != 0:
            odds = float(protected_label_positive.shape[0] / value_pred_for_label.shape[0])
        else:
            odds = 0
        odds_per_label.append(odds)
    if len(set(odds_per_label)) == 1:
        boolean_score = 'True'
    else:
        boolean_score = 'False'
    score = var_norm(odds_per_label)
    return boolean_score, score


def treatment_equality(y_true, y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the treatment equality measurement.
    :param y_true: Dataframe with 1 column. The true labels (indexes are important)
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    treatment_per_value = []
    protected_values = protected[protected.columns[0]].unique()
    possible_labels = [positive, -1]
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [-1 if x != positive else x for x in y_true[y_true.columns[0]]]
    y_pred[y_pred.columns[0]] = [1 if x == positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [1 if x == positive else x for x in y_true[y_true.columns[0]]]
    for value in protected_values:
        vtn, vfp, vfn, vtp, ntn, nfp, nfn, ntp = cm_value_n_value(value, protected, y_pred, y_true, [-1,1])
        treatment = float(abs(treat(ntn, nfp, nfn, ntp) - treat(vtn, vfp, vfn, vtp)))
        treatment_per_value.append(treatment)
    if len(set(treatment_per_value)) <= 1 and treatment_per_value[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        # maximum = float(y_pred.shape[0] / 2)
        maximum = float(y_pred.shape[0])
        minimum = 0
        normilized = min_max_norm(minimum, maximum, [max(treatment_per_value)])
        score = float(1 - normilized)
    return boolean_score, score


def equal_positive_pred(y_true, y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the equal positive prediction value measurement.
    :param y_true: Dataframe with 1 column. The true labels (indexes are important)
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    precision_per_value = []
    protected_values = protected[protected.columns[0]].unique()
    possible_labels = [positive, -1]
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [-1 if x != positive else x for x in y_true[y_true.columns[0]]]
    y_pred[y_pred.columns[0]] = [1 if x == positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [1 if x == positive else x for x in y_true[y_true.columns[0]]]
    for value in protected_values:
        vtn, vfp, vfn, vtp, ntn, nfp, nfn, ntp = cm_value_n_value(value, protected, y_pred, y_true, [-1, 1])
        precisions = float(abs(precision(ntn, nfp, nfn, ntp) - precision(vtn, vfp, vfn, vtp)))
        precision_per_value.append(precisions)
    if len(set(precision_per_value)) <= 1 and precision_per_value[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        score = float(1-max(precision_per_value))
    return boolean_score, score


def normalized_diff(y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the normalized difference measurement.
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :param kwargs:
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
            measurement fulfillment (1 means no bias).
    """
    nd_list = []
    protected_values = protected[protected.columns[0]].unique()
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    p_positive = y_pred[y_pred[y_pred.columns[0]] == positive].size / y_pred.size
    p_not_positive = y_pred[y_pred[y_pred.columns[0]] != positive].size / y_pred.size
    for value in protected_values:
        value_records, value_pred = get_value_records(value, protected, y_pred)
        not_value_records, not_value_pred = get_not_value_records(value, protected, y_pred)
        p_value = value_records.size / protected.size
        p_not_value = not_value_records.size / protected.size
        p_value_positive = value_pred[value_pred[value_pred.columns[0]] == positive].size / value_pred.size
        p_not_value_positive = not_value_pred[not_value_pred[not_value_pred.columns[0]] == positive].size / not_value_pred.size
        top = p_not_value_positive - p_value_positive
        bottom = max((p_positive/p_not_value),(p_not_positive/p_value))
        nd = top/bottom
        nd_list.append(abs(nd))
    if len(set(nd_list)) <= 1 and nd_list[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        score = float(1 - max(nd_list))
    return boolean_score, score


def elift_ratio(y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the normalized difference measurement.
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :param kwargs:
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
            measurement fulfillment (1 means no bias).
    """
    er_list = []
    protected_values = protected[protected.columns[0]].unique()
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    p_positive = y_pred[y_pred[y_pred.columns[0]] == positive].size / y_pred.size
    for value in protected_values:
        value_records, value_pred = get_value_records(value, protected, y_pred)
        p_value_positive = value_pred[value_pred[value_pred.columns[0]] == positive].size / value_pred.size
        if p_positive == 0:
            er = 0
        else:
            er = p_value_positive / p_positive
        if er <= 1:
            er_list.append(er)
        else:
            er_list.append(1/er)
    if len(set(er_list)) <= 1 and er_list[0] == 1:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        score = float(min(er_list))
    return boolean_score, score


def odds_ratio(y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the normalized difference measurement.
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :param kwargs:
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
            measurement fulfillment (1 means no bias).
    """
    or_list = []
    protected_values = protected[protected.columns[0]].unique()
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    for value in protected_values:
        value_records, value_pred = get_value_records(value, protected, y_pred)
        not_value_records, not_value_pred = get_not_value_records(value, protected, y_pred)
        p_value_positive = value_pred[value_pred[value_pred.columns[0]] == positive].size / value_pred.size
        p_value_negative = value_pred[value_pred[value_pred.columns[0]] != positive].size / value_pred.size
        p_not_value_positive = not_value_pred[
                                   not_value_pred[not_value_pred.columns[0]] == positive].size / not_value_pred.size
        p_not_value_negative = not_value_pred[
                                   not_value_pred[not_value_pred.columns[0]] != positive].size / not_value_pred.size
        top = p_not_value_positive*p_value_negative
        bottom = p_not_value_negative*p_value_positive
        if bottom == 0:
            orv = 0
        else:
            orv = top / bottom
        if orv <= 1:
            or_list.append(orv)
        else:
            or_list.append(1 / orv)
    if len(set(or_list)) <= 1 and or_list[0] == 1:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        score = float(min(or_list))
    return boolean_score, score


def mutual_info(y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the normalized difference measurement.
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :param kwargs:
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
            measurement fulfillment (1 means no bias).
    """
    MI_list = mutual_information_mi(y_pred, protected, positive)
    score = 1 - max(MI_list)
    if score == 1:
        boolean_score = True
    else:
        boolean_score = False
    return boolean_score, score


def conditional_parity(y_pred, protected, positive, d_constraints, **kwargs):
    """
    The method calculates the existence and a score for the conditional demographic parity measurement for a
    single protected feature.
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :param d_constraints: Dict. The domain constraints to be considered in. Contains the column data and the value used
            in the format {'column': <the data>, 'value': <the value>}.
            For example {'column': Dataframe[0,1,2,1,6], 'value': 1}
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    parities_per_value = []
    protected_values = protected[protected.columns[0]].unique()  # Get the protected feature possible values
    cond_data = d_constraints.get('column').copy()
    cond_value = d_constraints.get('value')
    cond_data = pd.DataFrame(cond_data.loc[cond_data[cond_data.columns[0]] == cond_value])
    for value in protected_values:
        value_records, value_pred = get_value_records(value, protected, y_pred)
        value_pred_cond_idx = value_pred.index.intersection(cond_data.index)
        value_cond_pred = value_pred.loc[value_pred_cond_idx, :]
        value_positive = pd.DataFrame(value_cond_pred.loc[value_cond_pred[value_cond_pred.columns[0]] == positive])
        value_positive_odds = float(value_positive.shape[0] / value_records.shape[0])
        not_value_records, not_value_pred = get_not_value_records(value, protected, y_pred)
        not_value_pred_cond_idx = not_value_pred.index.intersection(cond_data.index)
        value_cond_pred = not_value_pred.loc[not_value_pred_cond_idx, :]
        not_value_positive = pd.DataFrame(value_cond_pred.loc[value_cond_pred[value_cond_pred.columns[0]] == positive])
        not_value_odds = float(not_value_positive.shape[0] / not_value_records.shape[0])
        parity = float(abs(not_value_odds - value_positive_odds))
        parities_per_value.append(parity)
    if len(set(parities_per_value)) <= 1 and parities_per_value[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        score = float(1 - max(parities_per_value))
    return boolean_score, score


def equal_negative_pred(y_true, y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the equal negative prediction value measurement.
    :param y_true: Dataframe with 1 column. The true labels (indexes are important)
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    protected_values = protected[protected.columns[0]].unique()
    neg_per_value = []
    possible_labels = [positive, -1]
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [-1 if x != positive else x for x in y_true[y_true.columns[0]]]
    y_pred[y_pred.columns[0]] = [1 if x == positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [1 if x == positive else x for x in y_true[y_true.columns[0]]]
    for value in protected_values:
        vtn, vfp, vfn, vtp, ntn, nfp, nfn, ntp = cm_value_n_value(value, protected, y_pred, y_true, [-1, 1])
        neg = float(abs(enp(ntn, nfp, nfn, ntp) - enp(vtn, vfp, vfn, vtp)))
        neg_per_value.append(neg)
    if len(set(neg_per_value)) <= 1 and neg_per_value[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        score = float(1-max(neg_per_value))
    return boolean_score, score


def accuracy(y_true, y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the accuracy value measurement.
    :param y_true: Dataframe with 1 column. The true labels (indexes are important)
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    protected_values = protected[protected.columns[0]].unique()
    acc_per_value = []
    possible_labels = [positive, -1]
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [-1 if x != positive else x for x in y_true[y_true.columns[0]]]
    y_pred[y_pred.columns[0]] = [1 if x == positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [1 if x == positive else x for x in y_true[y_true.columns[0]]]
    for value in protected_values:
        vtn, vfp, vfn, vtp, ntn, nfp, nfn, ntp = cm_value_n_value(value, protected, y_pred, y_true, [-1, 1])
        accurqacy = float(abs(acc(ntn, nfp, nfn, ntp) - acc(vtn, vfp, vfn, vtp)))
        acc_per_value.append(accurqacy)
    if len(set(acc_per_value)) <= 1 and acc_per_value[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        score = float(1-max(acc_per_value))
    return boolean_score, score


def calibration(y_pred, protected, positive, risk_scores, **kwargs):
    """
    The method calculates the existence and a score for the calibration measurement.
    :param y_pred:  with 1 column. The records prediction (indexes are important)
    :param protected: DataDataFrameFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :param risk_scores: DataDataFrameFrame with 1 column. The records risk score values (indexes are important)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    is_fulfill = []
    norm_var = []
    cal_per_risk = []
    protected_values = protected[protected.columns[0]].unique()
    for risk in risk_scores[risk_scores.columns[0]].unique():
        for value in protected_values:
            value_records, value_pred = get_value_records(value, protected, y_pred)
            value_pred_for_risk = pd.DataFrame(value_pred.loc[risk_scores[risk_scores.columns[0]] == risk])
            protected_risk_positive = pd.DataFrame(
                value_pred_for_risk.loc[value_pred_for_risk[value_pred_for_risk.columns[0]] == positive])
            if value_pred_for_risk.shape[0] != 0:
                odds = float(protected_risk_positive.shape[0] / value_pred_for_risk.shape[0])
            else:
                odds = 0
            cal_per_risk.append(odds)
        if len(set(cal_per_risk)) == 1:
            is_fulfill.append('True')
        else:
            is_fulfill.append('False')
        norm_var.append(var_norm(cal_per_risk))
    if 'False' in is_fulfill:
        boolean_score = False
        score = min(norm_var)
    else:
        boolean_score = True
        score = 1
    return boolean_score, score


def prediction_parity(y_pred, protected, positive, risk_scores, hr_threshold, **kwargs):
    """
    The method calculates the existence and a score for the calibration measurement.
    :param y_pred:  with 1 column. The records prediction (indexes are important)
    :param protected: DataDataFrameFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :param risk_scores: DataDataFrameFrame with 1 column. The records risk score values (indexes are important)
    :param hr_threshold: float. Risk score threshold. Need to be in the risk score values range.
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    cal_per_risk = []
    protected_values = protected[protected.columns[0]].unique()
    for value in protected_values:
        value_records, value_pred = get_value_records(value, protected, y_pred)
        value_pred_for_risk = pd.DataFrame(value_pred.loc[risk_scores[risk_scores.columns[0]] > hr_threshold])
        protected_risk_positive = pd.DataFrame(
            value_pred_for_risk.loc[value_pred_for_risk[value_pred_for_risk.columns[0]] == positive])
        if value_pred_for_risk.shape[0] != 0:
            odds = float(protected_risk_positive.shape[0] / value_pred_for_risk.shape[0])
        else:
            odds = 0
        cal_per_risk.append(odds)
    score = var_norm(cal_per_risk)
    if score != 1:
        boolean_score = False
    else:
        boolean_score = True
    return boolean_score, score


def erbs(y_pred, protected, positive, risk_scores, hr_threshold, **kwargs):
    """
    The method calculates the existence and a score for the Error rate balance with score (ERBS) measurement.
    :param y_pred:  with 1 column. The records prediction (indexes are important)
    :param protected: DataDataFrameFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :param risk_scores: DataFrame with 1 column. The records risk score values (indexes are important)
    :param hr_threshold: float. Risk score threshold. Need to be in the risk score values range.
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
    measurement fulfillment (1 means no bias).
    """
    odds_higher = []
    odds_lower = []
    protected_values = protected[protected.columns[0]].unique()
    for value in protected_values:
        value_records, value_pred = get_value_records(value, protected, y_pred)
        value_pred_not_positive = value_pred[value_pred[value_pred.columns[0]] != positive]
        value_pred_not_positive_higher = pd.DataFrame(value_pred_not_positive.loc[
                                                          risk_scores[risk_scores.columns[0]] > hr_threshold])
        if value_pred_not_positive.shape[0] != 0:
                odds = float(value_pred_not_positive_higher.shape[0] / value_pred_not_positive.shape[0])
        else:
            odds = 0
        odds_higher.append(odds)
        value_pred_positive = value_pred[value_pred[value_pred.columns[0]] == positive]
        value_pred_positive_lower = pd.DataFrame(value_pred_positive.loc[
                                                          risk_scores[risk_scores.columns[0]] <= hr_threshold])
        if value_pred_positive.shape[0] != 0:
            odds = float(value_pred_positive_lower.shape[0] / value_pred_positive.shape[0])
        else:
            odds = 0
        odds_lower.append(odds)
    score = min(var_norm(odds_lower), var_norm(odds_higher))
    if score == 1 :
        boolean_score = True
    else:
        boolean_score = False
    return boolean_score, score


def equal_fpr(y_true, y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the equal FPR value measurement.
    :param y_true: Dataframe with 1 column. The true labels (indexes are important)
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
        measurement fulfillment (1 means no bias).
    """
    protected_values = protected[protected.columns[0]].unique()
    fpr_per_value = []
    possible_labels = [positive, -1]
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [-1 if x != positive else x for x in y_true[y_true.columns[0]]]
    y_pred[y_pred.columns[0]] = [1 if x == positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [1 if x == positive else x for x in y_true[y_true.columns[0]]]
    for value in protected_values:
        vtn, vfp, vfn, vtp, ntn, nfp, nfn, ntp = cm_value_n_value(value, protected, y_pred, y_true, [-1, 1])
        fpr_value = float(abs(fpr(ntn, nfp, nfn, ntp) - fpr(vtn, vfp, vfn, vtp)))
        fpr_per_value.append(fpr_value)
    if len(set(fpr_per_value)) <= 1 and fpr_per_value[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        score = float(1 - max(fpr_per_value))
    return boolean_score, score


def equal_fnr(y_true, y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the equal FNR value measurement.
    :param y_true: Dataframe with 1 column. The true labels (indexes are important)
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
        measurement fulfillment (1 means no bias).
    """
    protected_values = protected[protected.columns[0]].unique()
    fpr_per_value = []
    possible_labels = [positive, -1]
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [-1 if x != positive else x for x in y_true[y_true.columns[0]]]
    y_pred[y_pred.columns[0]] = [1 if x == positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [1 if x == positive else x for x in y_true[y_true.columns[0]]]
    for value in protected_values:
        vtn, vfp, vfn, vtp, ntn, nfp, nfn, ntp = cm_value_n_value(value, protected, y_pred, y_true, [-1, 1])
        fpr_value = float(abs(fnr(ntn, nfp, nfn, ntp) - fnr(vtn, vfp, vfn, vtp)))
        fpr_per_value.append(fpr_value)
    if len(set(fpr_per_value)) <= 1 and fpr_per_value[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        score = float(1 - max(fpr_per_value))
    return boolean_score, score


def erb(y_true, y_pred, protected, positive, **kwargs):
    """
    The method calculates the existence and a score for the ERB measurement.
    :param y_true: Dataframe with 1 column. The true labels (indexes are important)
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :param kwargs:
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
        measurement fulfillment (1 means no bias).
    """
    boolean_score_fpr, score_fpr = equal_fpr(y_true, y_pred, protected, positive)
    boolean_score_fnr, score_fnr = equal_fnr(y_true, y_pred, protected, positive)
    score = min(score_fpr,score_fnr)
    if score == 1:
        boolean_score = True
    else:
        boolean_score = False
    return boolean_score, score


def conditional_accuracy(y_true, y_pred, protected, positive, **kwargs):
    """
        The method calculates the existence and a score for the ERB measurement.
        :param y_true: Dataframe with 1 column. The true labels (indexes are important)
        :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
        :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
        :param positive: the positive label (if str in the data then need to be str)
        :param kwargs:
        :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
            measurement fulfillment (1 means no bias).
        """
    protected_values = protected[protected.columns[0]].unique()
    td_per_value = []
    possible_labels = [positive, -1]
    y_pred[y_pred.columns[0]] = [-1 if x != positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [-1 if x != positive else x for x in y_true[y_true.columns[0]]]
    y_pred[y_pred.columns[0]] = [1 if x == positive else x for x in y_pred[y_pred.columns[0]]]
    y_true[y_true.columns[0]] = [1 if x == positive else x for x in y_true[y_true.columns[0]]]
    for value in protected_values:
        vtn, vfp, vfn, vtp, ntn, nfp, nfn, ntp = cm_value_n_value(value, protected, y_pred, y_true, [-1, 1])
        tpd_value = float(abs(ntp - vtp)/protected.size)
        tnd_value = float(abs(ntn - vtn)/protected.size)
        td_per_value.append(tpd_value)
        td_per_value.append(tnd_value)
    if len(set(td_per_value)) <= 1 and td_per_value[0] == 0:
        boolean_score = True
        score = 1
    else:
        boolean_score = False
        score = float(1 - max(td_per_value))
    return boolean_score, score


def balance_residuals(y_true, y_pred, protected, positive, **kwargs):
    """
        The method calculates the existence and a score for the balance residuals measurement.
        :param y_true: Dataframe with 1 column. The true labels (indexes are important)
        :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
        :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
        :param positive: the positive label (if str in the data then need to be str)
        :param kwargs:
        :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
            measurement fulfillment (1 means no bias).
        """
    br_list = []
    protected_values = protected[protected.columns[0]].unique()
    y_pred[y_pred.columns[0]] = [0 if x != positive else 1 for x in y_pred[y_pred.columns[0]]]
    for value in protected_values:
        value_records, value_pred = get_value_records(value, protected, y_pred)
        not_value_records, not_value_pred = get_not_value_records(value, protected, y_pred)
        dif_value = [abs(y_true[y_true.columns[0]][x] - value_pred[value_pred.columns[0]][x]) for x in
                     list(value_pred.index)]
        dif_not_value = [abs(y_true[y_true.columns[0]][x] - not_value_pred[not_value_pred.columns[0]][x]) for x in
                         list(not_value_pred.index)]
        value_br = sum(dif_value) / value_pred.size
        not_value_br = sum(dif_not_value) / not_value_pred.size
        br_list.append(value_br-not_value_br)
    score = 1 - max(br_list)
    if score == 1:
        boolean_score = True
    else:
        boolean_score = False
    return boolean_score, score
