from attack import Attack
import pandas as pd
import numpy as np
import json


def prototype_critic_differentiator(o_model, testset_x, testset_y, trainset_x, trainset_y, scaler, critic_threshold,
                                    prototype_threshold, cat, results_dict, CONSTRAINTS=[], protected_place=-1):
    """

    :param o_model:The model to be assessed. Support the functionality model.predict(x)
    :param testset_x:DataFrame. The test dataset.
    :param testset_y: Series. The test labels.
    :param trainset_x: DataFrame. The train dataset.
    :param trainset_y: Series. The train labels.
    :param scaler: Scaler. Sklearn scaler that can change the data records to the representation that fed to o_model.
    :param critic_threshold: int. The highest number of iterations to determine if the record is a critic.
    :param prototype_threshold: int. The lowest number of iterations to determine if the record is a prototype.
    :param results_dict: Dict. Result dictionary.
    :param CONSTRAINTS: list. Default is empty. the data constrains.
    :return: results_dict: Dict. The updated result dictionary.
                data_with_proto_column_for_class: List[DataFrame]. Dataframe for each class with its
                prototypes/critics/non records.
    """
    data_with_proto_column_for_class = []
    x_adv_list = []
    classes = list(testset_y.value_counts().index.values)
    for curr_class in classes:
        iter_num_list = []
        proto_critic_list = []
        class_records_idx = np.where(testset_y == curr_class)[0]
        class_records = testset_x.loc[class_records_idx]
        proto_critic_list, iter_num_list, x_adv = cat.cat_wrapper(class_records, o_model, critic_threshold, prototype_threshold, trainset_x, protected_place)
        class_df = class_records.copy()
        class_df["iteration_num"] = iter_num_list
        class_df["prototype_or_critic"] = proto_critic_list
        data_with_proto_column_for_class.append(class_df)
        x_adv_list.append(x_adv)
    return results_dict, data_with_proto_column_for_class, x_adv_list


def get_value_records(value, data, y_pred):
    value_records = pd.DataFrame(data.loc[data[data.columns[0]] == value])
    value_pred = pd.DataFrame(y_pred.loc[value_records.index])
    return value_records, value_pred


def get_not_value_records(value, data, y_pred):
    value_records = pd.DataFrame(data.loc[data[data.columns[0]] != value])
    value_pred = pd.DataFrame(y_pred.loc[value_records.index])
    return value_records, value_pred


def odds_ratio(y_pred, protected, positive):
    """
    The method calculates the existence and a score for the normalized difference measurement.
    :param y_pred: DataFrame with 1 column. The records prediction (indexes are important)
    :param protected: DataFrame with 1 column. The records protected feature values (indexes are important)
    :param positive: the positive label (if str in the data then need to be str)
    :return: boolean score for the measurement fulfillment criteria and a numeric score between 0-1 for the
            measurement fulfillment (1 means no bias).
    """
    or_list = []
    protected_values = protected[protected.columns[0]].unique()
    y_pred[y_pred.columns[0]] = [-1 if x != positive else 1 for x in y_pred[y_pred.columns[0]]]
    positive = 1
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


def fairness_assessment_calculator(data_with_proto_column_for_class, protected_name, results_dict):
    score = 1.0
    scores = []
    for p_class in data_with_proto_column_for_class:
        y_pred = pd.DataFrame(p_class[p_class["prototype_or_critic"] != "non"]["prototype_or_critic"].values, columns=["prot_crit"]).reset_index(drop=True)
        protected = pd.DataFrame(p_class[p_class["prototype_or_critic"] != "non"][protected_name].values, columns=["protected"]).reset_index(drop=True)
        if protected_name == "Age":
            protected["protected"] = protected["protected"].apply(lambda x: x <= protected["protected"].median())
        boolean_score, score_odds = odds_ratio(y_pred, protected, "prototype")
        scores.append(score_odds)
    if results_dict.get("model_fairness_assessment_by_protected_by_class") is None:
        results_dict.update({"model_fairness_assessment_by_protected_by_class":{protected_name:scores}})
    else:
        results_dict.get("model_fairness_assessment_by_protected_by_class").update({protected_name:scores})
    score = min(scores)
    return results_dict, score


def fairness_estimation_using_prototypes(o_model, testset_x, testset_y, trainset_x, trainset_y, protected_name, scaler,
                                        critic_threshold, prototype_threshold, cat, results_dict, protected_place=-1):
    """
    The method perform fairness assessment on a given model and a protected feature.
    :param o_model: The model to be assessed. Support the functionality model.predict(x)
    :param testset_x:DataFrame. The test dataset.
    :param testset_y: Series. The test labels.
    :param trainset_x: DataFrame. The train dataset.
    :param trainset_y: Series. The train labels.
    :param protected_name: str. The name of the protected feature that is examined.
    :param scaler: Scaler. Sklearn scaler that can change the data records to the representation that fed to o_model.
    :param critic_threshold: int. The highest number of iterations to determine if the record is a critic.
    :param prototype_threshold: int. The lowest number of iterations to determine if the record is a prototype.
    :param results_dict: Dict. Result dictionary.
    :return: results_dict: Dict. The updated result dictionary.
                score: float. The fairness assessment of the given model and protected feature.
    """
    results_dict, data_with_proto_column_for_class, x_adv_df = prototype_critic_differentiator(o_model, testset_x, testset_y,
                                                                                     trainset_x, trainset_y, scaler,
                                                                                     critic_threshold,
                                                                                     prototype_threshold, cat, results_dict)
    results_dict, score = fairness_assessment_calculator(data_with_proto_column_for_class, protected_name, results_dict)
    # Update result dict
    if results_dict.get("model_fairness_assessment_by_protected") is None:
        results_dict.update({"model_fairness_assessment_by_protected":{protected_name:score}})
    else:
        results_dict.get("model_fairness_assessment_by_protected").update({protected_name:score})
    if results_dict.get("model_prototypes_by_protected") is None:
        results_dict.update({"model_prototypes_by_protected": {protected_name: data_with_proto_column_for_class}})
    else:
        results_dict.get("model_prototypes_by_protected").update({protected_name: data_with_proto_column_for_class})
    if results_dict.get("model_adv_by_protected") is None:
        results_dict.update({"model_adv_by_protected": {protected_name: x_adv_df}})
    else:
        results_dict.get("model_adv_by_protected").update({protected_name: x_adv_df})
    return results_dict, score


def fairness_estimation(o_model, protected_features, testset_x, testset_y, trainset_x, trainset_y, scaler,
                        critic_threshold, prototype_threshold, saving_path, CONSTRAINTS, b_seed,
                        protected_consideration=False, gb=None, model=None, exp=None):
    """
    The method perform fairness assessment on a given model.
    :param o_model: The model to be assessed. Support the functionality model.predict(x)
    :param protected_features: list[str]. List of the protected features columns names.
    :param testset_x: DataFrame. The test dataset.
    :param testset_y: Series. The test labels.
    :param trainset_x: DataFrame. The train dataset.
    :param trainset_y: Series. The train labels.
    :param scaler: Scaler. Sklearn scaler that can change the data records to the representation that fed to o_model.
    :param critic_threshold: int. The highest number of iterations to determine if the record is a critic.
    :param prptotype_threshold: int. The lowest number of iterations to determine if the record is a prototype.
    :return: Dict. Result dictionary.
    """
    scores = []
    results_dict = dict()
    cat = Attack(trainset_x, o_model.predict(trainset_x), trainset_x, o_model.predict(trainset_x), o_model, CONSTRAINTS,
                 scaler, gb=gb, model=model)
    if not protected_consideration:
        results_dict, data_with_proto_column_for_class, x_adv_df = prototype_critic_differentiator(o_model, testset_x,
                                                                                                   pd.Series(o_model.predict(testset_x)),
                                                                                                   trainset_x, trainset_y,
                                                                                                   scaler,
                                                                                                   critic_threshold,
                                                                                                   prototype_threshold, cat,
                                                                                                   results_dict)
        if results_dict.get("model_prototypes") is None:
            results_dict.update({"model_prototypes": data_with_proto_column_for_class})
        else:
            results_dict.update({"model_prototypes": data_with_proto_column_for_class})
        if results_dict.get("model_adv") is None:
            results_dict.update({"model_adv": x_adv_df})
        else:
            results_dict.update({"model_adv": x_adv_df})

        for prot in protected_features:
            results_dict, score = fairness_assessment_calculator(data_with_proto_column_for_class, prot, results_dict)
            scores.append(score)
            # Update result dict
            if results_dict.get("model_fairness_assessment_by_protected") is None:
                results_dict.update({"model_fairness_assessment_by_protected": {prot: score}})
            else:
                results_dict.get("model_fairness_assessment_by_protected").update({prot: score})

    else:
        for prot in protected_features:
            protected_place =testset_x.columns.get_loc(prot)
            results_dict, data_with_proto_column_for_class, x_adv_df = prototype_critic_differentiator(o_model,
                                                                                                       testset_x,
                                                                                                       pd.Series(
                                                                                                           o_model.predict(
                                                                                                               testset_x)),
                                                                                                       trainset_x,
                                                                                                       trainset_y,
                                                                                                       scaler,
                                                                                                       critic_threshold,
                                                                                                       prototype_threshold,
                                                                                                       cat,
                                                                                                       results_dict,
                                                                                                       protected_place=protected_place)
            if results_dict.get("model_prototypes_{}".format(prot)) is None:
                results_dict.update({"model_prototypes_{}".format(prot): data_with_proto_column_for_class})
            else:
                results_dict.update({"model_prototypes_{}".format(prot): data_with_proto_column_for_class})
            if results_dict.get("model_adv_{}".format(prot)) is None:
                results_dict.update({"model_adv_{}".format(prot): x_adv_df})
            else:
                results_dict.update({"model_adv_{}".format(prot): x_adv_df})

            results_dict, score = fairness_assessment_calculator(data_with_proto_column_for_class, prot, results_dict)
            scores.append(score)
            # Update result dict
            if results_dict.get("model_fairness_assessment_by_protected") is None:
                results_dict.update({"model_fairness_assessment_by_protected": {prot: score}})
            else:
                results_dict.get("model_fairness_assessment_by_protected").update({prot: score})

    print("final_model_fairness_assessment_score: ", min(scores), " from feature: ",
          protected_features[scores.index(min(scores))])
    results_dict_final = {}
    results_dict_final.update({"final_model_fairness_assessment_score": min(scores),
                               "final_model_fairness_assessment_protected": protected_features[
                                   scores.index(min(scores))],
                               "model_fairness_assessment_score_all_scores": scores})
    results_dict.update({"final_model_fairness_assessment_score": min(scores),
                         "final_model_fairness_assessment_protected": protected_features[scores.index(min(scores))],
                         "model_fairness_assessment_score_all_scores": scores})
    save_all_outputs(saving_path, results_dict, results_dict_final, b_seed, protected_features, exp=exp)
    return results_dict


def save_all_outputs(saving_path, results_dict, results_dict_final, b_seed, protected, exp=00):
    fp = open('{}{}_{}_results_dict_final.json'.format(saving_path, exp, b_seed), 'w')
    json.dump(results_dict_final, fp)
    fp.close()

    if "model_prototypes" in results_dict:
        model_prototypes = results_dict.get("model_prototypes")
        class_n = 0
        for label_class in model_prototypes:
            label_class.to_csv("{}{}_{}_model_prototypes_class_{}.csv".format(saving_path, exp, b_seed, class_n),
                               index=False)
            class_n = class_n + 1
        class_n = 0
        model_adv = results_dict.get("model_adv")
        for label_class in model_adv:
            label_class.to_csv("{}{}_{}_model_adv_class_{}.csv".format(saving_path, exp, b_seed, class_n),
                               index=False)
            class_n = class_n + 1
    else:
        for prot in protected:
            model_prototypes = results_dict.get("model_prototypes_{}".format(prot))
            class_n = 0
            for label_class in model_prototypes:
                label_class.to_csv("{}{}_model_prototypes_class_{}_prot_{}.csv".format(saving_path, b_seed, class_n, prot),
                                   index=False)
                class_n = class_n + 1
            class_n = 0
            model_adv = results_dict.get("model_adv_{}".format(prot))
            for label_class in model_adv:
                label_class.to_csv("{}{}_{}_model_adv_class_{}_prot_{}.csv".format(saving_path, exp, b_seed, class_n, prot),
                                   index=False)
                class_n = class_n + 1
