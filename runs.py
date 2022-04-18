import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict
from sklearn.model_selection import train_test_split

from attack import categorical, binary
from bias_estimation import fairness_estimation, fairness_assessment_calculator
from joblib import dump
from xgboost import XGBClassifier


def load_compas():
    data = pd.read_csv("datasets/compas-raw.csv")
    data = data.drop(["id", "name", "first", "last", "dob", "days_b_screening_arrest",
                      "c_jail_in", "c_jail_out", "c_case_number", "c_arrest_date",
                      "c_charge_desc","r_case_number","r_charge_degree","violent_recid",
                      "r_days_from_arrest","r_offense_date","r_jail_in","r_jail_out",
                      "is_violent_recid","vr_case_number","vr_charge_degree","vr_offense_date",
                      "vr_charge_desc","type_of_assessment","decile_score.1","score_text",
                      "screening_date","v_type_of_assessment","v_decile_score","v_score_text",
                      "v_screening_date","in_custody","out_custody","priors_count.1",
                      "start","end","event","two_year_recid","c_days_from_compas",
                      "r_charge_desc"], axis=1)
    data = data.rename(columns={"sex": "gender", "is_recid":"label"})
    data["screening_year"] = data["compas_screening_date"].apply(lambda x: int(str(x).split("-")[0]))
    data["screening_month"] = data["compas_screening_date"].apply(lambda x: int(str(x).split("-")[1]))
    data["screening_day"] = data["compas_screening_date"].apply(lambda x: int(str(x).split("-")[2]))
    data = data.drop(["compas_screening_date"], axis=1)
    data = data.dropna(subset=['c_offense_date'])
    data["offense_year"] = data["c_offense_date"].apply(lambda x: int(str(x).split("-")[0]))
    data["offense_month"] = data["c_offense_date"].apply(lambda x: int(str(x).split("-")[1]))
    data["offense_day"] = data["c_offense_date"].apply(lambda x: int(str(x).split("-")[2]))
    data = data.drop(["c_offense_date"], axis=1)
    new_race = []
    for i, row in data.iterrows():
        if row["race"] == "Asian" or row["race"] == "Native American":
            new_race.append("Other")
        else:
            new_race.append(row["race"])
    data["race"] = new_race
    return data


def calculate_proto(class_file, prot_thresh, crit_thresh):
    new_prot = []
    for i, row in class_file.iterrows():
        if row["iteration_num"] >= prot_thresh:
            new_prot.append("prototype")
        elif row["iteration_num"] <= crit_thresh:
            new_prot.append("critic")
        else:
            new_prot.append("non")
    class_file["prototype_or_critic"] = new_prot
    return class_file


def run_using_prototype_file(class_0_file_path, class_1_file_path, prot_thresh=0, crit_thresh=0):
    scores = []
    class_0_file = pd.read_csv(class_0_file_path)
    class_1_file = pd.read_csv(class_1_file_path)

    if prot_thresh != 0 and crit_thresh != 0:
        class_0_file = calculate_proto(class_0_file, prot_thresh, crit_thresh)
        class_1_file = calculate_proto(class_1_file, prot_thresh, crit_thresh)

    data_with_proto_column_for_class = [class_0_file, class_1_file]
    compas_protected = ['race', 'gender', 'age_cat']
    results_dict = {}
    for prot in compas_protected:
        results_dict, score = fairness_assessment_calculator(data_with_proto_column_for_class, prot, results_dict)
        # Update result dict
        if results_dict.get("model_fairness_assessment_by_protected") is None:
            results_dict.update({"model_fairness_assessment_by_protected": {prot: score}})
        else:
            results_dict.get("model_fairness_assessment_by_protected").update({prot: score})
        if results_dict.get("model_prototypes_by_protected") is None:
            results_dict.update({"model_prototypes_by_protected": {prot: data_with_proto_column_for_class}})
        else:
            results_dict.get("model_prototypes_by_protected").update({prot: data_with_proto_column_for_class})
        scores.append(score)
        print("protected: {} score: {}".format(prot, score))
    print("final_model_fairness_assessment_score: ", min(scores), " from feature: ",
          compas_protected[scores.index(min(scores))])
    results_dict.update({"final_model_fairness_assessment_score": min(scores),
                         "final_model_fairness_assessment_protected": compas_protected[scores.index(min(scores))],
                         "model_fairness_assessment_score_all_scores": scores})
    return results_dict


def run_compas_60_40_cv_xgboost(exp_num, best_seed):
    compas_protected = ['race', 'gender', 'age_cat']
    categorical_compas = ['race', 'gender', 'age_cat', 'c_charge_degree']
    data = load_compas()
    x_data_cat = data.drop(['label'], axis=1)[categorical_compas]
    x_data_not_cat = data.drop(['label'], axis=1)[data.drop(['label'], axis=1).columns.difference(categorical_compas)]
    y_data = data['label']
    y_data = y_data.reset_index(drop=True)
    d = defaultdict(LabelEncoder)
    x_data_cat = x_data_cat.apply(lambda x: d[x.name].fit_transform(x))
    x_data = pd.concat([x_data_cat, x_data_not_cat], axis=1).reset_index(drop=True)
    scaler = StandardScaler()
    scaler.fit(x_data)
    dump(scaler, 'datasets/folds/compas/std_scaler.bin', compress=True)

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=best_seed)
    x_train = X_train.reset_index(drop=True)
    x_train.to_csv("datasets/folds/compas/{}_{}_x_train.csv".format(exp_num, best_seed), index=False)
    x_test = X_test.reset_index(drop=True)
    x_test.to_csv("datasets/folds/compas/{}_{}_x_test.csv".format(exp_num, best_seed), index=False)
    y_test = y_test.reset_index(drop=True)
    y_test.to_csv("datasets/folds/compas/{}_{}_y_test.csv".format(exp_num, best_seed), index=False)
    y_train = y_train.reset_index(drop=True)
    y_train.to_csv("datasets/folds/compas/{}_{}_y_train.csv".format(exp_num, best_seed), index=False)
    # print("test size:", y_test.shape)
    x_data_scaled = scaler.transform(x_train)
    o_model = XGBClassifier(n_estimators=100, max_depth=4, random_state=best_seed, n_jobs=-1)
    o_model.fit(x_train, y_train.to_numpy())
    pickle.dump(o_model, open("datasets/folds/compas/{}_{}_model_XGBClassifier.pkl".format(exp_num, best_seed), 'wb'))

    CONSTRAINTS = {'race':[categorical(0,5)],
                   'gender':[binary],
                   'age_cat':[categorical(0,2)],
                   'c_charge_degree':[binary],
                   'age':[categorical(18,100)],
                   'decile_score':[categorical(1,10)],
                   'juv_fel_count':[categorical(0,10)],
                   'juv_misd_count':[categorical(0,5)],
                   'juv_other_count':[categorical(0,17)],
                   'offense_day':[categorical(1,31)],
                   'offense_month':[categorical(1,12)],
                   'offense_year':[categorical(1992,2014)],
                   'priors_count':[categorical(0,36)],
                   'screening_day':[categorical(1,31)],
                   'screening_month':[categorical(1,12)],
                   'screening_year':[categorical(2013,2014)]
                   }

    saving_path = "outputs/"
    results_dict = fairness_estimation(o_model, compas_protected, x_test[:30], y_test[:30], x_train, y_train, scaler, 2, 8, saving_path, CONSTRAINTS, best_seed, protected_consideration=True)
    print("Final score : {} \n".format(results_dict.get("final_model_fairness_assessment_score")))
    print("protected are: {}\n".format(compas_protected))
    print("All scores : {} \n".format(results_dict.get("model_fairness_assessment_score_all_scores")))


def compas_brut_force_assessments(name):
    class_0_file_path = "datasets/folds/compas/{}_model_prototypes_class_0_prot_gender.csv".format(name)
    class_1_file_path = "datasets/folds/compas/{}_model_prototypes_class_1_prot_gender.csv".format(name)
    # compas_protected = ['race', 'gender', 'age_cat']
    compas_protected = ['gender'] # changed manually for each evaluation
    results_df = None
    # proto_tresh = range(2, 21)
    # critic_tresh = range(1, 20)
    proto_tresh = [17]
    critic_tresh = [1]
    for t1 in proto_tresh:
        for t2 in critic_tresh:
            if t1>t2:
                results_dict = run_using_prototype_file(class_0_file_path, class_1_file_path, prot_thresh=t1, crit_thresh=t2)
                fairness_row = fairness_row_calc(results_dict, protected=compas_protected)
                if results_df is None:
                    columns = list(fairness_row.keys())
                    columns.append("t_proto")
                    columns.append("t_critic")
                    results_df = pd.DataFrame(columns=columns)
                fairness_row.update({"t_proto":t1})
                fairness_row.update({"t_critic": t2})
                results_df = results_df.append(fairness_row, ignore_index=True)
                print()
    results_df.to_csv("outputs/brut_force_{}_gender.csv".format(name))


def fairness_row_calc(results_dict, protected):
    print()
    dict_to_row = {}
    for prot in protected:
        data = results_dict.get("model_fairness_assessment_by_protected_by_class").get(prot)
        dict_to_row.update({"{}_0".format(prot): data[0]})
        dict_to_row.update({"{}_1".format(prot): data[1]})
        data = results_dict.get("model_fairness_assessment_by_protected").get(prot)
        dict_to_row.update({"{}_final".format(prot): data})
    return dict_to_row


def load_german():
    data = pd.read_csv("datasets/german_raw.csv")
    data = data.rename(columns={"Sex": "Gender"})
    data = data.rename(columns={"Risk": "label"})
    new_race = []
    for i, row in data.iterrows():
        if row["race"] == "Asian-Pac-Islander" or row["race"] == "Amer-Indian-Eskimo":
            new_race.append("Other")
        else:
            new_race.append(row["race"])
    data["race"] = new_race
    data["occupation"] = data["occupation"].replace("?", "Other-service")
    data["native-country"] = data["native-country"].replace("?", "Other")
    data["label"] = data["label"].replace("<=50K", 0)
    data["label"] = data["label"].replace(">50K", 1)
    return data


def run_german_60_40_cv_xgboost(exp_num, best_seed):
    # german_protected = ['Gender', "Age"]
    german_protected = ["Age"] # changed manually according to the experiment
    categorical_german = ['Gender', 'Housing', 'Saving accounts', 'Checking account',
                          'Purpose']
    data = load_german()
    x_data_cat = data.drop(['label'], axis=1)[categorical_german]
    x_data_not_cat = data.drop(['label'], axis=1)[
        data.drop(['label'], axis=1).columns.difference(categorical_german)]
    y_data = data['label']
    y_data = y_data.reset_index(drop=True)
    d = defaultdict(LabelEncoder)
    x_data_cat = x_data_cat.apply(lambda x: d[x.name].fit_transform(x))
    x_data = pd.concat([x_data_cat, x_data_not_cat], axis=1).reset_index(drop=True)
    scaler = StandardScaler()
    scaler.fit(x_data)
    dump(scaler, 'datasets/folds/german/std_scaler.bin', compress=True)

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=best_seed)
    x_train = X_train.reset_index(drop=True)
    x_train.to_csv("datasets/folds/german/{}_{}_x_train.csv".format(exp_num, best_seed), index=False)
    x_test = X_test.reset_index(drop=True)
    x_test.to_csv("datasets/folds/german/{}_{}_x_test.csv".format(exp_num, best_seed), index=False)
    y_test = y_test.reset_index(drop=True)
    y_test.to_csv("datasets/folds/german/{}_{}_y_test.csv".format(exp_num, best_seed), index=False)
    y_train = y_train.reset_index(drop=True)
    y_train.to_csv("datasets/folds/german/{}_{}_y_train.csv".format(exp_num, best_seed), index=False)
    x_data_scaled = scaler.transform(x_train)
    o_model = XGBClassifier(n_estimators=100, max_depth=4, random_state=best_seed, n_jobs=-1)
    o_model.fit(x_train, y_train.to_numpy())
    pickle.dump(o_model,
                open("datasets/folds/german/{}_{}_model_XGBClassifier.pkl".format(exp_num, best_seed), 'wb'))

    CONSTRAINTS = {'Gender': [binary],
                   'Age': [categorical(19, 75)],
                   'Job': [categorical(0, 3)],
                   'Housing': [categorical(0, 2)],
                   'Saving accounts': [categorical(0, 4)],
                   'Checking account': [categorical(0, 3)],
                   'Credit amount': [categorical(250, 18424)],
                   'Duration': [categorical(4, 72)],
                   'Purpose': [categorical(0, 7)]
                   }

    saving_path = "outputs/"
    results_dict = fairness_estimation(o_model, german_protected, x_test, y_test, x_train,
                                       y_train,
                                       scaler, 2, 8, saving_path, CONSTRAINTS, best_seed,
                                       protected_consideration=True)
    print("Final score : {} \n".format(results_dict.get("final_model_fairness_assessment_score")))
    print("protected are: {}\n".format(german_protected))
    print("All scores : {} \n".format(results_dict.get("model_fairness_assessment_score_all_scores")))


def german_run_using_prototype_file(class_0_file_path, class_1_file_path, prot_thresh=0, crit_thresh=0):
    scores = []
    class_0_file = pd.read_csv(class_0_file_path)
    class_1_file = pd.read_csv(class_1_file_path)

    if prot_thresh != 0 and crit_thresh != 0:
        class_0_file = calculate_proto(class_0_file, prot_thresh, crit_thresh)
        class_1_file = calculate_proto(class_1_file, prot_thresh, crit_thresh)

    data_with_proto_column_for_class = [class_0_file, class_1_file]
    german_protected = ['Gender','Age']
    results_dict = {}
    for prot in german_protected:
        results_dict, score = fairness_assessment_calculator(data_with_proto_column_for_class, prot, results_dict)
        # Update result dict
        if results_dict.get("model_fairness_assessment_by_protected") is None:
            results_dict.update({"model_fairness_assessment_by_protected": {prot: score}})
        else:
            results_dict.get("model_fairness_assessment_by_protected").update({prot: score})
        if results_dict.get("model_prototypes_by_protected") is None:
            results_dict.update({"model_prototypes_by_protected": {prot: data_with_proto_column_for_class}})
        else:
            results_dict.get("model_prototypes_by_protected").update({prot: data_with_proto_column_for_class})
        scores.append(score)
        print("protected: {} score: {}".format(prot, score))
    print("final_model_fairness_assessment_score: ", min(scores), " from feature: ",
          german_protected[scores.index(min(scores))])
    results_dict.update({"final_model_fairness_assessment_score": min(scores),
                         "final_model_fairness_assessment_protected": german_protected[scores.index(min(scores))],
                         "model_fairness_assessment_score_all_scores": scores})
    return results_dict


def german_brut_force_assessments(name):
    class_0_file_path = "datasets/folds/german/{}_model_prototypes_class_0_prot_Gender.csv".format(name)
    class_1_file_path = "datasets/folds/german/{}_model_prototypes_class_1_prot_Gender.csv".format(name)
    german_protected = ['Age']
    results_df = None
    proto_tresh = range(2, 21)
    critic_tresh = range(1, 20)
    for t1 in proto_tresh:
        for t2 in critic_tresh:
            if t1 > t2:
                results_dict = german_run_using_prototype_file(class_0_file_path, class_1_file_path, prot_thresh=t1,
                                                        crit_thresh=t2)
                fairness_row = fairness_row_calc(results_dict, protected=german_protected)
                if results_df is None:
                    columns = list(fairness_row.keys())
                    columns.append("t_proto")
                    columns.append("t_critic")
                    results_df = pd.DataFrame(columns=columns)
                fairness_row.update({"t_proto": t1})
                fairness_row.update({"t_critic": t2})
                results_df = results_df.append(fairness_row, ignore_index=True)
                print()
    results_df.to_csv("outputs/german_brut_force_{}_Age.csv".format(name))


def load_synthetic():
    data = pd.read_csv("datasets/sinthetic_partial.csv")
    data = data.rename(columns={"y": "label"})
    return data


def run_synthetic_60_40_cv_xgboost(exp_num, best_seed):
    synthetic_protected = ['fair', 'biased']
    categorical_synthetic = []
    data = load_synthetic()
    y_data = data['label']
    y_data = y_data.reset_index(drop=True)
    d = defaultdict(LabelEncoder)
    x_data = data.drop(['label'], axis=1)
    scaler = StandardScaler()
    scaler.fit(x_data)
    dump(scaler, 'datasets/folds/synthetic/std_scaler.bin', compress=True)

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=best_seed)
    x_train = pd.DataFrame(np.repeat(x_data.values, 20, axis=0))
    x_train.columns = x_data.columns
    x_train = x_train.reset_index(drop=True)
    x_train.to_csv("datasets/folds/synthetic/{}_{}_x_train.csv".format(exp_num, best_seed), index=False)
    x_test = x_data.reset_index(drop=True)
    x_test.to_csv("datasets/folds/synthetic/{}_{}_x_test.csv".format(exp_num, best_seed), index=False)
    y_test = y_data.reset_index(drop=True)
    y_test.to_csv("datasets/folds/synthetic/{}_{}_y_test.csv".format(exp_num, best_seed), index=False)
    y_train = pd.DataFrame(np.repeat(y_data.values, 20, axis=0))
    y_train.columns = ["label"]
    y_train = y_train.reset_index(drop=True)
    y_train.to_csv("datasets/folds/synthetic/{}_{}_y_train.csv".format(exp_num, best_seed), index=False)
    o_model = XGBClassifier(n_estimators=100, max_depth=4, random_state=best_seed, n_jobs=-1)
    o_model.fit(x_train, y_train.to_numpy())
    pickle.dump(o_model,
                open("datasets/folds/synthetic/{}_{}_model_XGBClassifier.pkl".format(exp_num, best_seed), 'wb'))

    CONSTRAINTS = {'fair': [binary],
                   'b': [binary],
                   'biased': [binary],
                   'f1': [binary],
                   'f2': [binary],
                   'f3': [binary],
                   'f4': [binary],
                   'f5': [binary],
                   'f6': [binary],
                   'f7': [binary]
                   }

    saving_path = "outputs/"
    results_dict = fairness_estimation(o_model, synthetic_protected, x_test, y_test, x_train,
                                       y_train,
                                       scaler, 2, 8, saving_path, CONSTRAINTS, best_seed,
                                       protected_consideration=True)
    print("Final score : {} \n".format(results_dict.get("final_model_fairness_assessment_score")))
    print("protected are: {}\n".format(synthetic_protected))
    print("All scores : {} \n".format(results_dict.get("model_fairness_assessment_score_all_scores")))


def synthetic_brut_force_assessments(name):
    class_0_file_path = "outputs/{}_model_prototypes_class_0_prot_biased.csv".format(name)
    class_1_file_path = "outputs/{}_model_prototypes_class_1_prot_biased.csv".format(name)
    german_protected = ['biased']
    results_df = None
    proto_tresh = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    critic_tresh = [1]
    for t1 in proto_tresh:
        for t2 in critic_tresh:
            if t1 > t2:
                results_dict = synthetic_run_using_prototype_file(class_0_file_path, class_1_file_path, prot_thresh=t1,
                                                        crit_thresh=t2)
                fairness_row = fairness_row_calc(results_dict, protected=german_protected)
                if results_df is None:
                    columns = list(fairness_row.keys())
                    columns.append("t_proto")
                    columns.append("t_critic")
                    results_df = pd.DataFrame(columns=columns)
                fairness_row.update({"t_proto": t1})
                fairness_row.update({"t_critic": t2})
                results_df = results_df.append(fairness_row, ignore_index=True)
                print()
    results_df.to_csv("outputs/synthetic_brut_force_{}_biased.csv".format(name))


def synthetic_run_using_prototype_file(class_0_file_path, class_1_file_path, prot_thresh=0, crit_thresh=0):
    scores = []
    class_0_file = pd.read_csv(class_0_file_path)
    class_1_file = pd.read_csv(class_1_file_path)

    if prot_thresh != 0 and crit_thresh != 0:
        class_0_file = calculate_proto(class_0_file, prot_thresh, crit_thresh)
        class_1_file = calculate_proto(class_1_file, prot_thresh, crit_thresh)

    data_with_proto_column_for_class = [class_0_file, class_1_file]
    synthetic_protected = ['biased', 'fair']
    results_dict = {}
    for prot in synthetic_protected:
        results_dict, score = fairness_assessment_calculator(data_with_proto_column_for_class, prot, results_dict)
        # Update result dict
        if results_dict.get("model_fairness_assessment_by_protected") is None:
            results_dict.update({"model_fairness_assessment_by_protected": {prot: score}})
        else:
            results_dict.get("model_fairness_assessment_by_protected").update({prot: score})
        if results_dict.get("model_prototypes_by_protected") is None:
            results_dict.update({"model_prototypes_by_protected": {prot: data_with_proto_column_for_class}})
        else:
            results_dict.get("model_prototypes_by_protected").update({prot: data_with_proto_column_for_class})
        scores.append(score)
        print("protected: {} score: {}".format(prot, score))
    print("final_model_fairness_assessment_score: ", min(scores), " from feature: ",
          synthetic_protected[scores.index(min(scores))])
    results_dict.update({"final_model_fairness_assessment_score": min(scores),
                         "final_model_fairness_assessment_protected": synthetic_protected[scores.index(min(scores))],
                         "model_fairness_assessment_score_all_scores": scores})
    return results_dict


if __name__ == "__main__":
    # Run COMPAS experiments
    run_compas_60_40_cv_xgboost(1, 2100)
    run_compas_60_40_cv_xgboost(1, 2200)
    run_compas_60_40_cv_xgboost(1, 2400)
    run_compas_60_40_cv_xgboost(1, 2500)
    run_compas_60_40_cv_xgboost(1, 2600)
    # Run Statlog experiments
    run_german_60_40_cv_xgboost(1, 78001)
    run_german_60_40_cv_xgboost(1, 78002)
    run_german_60_40_cv_xgboost(1, 78004)
    run_german_60_40_cv_xgboost(1, 78005)
    run_german_60_40_cv_xgboost(1, 78006)
    # Run synthetic experiments
    run_synthetic_60_40_cv_xgboost(1, 111124)