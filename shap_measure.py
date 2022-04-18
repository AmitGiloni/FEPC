from collections import defaultdict
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from xgboost import XGBClassifier
from runs import load_compas, load_synthetic, load_german


def shap_measurement(o_model, X_test, prot_name, privilaged_value):
    differences = []

    explainer = shap.Explainer(o_model)
    shap_values = pd.DataFrame(explainer(X_test).values, columns=X_test.columns)
    prot_shap_values = shap_values[prot_name]

    if prot_name == "Age":
        X_test[prot_name] = X_test[prot_name].apply(lambda x: x <= X_test[prot_name].median())

    unique_prot_values = X_test[prot_name].unique()
    priv_indexes = list(X_test.loc[X_test[prot_name] == privilaged_value].index)
    priv_shap_avg = prot_shap_values[priv_indexes].mean()
    for value in unique_prot_values:
        if value != privilaged_value:
            value_index = list(X_test.loc[X_test[prot_name] == value].index)
            value_shap_avg = prot_shap_values[value_index].mean()
            differences.append(abs(value_shap_avg - priv_shap_avg))
    return max(differences)


def run_shap_compas(best_seed):
    # compas_protected = ['race', 'gender', 'age_cat']
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

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=best_seed)
    x_train = X_train.reset_index(drop=True)
    x_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_data_scaled = scaler.transform(x_train)
    o_model = XGBClassifier(n_estimators=100, max_depth=4, random_state=best_seed, n_jobs=-1)
    o_model.fit(x_train, y_train.to_numpy())

    race_fairness = shap_measurement(o_model, x_test, "race", 1)
    gender_fairness = shap_measurement(o_model, x_test, "gender", 1)
    age_cat_fairness = shap_measurement(o_model, x_test, "age_cat", 1)

    return race_fairness, gender_fairness, age_cat_fairness


def run_shap_synthetic(best_seed):
    # synthetic_protected = ['fair', 'biased']
    categorical_synthetic = []
    data = load_synthetic()
    y_data = data['label']
    y_data = y_data.reset_index(drop=True)
    d = defaultdict(LabelEncoder)
    x_data = data.drop(['label'], axis=1)
    scaler = StandardScaler()
    scaler.fit(x_data)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=best_seed)
    x_train = pd.DataFrame(np.repeat(x_data.values, 20, axis=0))
    x_train.columns = x_data.columns
    x_train = x_train.reset_index(drop=True)
    x_test = x_data.reset_index(drop=True)
    y_test = y_data.reset_index(drop=True)
    y_train = pd.DataFrame(np.repeat(y_data.values, 20, axis=0))
    y_train.columns = ["label"]
    y_train = y_train.reset_index(drop=True)
    o_model = XGBClassifier(n_estimators=100, max_depth=4, random_state=best_seed, n_jobs=-1)
    o_model.fit(x_train, y_train.to_numpy())

    fair_fairness = shap_measurement(o_model, x_test, "fair", 1)
    biased_fairness = shap_measurement(o_model, x_test, "biased", 1)
    return fair_fairness, biased_fairness


def run_shap_german(best_seed):
    # german_protected = ['Gender']
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

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=best_seed)
    x_train = X_train.reset_index(drop=True)
    x_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_data_scaled = scaler.transform(x_train)
    o_model = XGBClassifier(n_estimators=100, max_depth=4, random_state=best_seed, n_jobs=-1)
    o_model.fit(x_train, y_train.to_numpy())

    gender_fairness = shap_measurement(o_model, x_test, "Gender", 1)
    age_cat_fairness = shap_measurement(o_model, x_test, "Age", 1)

    return gender_fairness, age_cat_fairness



if __name__ == '__main__':
    # Run COMPAS experiments
    c1_race_fairness, c1_gender_fairness, c1_age_cat_fairness = run_shap_compas(2100)
    c2_race_fairness, c2_gender_fairness, c2_age_cat_fairness = run_shap_compas(2200)
    c4_race_fairness, c4_gender_fairness, c4_age_cat_fairness = run_shap_compas(2400)
    c5_race_fairness, c5_gender_fairness, c5_age_cat_fairness = run_shap_compas(2500)
    c6_race_fairness, c6_gender_fairness, c6_age_cat_fairness = run_shap_compas(2600)
    # Run Statlog experiments
    g1_gender_fairness, g1_age_fairness = run_shap_german(78001)
    g2_gender_fairness, g2_age_fairness = run_shap_german(78002)
    g4_gender_fairness, g4_age_fairness = run_shap_german(78004)
    g5_gender_fairness, g5_age_fairness = run_shap_german(78005)
    g6_gender_fairness, g6_age_fairness = run_shap_german(78006)
    # Run synthetic experiments
    s_fair_fairness, s_biased_fairness = run_shap_synthetic(111124)

