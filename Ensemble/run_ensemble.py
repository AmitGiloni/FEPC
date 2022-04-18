import pickle
import pandas as pd
import seaborn as sns
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from joblib import load
from Submission.Ensemble.bias_measures.execute_detection.execute import measurements_wrapper


def get_results_df(results_dict):
    """
    The method receives the results dictionary and convert it to a Dataframe according to the protected feature.
    :param results_dict: The results dict produced by the "run" function.
    :return: A Dataframe according to the protected feature.
    """
    preper_dict = {}
    for pf in results_dict.keys():
        results_list = []
        for measure in list(results_dict.get(pf).columns):
            results_list.append(results_dict.get(pf)[measure]['Bias Score'])
        preper_dict.update({pf: results_list})
    return pd.DataFrame.from_dict(preper_dict, orient='index',
                                  columns=list(results_dict.get(list(results_dict.keys())[0]).columns))


def run(data, trained_model, conf_name, protected_features, prot_treshold_cut_dict, labels=None, score_column=None):
    """
    The method runs the fairness analysis.
    :param data:Dataframe. The test data.
    :param trained_model: The model to be evaluated.
    :param conf_name: The name of the experiment/configuration. Used in the saving of the results.
    :param protected_features: List of Strings. The protected features of the data.
    :param prot_treshold_cut_dict: Dict. The protected features thresholds for the measurement run.
    :param labels: Optional. The ground truth labels.
    :param score_column: Optional. String. The name of the risk score column.
    :return: Dataframe of the assessments results.
    """
    results_dict = dict()
    for pf in protected_features:
        pf_m = measurements_wrapper(data=data, data_encoding='none', labels=labels, possible_labels=None,
                                    pf_name=pf, pf_columns=[pf], pf_columns_map=None, model=trained_model,
                                    score_column=score_column, positive_classes=[1], hr_threshold=None,
                                    d_constraints=None, prot_treshold_cut=prot_treshold_cut_dict.get(pf))
        results_dict.update({pf: pf_m})
    target_r = get_results_df(results_dict)
    target_r.to_csv('outputs/ensemble/{}_results.csv'.format(conf_name))
    score = target_r.values.min() * 100
    logger.info("Final model score: {}%".format(score))
    return target_r


def xgboost_fairness_analysis_compas(exp_num):
    """
    :param exp_num: The experiment number
    """
    model_path = "data/{}_model_XGBClassifier.pkl".format(exp_num)
    test_x_path = "data/{}_x_test.csv".format(exp_num)
    test_y_path = "data/{}_y_test.csv".format(exp_num)
    scaler_path = "data/std_scaler.bin"

    model = pickle.load(open(model_path, 'rb'))
    test_x = pd.read_csv(test_x_path)
    test_y = pd.read_csv(test_y_path)
    scaler = load(scaler_path)

    protected = ['race', 'gender', 'age_cat']

    prot_thresholds_cut_dict = {'race': None, 'gender': None, 'age_cat': None}
    results = run(test_x, model, "{}_xgboost_fairness_analysis".format(exp_num), protected, prot_thresholds_cut_dict, labels=test_y)
    graph(results, "race", "{}_xgboost_fairness_analysis".format(exp_num))
    graph(results, "gender", "{}_xgboost_fairness_analysis".format(exp_num))
    graph(results, "age_cat", "{}_xgboost_fairness_analysis".format(exp_num))


def graph(data, name_pf, name_test):
    """
    Saving the graphs of the fairness analysis
    :param data: The data
    :param name_pf: The protected feature name
    :param name_test: The configuration name
    :return:
    """
    data = data.rename(columns=lambda x: x.replace("_", " "))
    data = data.T
    data = data.reset_index()
    data_color = [x for x in list(data[name_pf])]
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(70, 50))

    my_cmap = plt.cm.get_cmap('rocket')
    colors = my_cmap(data_color)
    ax.bar(data["index"], data[name_pf], color=colors)
    sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0, max(data_color)))
    sm.set_array([])

    cbar = plt.colorbar(sm)
    cbar.set_label('Fairness Level', rotation=90, labelpad=25)

    plt.xticks(data["index"], rotation=90)
    for index, row in data.iterrows():
        ax.text(row.name, row[name_pf], round(row[name_pf], 2), color='black', ha="center")
    plt.ylabel("Fairness Score")
    plt.title("Fairness Assessments for \"{}\" in {}".format(name_pf, name_test))
    fig.savefig("outputs/ensemble/Fairness_Assessments_{}_in_{}.png".format(name_pf, name_test))


def xgboost_fairness_analysis_german(exp_num):
    """
    :param exp_num: The experiment number
    """
    model_path = "data/{}_model_XGBClassifier.pkl".format(exp_num)
    test_x_path = "data/{}_x_test.csv".format(exp_num)
    test_y_path = "data/{}_y_test.csv".format(exp_num)
    # scaler_path = "data/german_std_scaler.bin"

    model = pickle.load(open(model_path, 'rb'))
    test_x = pd.read_csv(test_x_path)
    test_y = pd.read_csv(test_y_path)
    # scaler = load(scaler_path)

    protected = ['Gender','Age']

    prot_thresholds_cut_dict = {'Gender': None, 'Age': test_x["Age"].median()}
    results = run(test_x, model, "{}_german_xgboost_fairness_analysis".format(exp_num), protected, prot_thresholds_cut_dict, labels=test_y, score_column="Duration")
    graph(results, "Gender", "{}_german_xgboost_fairness_analysis".format(exp_num))
    graph(results, "Age", "{}_german_xgboost_fairness_analysis".format(exp_num))


def xgboost_fairness_analysis_synthetic(exp_num):
    """
    :param exp_num: The experiment number
    """
    model_path = "data/{}_model_XGBClassifier.pkl".format(exp_num)
    test_x_path = "data/{}_x_test.csv".format(exp_num)
    test_y_path = "data/{}_y_test.csv".format(exp_num)

    model = pickle.load(open(model_path, 'rb'))
    test_x = pd.read_csv(test_x_path)
    test_y = pd.read_csv(test_y_path)

    protected = ['fair', 'biased']

    prot_thresholds_cut_dict = {'fair': None, 'biased': None}
    results = run(test_x, model, "{}_synthetic_xgboost_fairness_analysis".format(exp_num), protected, prot_thresholds_cut_dict, labels=test_y, score_column=None)
    graph(results, "fair", "{}_synthetic_xgboost_fairness_analysis".format(exp_num))
    graph(results, "biased", "{}_synthetic_xgboost_fairness_analysis".format(exp_num))



if __name__ == '__main__':
    xgboost_fairness_analysis_compas(exp_num="1_2100")
    xgboost_fairness_analysis_compas(exp_num="1_2200")
    xgboost_fairness_analysis_compas(exp_num="1_2400")
    xgboost_fairness_analysis_compas(exp_num="1_2500")
    xgboost_fairness_analysis_compas(exp_num="1_2600")

    xgboost_fairness_analysis_german(exp_num="1_78001")
    xgboost_fairness_analysis_german(exp_num="1_78002")
    xgboost_fairness_analysis_german(exp_num="1_78004")
    xgboost_fairness_analysis_german(exp_num="1_78005")
    xgboost_fairness_analysis_german(exp_num="1_78006")

    xgboost_fairness_analysis_synthetic(exp_num="1_111124")
