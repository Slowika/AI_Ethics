import os
import fasttext as ft
import pandas as pd
from sys import argv
from utilities import *

TRAINING_DEFAULTS = {
    "lr": 0.1,
    "dim": 300,
    "epoch": 5,
    "wordNgrams": 1,
}

EMBEDDING_FILENAMES = (
    "wiki-news-300d-sample.vec",
    "crawl-300d-sample.vec",
    "wiki-news-300d-subword-sample.vec",
    "crawl-300d-subword-sample.vec"
)

EMBEDDING_NAMES = {
    "wiki-news-300d-sample.vec": "wiki",
    "crawl-300d-sample.vec": "cc",
    "wiki-news-300d-subword-sample.vec": "wiki-sub",
    "crawl-300d-subword-sample.vec": "cc-sub"
}


def train_models(train_data_path, embedding_files=EMBEDDING_FILENAMES, params=None):
    """
    Function for training multiple fasttext classification models, using the same parameters and
    architecture and different word embeddings
    :param train_data_path: a string containing the path to the training data
    :param embedding_files: a list or tuple of strings, containing the filename of each word embedding to load.
    :param params: a dictionary of parameters for the fasttext classification models.
    :return: a list of models of length len(embedding_files) + 1.
    """
    if params is None:
        params = TRAINING_DEFAULTS

    models = []
    print("--------------------------------")
    print("Training baseline model.")
    baseline = ft.train_supervised(input=train_data_path, **params)
    models.append(baseline)

    for embedding in embedding_files:
        print("--------------------------------")
        print("Training model with embedding:", embedding)
        m = ft.train_supervised(input=train_data_path, **params, pretrainedVectors=f"data/{embedding}")
        models.append(m)

    print("--------------------------------")
    return models


def test_models(test_path, models, model_names):
    """
    Function for testing multiple fasttext classification models; gets the precision, recall and f1 score.
    :param test_path: a string containing the path to the training data
    :param models: a list containing trained fasttext classification models.
                    The baseline (no pretrained embeddings) model is always the first element.
    :param model_names: a list or tuple of strings, containing the short name of each model
    :return: a dataframe containing the precision, recall and F1 score for each model.
    """
    labelwise = []
    for m in models:
        _, p, r = m.test(test_path)
        f = 2* (p*r)/(p+r)
        label_values = m.test_label(test_path)
        label_values['_________overall'] = {"precision": p, "recall": r, "f1score": f}
        label_df = pd.DataFrame.from_records(label_values).T
        label_df.index = label_df.index.str.slice(9)
        labelwise.append(label_df)

    results = pd.concat(labelwise, keys=model_names)
    return results


def get_predictions(test_path, models, model_names, params=None):
    """
    Function that saves the predictions for each model and their probabilities, as well as finding where all
    all the models disagree.
    :param test_path: a string containing the path to the training data
    :param models: a list containing trained fasttext classification models.
                    The baseline (no pretrained embeddings) model is always the first element.
    :param model_names: a list or tuple of strings, containing the short name of each model
    :return:
    """
    if params is None:
        params = TRAINING_DEFAULTS
    lr = params["lr"]
    ep = params["epoch"]
    dim = params["dim"]
    ngrams = params["wordNgrams"]

    samples = pd.read_csv(test_path, sep='\n', header=None, names=["s"], dtype=object)["s"]
    true_labels = samples.str.split().str.get(0)
    samples = samples.str.replace(r'__label__[A-Za-z-]* ', "").str.strip()

    predictions = []
    probabilities = []

    for m in models:
        labels, probs = flatten_ft_predictions(m.predict(list(samples), k=1))
        predictions.append(pd.Series(labels, dtype=object))
        probabilities.append(pd.Series(probs))

    cols = ["sample", "true_label"]
    for n in model_names:
        cols.append(f"{n}_pred")
        cols.append(f"{n}_prob")

    df = pd.DataFrame([], columns=cols)
    df.loc[:, "sample"] = samples
    df.loc[:, "true_label"] = true_labels

    pred_cols = []
    for i in range(len(model_names)):
        df.loc[:, cols[i*2 + 2]] = predictions[i]
        df.loc[:, cols[i*2 + 3]] = probabilities[i]
        pred_cols.append(cols[i*2 + 2])

    fname = f"{os.getcwd()}/results/preds_lr{lr}_ep{ep}_d{dim}_ng{ngrams}.csv"
    df.to_csv(fname)

    _, is_diff = find_prediction_disagreement(samples, predictions)

    diffs = df[is_diff][["sample"] + pred_cols]
    diff_fname = f"{os.getcwd()}/results/diffs_lr{lr}_ep{ep}_d{dim}_ng{ngrams}.csv"
    diffs.to_csv(diff_fname)


if __name__ == "__main__":

    if len(argv) < 3:
        print("Training and test data paths required.")
        exit()
    train_data = argv[1]
    test_data = argv[2]
    if len(argv) > 3:
        embeddings = argv[3:]
    else:
        embeddings = EMBEDDING_FILENAMES

    if not os.path.exists("results"):
        os.makedirs("results")

    # Change params here if you like
    params = TRAINING_DEFAULTS

    print("Training models")
    models = train_models(train_data, embeddings, params)

    print("Getting test results")
    names = ["baseline"] + [EMBEDDING_NAMES[x] for x in embeddings]
    test_results = test_models(test_data, models, names)
    lr = params["lr"]
    ep = params["epoch"]
    dim = params["dim"]
    ngrams = params["wordNgrams"]
    result_fname = f"{os.getcwd()}/results/pr_scores_lr{lr}_ep{ep}_d{dim}_ng{ngrams}.csv"
    test_results.to_csv(result_fname)
