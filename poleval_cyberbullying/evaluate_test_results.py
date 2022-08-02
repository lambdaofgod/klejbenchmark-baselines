import fire
import pandas as pd
from sklearn import metrics


def evaluate_poleval_cyberbullying_detection(predictions_path):
    labels_pred = pd.read_csv(predictions_path)
    labels_test = pd.read_csv("test_labels.tsv")
    f1_score = metrics.f1_score(labels_test, labels_pred)
    print(f"F1: {round(f1_score * 100, 1)}")
    print(metrics.classification_report(labels_test, labels_pred))


fire.Fire(evaluate_poleval_cyberbullying_detection)
