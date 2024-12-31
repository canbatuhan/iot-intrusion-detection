import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, multilabel_confusion_matrix, classification_report, precision_score, recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def dataset_summary(df):
    print(25*"-", "SUMMARY", 25*"-")
    print("Number of rows :", len(df))
    print("Number of columns :", len(df.columns))
    print(75*"-")

def dataset_detailed_summary(df, column_name):
    print(25*"-", "DETAILED SUMMARY", 25*"-")
    print(df[column_name].value_counts())
    print("Number of rows :", len(df))
    print("Number of columns :", len(df.columns))
    print(75*"-")

def attack_type_distribution(df, label_name):
    plt.figure(figsize=(20, 10))
    sns.countplot(data=df, x=label_name, palette="Set2")
    plt.title("Attack Type Distribution")
    plt.xlabel("Attack Type")
    plt.ylabel("Number of Records")
    plt.show()

def correlation_matrix(df, label_name):
    corr_matrix = df.drop([label_name], axis=1).corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap of Numerical Features")
    plt.show()

def print_score(y_true, y_pred):
    print("Accuracy (Sample-wise):", accuracy_score(y_true, y_pred))
    print("Precision (Micro):", precision_score(y_true, y_pred, average="micro"))
    print("Recall (Micro):", recall_score(y_true, y_pred, average="micro"))
    print("F1 Score (Micro):", f1_score(y_true, y_pred, average="micro"))

    print("\nPrecision-Recall-F1 (Macro Average):")
    print(precision_recall_fscore_support(y_true, y_pred, average="macro"))

    print("\nPrecision-Recall-F1 (Weighted Average):")
    print(precision_recall_fscore_support(y_true, y_pred, average="weighted"))

    print("\nClassification Report (Label-wise):")
    print(classification_report(y_true, y_pred))

def conf_matrix(y_true, y_pred):
    y_true_single = y_true.argmax(axis=1)
    y_pred_single = y_pred.argmax(axis=1)

    cm = confusion_matrix(y_true_single, y_pred_single)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def conf_matrix_only_attacks(y_true, y_pred):
    y_true_single = y_true.argmax(axis=1)
    y_pred_single = y_pred.argmax(axis=1)

    cm = confusion_matrix(y_true_single, y_pred_single)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()