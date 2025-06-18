import dataCleaning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    make_scorer,classification_report)
from scipy.stats import uniform, randint
from sklearn.utils.class_weight import compute_class_weight


def splitDataframe (df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

# This is specific for XGB
#def calculate_scale_pos_weight(y_train):
    # Weight = (number of negative class) / (number of positive class)
#    num_neg = (y_train == 0).sum()
#    num_pos = (y_train == 1).sum()
#    return num_neg / num_pos

def calculate_class_weights(y_train):
    """Return dictionary for class_weight parameter in RandomForestClassifier"""
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)
    return {0: class_weights[0], 1: class_weights[1]}

def applySMOTE(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def modelBuilding(X_train, y_train, class_weight=None):
    model = RandomForestClassifier(
        random_state=42,
        class_weight=class_weight,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def modelPredict(model, X_test):
    return model.predict(X_test)

def evaluatePrediction(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, cm

def hyperparameterTuning(X_train, y_train):
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(5, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }

    f1_scorer = make_scorer(f1_score)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,
        scoring=f1_scorer,
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    return random_search.best_params_, random_search.best_score_

def modelTuning(params, X_train, y_train):
    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **params
    )
    model.fit(X_train, y_train)
    return model

def printEvaluation(accuracy, precision, recall, f1, cm):
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    oneHotEncodedDf = dataCleaning.main()
    print(oneHotEncodedDf.head())

    X_train, X_test, y_train, y_test = splitDataframe(oneHotEncodedDf)

    # Weighted model (no SMOTE)
    class_weights = calculate_class_weights(y_train)
    print("Class Weights:", class_weights)
    model_weighted = modelBuilding(X_train, y_train, class_weight=class_weights)
    y_pred_weighted = modelPredict(model_weighted, X_test)
    print("Predictions (Weighted RF):", y_pred_weighted)
    acc_w, prec_w, rec_w, f1_w, cm_w = evaluatePrediction(y_pred_weighted, y_test)
    print("\nEvaluation (Weighted Random Forest):")
    printEvaluation(acc_w, prec_w, rec_w, f1_w, cm_w)
    print("\n" + "=" * 30 + "\n")

    # SMOTE model
    X_train_smote, y_train_smote = applySMOTE(X_train, y_train)
    model_smote = modelBuilding(X_train_smote, y_train_smote)
    y_pred_smote = modelPredict(model_smote, X_test)
    print("Predictions (SMOTE RF):", y_pred_smote)
    acc_s, prec_s, rec_s, f1_s, cm_s = evaluatePrediction(y_pred_smote, y_test)
    print("\nEvaluation (Random Forest + SMOTE):")
    printEvaluation(acc_s, prec_s, rec_s, f1_s, cm_s)
    print("\n" + "=" * 30 + "\n")

    # Tuning with SMOTE data
    best_params, best_score = hyperparameterTuning(X_train_smote, y_train_smote)
    tuned_model = modelTuning(best_params, X_train_smote, y_train_smote)
    print("Best Hyperparameters:", best_params)
    print("Best CV F1 Score:", best_score)

    y_pred_tuned = modelPredict(tuned_model, X_test)
    acc_t, prec_t, rec_t, f1_t, cm_t = evaluatePrediction(y_pred_tuned, y_test)
    print("\nEvaluation (Tuned Random Forest):")
    printEvaluation(acc_t, prec_t, rec_t, f1_t, cm_t)
    print("\n" + "=" * 30 + "\n")

    print("Classification Report (Tuned RF):")
    print(classification_report(y_test, y_pred_tuned))
