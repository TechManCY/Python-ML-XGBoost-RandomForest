import dataCleaning
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer, f1_score
import xgboost as xgb
from scipy.stats import uniform, randint

def splitDataframe (df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def applySMOTE(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def modelBuilding (X_train, y_train): 
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss', 
        random_state=42
    )

    model.fit(X_train, y_train)
    return model 

def modelPredict (model, X_test):
    y_pred = model.predict(X_test)
    return y_pred 

def evaluatePrediction (y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return  accuracy, precision, recall, f1 , cm

def hyperparameterTuning(X_train, y_train):
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )

    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'scale_pos_weight': [1, 3, 5, 7, 10]
    }

    f1_scorer = make_scorer(f1_score)

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
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

def modelTuning (params, X_train, y_train):
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        **params)

    final_model.fit(X_train, y_train)
    return final_model

def printEvaluation(accuracy, precision, recall, f1 , cm):
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
    model = modelBuilding(X_train, y_train)
    y_pred = modelPredict (model, X_test)
    print(y_pred)
    accuracy, precision, recall, f1 , cm= evaluatePrediction (y_pred, y_test)
    # Print results
    print("Model Evaluation Metrics:")
    printEvaluation(accuracy, precision, recall, f1 , cm)
    print("\n" + "="*30 + "\n")

    X_train_smote, y_train_smote = applySMOTE(X_train, y_train)
    model_smote = modelBuilding(X_train_smote, y_train_smote)
    y_pred_smote = modelPredict (model_smote, X_test)
    print(y_pred_smote)
    accuracy_smote, precision_smote, recall_smote, f1_smote , cm_smote= evaluatePrediction (y_pred_smote, y_test)
    # Print results
    print("Model Evaluation Metrics after SMOTE:")
    printEvaluation(accuracy_smote, precision_smote, recall_smote, f1_smote , cm_smote)
    print("\n" + "="*30 + "\n")

    if (f1_smote > f1):
        best_params, best_score = hyperparameterTuning(X_train_smote, y_train_smote)
        tunedModel = modelTuning(best_params,X_train_smote, y_train_smote)
    else: 
        best_params, best_score = hyperparameterTuning(X_train, y_train)
        tunedModel = modelTuning(best_params, X_train, y_train)
    print("Best hyperparameters:", best_params)
    print("Best CV F1 score:", best_score)
        
    y_pred_tuned = modelPredict (tunedModel, X_test)
    accuracy_tuned, precision_tuned, recall_tuned, f1_tuned , cm_tuned= evaluatePrediction (y_pred_tuned, y_test)
    print("\nEvaluation of final tuned model on test data:")
    printEvaluation(accuracy_tuned, precision_tuned, recall_tuned, f1_tuned , cm_tuned)
    print("\n" + "="*30 + "\n")