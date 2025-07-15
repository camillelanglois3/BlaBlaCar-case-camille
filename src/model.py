import pandas as pd
import numpy as np
import time
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_curve, accuracy_score, precision_score, recall_score, 
                        f1_score, average_precision_score, confusion_matrix, classification_report)



def evaluate_and_plot(models, X_train, y_train, X_test, y_test, scoring, cv):
    cv_results = {}
    pr_curves = {}
    test_metrics = []

    for name, model in models.items():
        print(f"Training: {name}")
        start = time.time()

        pipeline = Pipeline([
            # ('smote', SMOTE(random_state=42)),
            ('clf', model)
        ])

        scores = cross_validate(
            pipeline,
            X_train, y_train,
            scoring=scoring,
            cv=cv,
            return_train_score=False,
            return_estimator=True
        )

        elapsed = time.time() - start
        mean_scores = {k: np.mean(v) for k, v in scores.items() if k.startswith('test_')}
        mean_scores['fit_time'] = elapsed
        cv_results[name] = mean_scores

        # Test on last model of CV
        best_estimator = scores['estimator'][-1]
        best_estimator.fit(X_train, y_train)
        start_inference = time.time()
        y_proba = best_estimator.predict_proba(X_test)[:, 1]
        inference_time = time.time() - start_inference

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_curves[name] = (precision, recall)

        test_metrics.append({
            "model": name,
            "test_accuracy": accuracy_score(y_test, best_estimator.predict(X_test)),
            "test_precision": precision_score(y_test, (y_proba >= 0.5)),
            "test_recall": recall_score(y_test, (y_proba >= 0.5)),
            "test_f1": f1_score(y_test, (y_proba >= 0.5)),
            "test_auc_pr": average_precision_score(y_test, y_proba),
            "inference_time": inference_time
        })

    return pd.DataFrame(cv_results).T, pd.DataFrame(test_metrics), pr_curves


def plot_pr_curves(pr_curves, y_test, X_test):
    plt.figure(figsize=(8, 6))
    for name, (precision, recall) in pr_curves.items():
        ap = np.trapz(precision, recall)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(True)
    plt.show()


def final_evaluation(y_proba, y_test, best_threshold, inference_time, save=False):
    preds = (y_proba >= best_threshold).astype(int)
    print(classification_report(y_test, preds))
    print("test_precision", precision_score(y_test, preds))
    print("test_recall", recall_score(y_test, preds))
    print("test_f1", f1_score(y_test, preds))
    print("inference time", inference_time)

    # Matrix confusion
    cm = confusion_matrix(y_test, preds)
    # Normalisation by total (option : axis=1 ou axis=0 pour ligne/colonne)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Annotations : class percentage
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percent = cm[i, j] / cm.sum(axis=1)[i] * 100  # Pourcentage par ligne
            annot[i, j] = f"{percent:.1f}%\n({cm[i, j]})"

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_normalized, annot=annot, fmt='', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix - XGBoost')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    if save:
        report_dict = classification_report(y_test, preds, output_dict=True)
        with open("../results/classification_report.json", "w") as f:
            json.dump(report_dict, f, indent=4)
        plt.savefig('../results/confusion_matrix.png')
    plt.show()


def preparation_modeling():
    print("------ Data formatting for model ------")
    core_features = [
        'unit_seat_price_eur',
        # 'segment_distance_km', # highly correlated to price
        'to_cluster_popularity',
        'from_cluster_popularity',
        'is_main_segment',
        'price_x_popularity', 
        'seats_x_distance' # Even if correlated with distance and price, seats_x_distance could capture a multiplicative effect
    ]

    bonus_features = [
        'driver_account_age_days',
        'driver_trip_count',
        'departure_hour',
        'departure_weekday',
        'hours_before_departure',
        'is_auto_accept_mode',
        'fixed_signup_country_grouped',
        'is_weekend',
        'is_holiday',
    ]

    bool_features = [
        'is_main_segment',
        'is_auto_accept_mode',
        'is_holiday', 
        'success', 
    ]

    df = pd.read_parquet('../data/processed/enriched_dataset.parquet')

    for var in bool_features:
        df[var] = df[var].astype(int)

    # transformation departure_hour
    df['hour_sin'] = np.sin(2 * np.pi * df['departure_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['departure_hour'] / 24)

    # one-hot encoding
    one_hot_features = [
        'departure_weekday',
        'fixed_signup_country_grouped', 
    ]
    df = pd.get_dummies(df, columns=one_hot_features, drop_first=True)

    bonus_features += ['hour_sin', 'hour_cos', 'departure_weekday_1', 'departure_weekday_2', 
                   'departure_weekday_3', 'departure_weekday_4', 'departure_weekday_5', 
                   'departure_weekday_6', 'fixed_signup_country_grouped_foreign', 
                   'fixed_signup_country_grouped_missing',
                   ] 

    bonus_features.remove('departure_hour')
    bonus_features.remove('departure_weekday')
    bonus_features.remove('fixed_signup_country_grouped')

    X = df[core_features + bonus_features].copy()
    y = df['success']

    # Spliting the dataset in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardisation
    scaler = StandardScaler()

    # Numerical columns to scale
    num_cols = [
        'unit_seat_price_eur',
        'segment_distance_km',
        'to_cluster_popularity',
        'from_cluster_popularity',
        'price_x_popularity', 
        'seats_x_distance',
        'driver_account_age_days', # attention
        'driver_trip_count', # attention
        'departure_hour',
        'hours_before_departure', 
    ]

    core_num_cols = list(set(core_features + bonus_features) & set(num_cols))

    X_train[core_num_cols] = scaler.fit_transform(X_train[core_num_cols])
    X_test[core_num_cols] = scaler.transform(X_test[core_num_cols])

    # Saving
    print("Saving model datasets")
    joblib.dump(X_train, '../data/processed/X_train.pkl')
    joblib.dump(y_train, '../data/processed/y_train.pkl')
    joblib.dump(X_test, '../data/processed/X_test.pkl')
    joblib.dump(y_test, '../data/processed/y_test.pkl')


def train_and_test_model():
    X_train = joblib.load('../data/processed/X_train.pkl')
    y_train = joblib.load('../data/processed/y_train.pkl')
    X_test = joblib.load('../data/processed/X_test.pkl')
    y_test = joblib.load('../data/processed/y_test.pkl')

    best_params = joblib.load('../results/best_params_xgb.pkl')
    optimal_threshold = joblib.load('../results/optimal_threshold.pkl')
    final_features = joblib.load('../results/final_features.pkl')

    X_train = X_train[final_features]
    X_test = X_test[final_features]

    # model training
    model = XGBClassifier(**best_params, use_label_encoder=False, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    # evaluation
    y_pred = model.predict(X_test)
    start_inference = time.time()
    y_proba = model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_inference

    final_evaluation(y_proba, y_test, optimal_threshold, inference_time, save=True)