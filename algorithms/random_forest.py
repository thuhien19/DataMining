from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def run_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
):

   
    # MODEL
   
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
        
    )

    # TRAIN

    model.fit(
        X_train,
        y_train
    )

    # PREDICT

    y_pred = model.predict(
        X_test
    )

    # EVALUATION

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    return (
        model,
        y_pred,
        accuracy,
        report,
        matrix
       
    )

