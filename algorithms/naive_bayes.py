import pandas as pd

def naive_bayes_predict(data, input_values, target_col, laplace=False):
    classes = data[target_col].unique()
    total_rows = len(data)

    results = {}

    for c in classes:
        class_data = data[data[target_col] == c]
        class_count = len(class_data)

        prior = class_count / total_rows
        probability = prior

        detail = {
            "Prior": prior,
            "Conditional Probabilities": {}
        }

        for attr, value in input_values.items():
            count_value_in_class = len(
                class_data[class_data[attr] == value]
            )

            unique_values = data[attr].nunique()

            if laplace:
                conditional_prob = (
                    count_value_in_class + 1
                ) / (
                    class_count + unique_values
                )
            else:
                conditional_prob = count_value_in_class / class_count

            probability *= conditional_prob

            detail["Conditional Probabilities"][f"P({attr}={value}|{c})"] = conditional_prob

        detail["Final Probability"] = probability
        results[c] = detail

    predicted_class = max(
        results,
        key=lambda c: results[c]["Final Probability"]
    )

    result_table = []

    for c, info in results.items():
        row = {
            "Class": c,
            "Prior": info["Prior"],
            "Final Probability": info["Final Probability"]
        }

        for k, v in info["Conditional Probabilities"].items():
            row[k] = v

        result_table.append(row)

    result_df = pd.DataFrame(result_table)

    return predicted_class, result_df