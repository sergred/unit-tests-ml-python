import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt


def percentiles_of_probas(predictions):
    probs_class_a = np.transpose(predictions)[0]
    probs_class_b = np.transpose(predictions)[1]
    features_a = np.percentile(probs_class_a, np.arange(0, 101, 5))
    features_b = np.percentile(probs_class_b, np.arange(0, 101, 5))
    return np.concatenate((features_a, features_b), axis=0)


def train_random_forest_regressor(test_data, y_test, perturbations_to_apply, model, learner):

    generated_training_data = []

    for perturbation in perturbations_to_apply:
        corrupted_test_data = perturbation.transform(test_data)

        predictions = model.predict_proba(corrupted_test_data)
        features = percentiles_of_probas(predictions)

        score_on_corrupted_test_data = learner.score(y_test, model.predict(corrupted_test_data))

        example = np.concatenate((features, [score_on_corrupted_test_data]), axis=0)

        generated_training_data.append(example)

    num_features = 42

    X = np.array(generated_training_data)[:, :num_features]
    y = np.array(generated_training_data)[:, num_features]

    param_grid = {
        'learner__n_estimators': np.arange(5,20,5),
        'learner__criterion': ['mae']
    }

    meta_regressor_pipeline = Pipeline([
        ('scaling', StandardScaler()),
        ('learner', RandomForestRegressor(criterion='mae'))
    ])

    meta_regressor = GridSearchCV(meta_regressor_pipeline, param_grid, scoring='neg_mean_absolute_error').fit(X, y)

    return meta_regressor


def evaluate_regressor(target_data, y_target, perturbations_to_apply, model, meta_regressor, learner, dataset_name,
                       perturbations_name):

    predicted_scores = []
    true_scores = []

    for perturbation in perturbations_to_apply:
        corrupted_target_data = perturbation.transform(target_data)

        predictions = model.predict_proba(corrupted_target_data)
        features = percentiles_of_probas(predictions)

        score_on_corrupted_target_data = learner.score(y_target, model.predict(corrupted_target_data))
        predicted_score_on_corrupted_target_data = meta_regressor.predict([features])

        predicted_scores.append(predicted_score_on_corrupted_target_data)
        true_scores.append(score_on_corrupted_target_data)

    plt.plot([0, 1], [0, 1], '-', color='grey', alpha=0.5)

    min_score = np.min(predicted_scores + true_scores) - 0.05
    max_score = np.max(predicted_scores + true_scores) + 0.05

    plt.scatter(true_scores, predicted_scores, alpha=0.05)

    plt.xlabel("true " + learner.scoring, fontsize=18)
    plt.ylabel("predicted " + learner.scoring, fontsize=18)

    plt.xlim((min_score, max_score))
    plt.ylim((min_score, max_score))

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(true_scores, predicted_scores)
    mae = mean_absolute_error(true_scores, predicted_scores)

    text_x = min_score + ((max_score - min_score) / 3.0)
    text_y = min_score + ((max_score - min_score) / 10.0)

    plt.text(text_x, text_y, "MSE %.5f   MAE %.4f" % (mse, mae), fontsize=12,
             bbox=dict(facecolor='none', edgecolor='black', pad=10.0))

    print("MSE %.5f, MAE %.4f" % (mse, mae))

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.legend(['perfect', 'predicted'], fontsize=18)
    plt.gcf().set_size_inches(6, 5)

    plot_file = 'ssc/figures/' + "__".join([dataset_name, perturbations_name, learner.name, learner.scoring]) + ".pdf"

    print("Writing plot to " + plot_file)
    plt.tight_layout()
    plt.gcf().savefig(plot_file, dpi=300)

    plt.clf()
    plt.cla()
    plt.close()

    results_file = 'ssc/results/' + "__".join([dataset_name, perturbations_name, learner.name, learner.scoring]) + ".tsv"

    with open(results_file, 'w') as the_file:
        the_file.write('y_true\ty_pred\n')
        for true_score, predicted_score in zip(true_scores, predicted_scores):
            the_file.write('%s\t%s\n' % (true_score, predicted_score[0]))

    return mse, mae, plot_file
