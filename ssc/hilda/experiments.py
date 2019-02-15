from hilda.meta_regressors import train_random_forest_regressor, evaluate_regressor, compare_against_calibration_baseline


def reapply_perturbations(dataset, learner, perturbations_for_training,
                          perturbations_for_evaluation, perturbations_name):

    train_data, test_data, target_data = learner.split(dataset.df)

    y_train = dataset.labels_from(train_data)
    y_test = dataset.labels_from(test_data)
    y_target = dataset.labels_from(target_data)

    print("\nTraining model on perturbed data.")
    model = learner.fit(dataset, train_data)

    score_on_train_data = learner.score(y_train, model.predict(train_data))
    score_on_noncorrupted_test_data = learner.score(y_test, model.predict(test_data))
    score_on_noncorrupted_target_data = learner.score(y_target, model.predict(target_data))

    print(learner.scoring, "on train data: ", score_on_train_data)
    print(learner.scoring, "on test data: ", score_on_noncorrupted_test_data)
    print(learner.scoring, "on target data: ", score_on_noncorrupted_target_data)

    print("\nTraining meta regressor on perturbed test data.")
    meta_regressor = train_random_forest_regressor(test_data, y_test, perturbations_for_training, model, learner)

    print("\nEvaluating meta regressor on perturbed target data.")

    mse, mae, plot_file = evaluate_regressor(target_data, y_target, perturbations_for_evaluation,
                                                             model, meta_regressor, learner, dataset.name(),
                                                             perturbations_name)

    log_line = "\t".join(["reapply_perturbations", dataset.name(), str(score_on_train_data),
                          str(score_on_noncorrupted_test_data), str(score_on_noncorrupted_target_data),
                          learner.name, learner.scoring, perturbations_name, str(mse), str(mae), plot_file])
    print(log_line)
    return log_line, model, mse, mae


def compare_to_calibration(dataset, learner, perturbations_for_training,
                           perturbations_for_evaluation, perturbations_name):

    train_data, test_data, target_data = learner.split(dataset.df)

    y_train = dataset.labels_from(train_data)
    y_test = dataset.labels_from(test_data)
    y_target = dataset.labels_from(target_data)

    print("\nTraining model on perturbed data.")
    model = learner.fit(dataset, train_data)

    score_on_train_data = learner.score(y_train, model.predict(train_data))
    score_on_noncorrupted_test_data = learner.score(y_test, model.predict(test_data))
    score_on_noncorrupted_target_data = learner.score(y_target, model.predict(target_data))

    print(learner.scoring, "on train data: ", score_on_train_data)
    print(learner.scoring, "on test data: ", score_on_noncorrupted_test_data)
    print(learner.scoring, "on target data: ", score_on_noncorrupted_target_data)

    print("\nTraining meta regressor on perturbed test data.")
    meta_regressor = train_random_forest_regressor(test_data, y_test, perturbations_for_training, model, learner)

    print("\nEvaluating meta regressor on perturbed target data.")

    compare_against_calibration_baseline(target_data, y_target, perturbations_for_evaluation,
                                                             model, meta_regressor, learner, dataset.name(),
                                                             perturbations_name)
