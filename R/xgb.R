library(xgboost)
predictors <- c(
  "gender", "issue_age", "face_amount", "post_level_premium_structure",
  "premium_jump_ratio", "risk_class", "premium_mode"
)

rec_xgb <- recipe(f, data = training) %>%
  step_string2factor(all_predictors()) %>%
  prep(retain = TRUE, stringsAsFactors = FALSE)

make_xgb_data <- function(data, rec_xgb) {
  data <- bake(rec_xgb, data)
  xgb.DMatrix(
    data = Matrix::sparse.model.matrix(
      ~ . - 1,
      data = select(data, predictors)
    ),
    label = data$lapse_count_rate
  )
}
training_data <- make_xgb_data(training, rec_xgb)
validation_data <- make_xgb_data(validation, rec_xgb)

param_list <- list(
  subsample = c(0.5, 0.7),
  max_depth = c(5, 10),
  # colsample_bytree = c(1),
  eta = c(0.1, 0.2)
)

xgb_models <- param_list %>%
  purrr::cross() %>%
  purrr::map(xgb.cv, data = training_data, nrounds = 400, metrics = "rmse", nfold = 3,
             early_stopping_rounds = 10, objective = "reg:logistic")

cv_summary <- xgb_models %>%
  map_df(~ c(
    .x$params,
    best_iteration = .x$best_iteratio,
    rmse = .x$evaluation_log$test_rmse_mean %>% tail(1)
  ))

xgb_model <- xgboost(data = training_data, params = list(
  subsample = 0.7,
  max_depth = 5,
  eta = 0.1), nrounds = 400)

prediction <- predict(xgb_model, validation_data)

validation_summary <- validation %>% mutate(
  predicted_count_rate = prediction
)

metrics(validation_summary, lapse_count_rate, predicted_count_rate)
# A tibble: 1 x 3
# rmse   rsq   mae
# <dbl> <dbl> <dbl>
#1 0.268 0.372 0.194
