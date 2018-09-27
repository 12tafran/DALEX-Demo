library(shocklapse)
library(tidyverse)
library(recipes)
library(keras)
library(yardstick)
library(DALEX)

predictors <- c(
  "gender", "issue_age", "face_amount", "post_level_premium_structure",
  "premium_jump_ratio", "risk_class", "premium_mode"
)
responses <- c("lapse_count_rate", "lapse_amount_rate")

f <- as.formula(paste(
  paste(responses, collapse = " + "),
  paste(predictors, collapse = " + "),
  sep = "~"
))

rec_nn <- recipe(f, data = training) %>%
  step_string2factor(all_predictors()) %>%
  # step_ordinalscore(all_predictors(), convert = function(x) as.integer(x) - 1) %>%
  prep(retain = TRUE, stringsAsFactors = FALSE)

make_keras_data <- function(data) {
  data <- data %>%
    map_if(is.factor, ~ as.integer(.x) - 1) %>%
    map_at("gender", ~ keras::to_categorical(.x, 2) %>% array_reshape(c(length(.x), 2))) %>%
    map_at("post_level_premium_structure",
           ~ keras::to_categorical(.x, 2) %>% array_reshape(c(length(.x), 2))) %>%
    map_at("premium_mode", ~ keras::to_categorical(.x, 6) %>%
             array_reshape(c(length(.x), 6)))

  list(x = data[predictors],
       y = data[responses])
}

training_data <- make_keras_data(juice(rec_nn))
testing_data <- bake(rec_nn, testing) %>%
  make_keras_data()
validation_data <- bake(rec_nn, validation) %>%
  make_keras_data()

# Build network.
# Note that the one-hot encoded inputs will have shape > 1. The rest
#  will be fed to embedding layers so we keep the original representation.
input_gender <- layer_input(shape = 2, name = "gender")
input_issue_age_group <- layer_input(shape = 1, name = "issue_age")
input_face_amount_band <- layer_input(shape = 1, name = "face_amount")
input_post_level_premium_structure <- layer_input(shape = 2, name = "post_level_premium_structure")
input_prem_jump_d11_d10 <- layer_input(shape = 1, name = "premium_jump_ratio")
input_risk_class <- layer_input(shape = 1, name = "risk_class")
input_premium_mode <- layer_input(shape = 6, name = "premium_mode")

#
# output_issue_age_group <- input_issue_age_group %>%
#   layer_embedding(7, 6) %>%
#   layer_flatten()

concat_inputs <- layer_concatenate(list(
  input_gender,
  # output_issue_age_group,
  input_issue_age_group %>%
    layer_embedding(7, 2) %>%
    layer_flatten(),
  input_face_amount_band %>%
    layer_embedding(4, 2) %>%
    layer_flatten(),
  input_post_level_premium_structure,
  input_prem_jump_d11_d10 %>%
    layer_embedding(25, 24) %>%
    layer_flatten(),
  input_risk_class %>%
    layer_embedding(9, 2) %>%
    layer_flatten(),
  input_premium_mode
))

main_layer <- concat_inputs %>%
  layer_dense(units = 32, activation = "relu")


output_count_rate <- main_layer %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid", name = "lapse_count_rate")

output_amount_rate <- main_layer %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid", name = "lapse_amount_rate")

model <- keras_model(
  inputs = c(input_gender, input_issue_age_group, input_face_amount_band,
             input_post_level_premium_structure, input_prem_jump_d11_d10,
             input_risk_class, input_premium_mode),
  outputs = c(output_count_rate, output_amount_rate)
)

model %>%
  compile(
    optimizer = optimizer_adam(amsgrad = TRUE),
    loss = "mse",
    loss_weights = c(0.8, 0.2)
  )

history <- model %>%
  fit(
    x = training_data$x,
    y = training_data$y,
    batch_size = 256,
    epochs = 100#,
    # validation_split = 0.2
  )

predictions <- predict(
  model,
  validation_data$x
)

# WIP. Here `validation_summary` isn't a summary (yet), we're just
#  cbinding the predictions to the original data.
validation_summary <- validation %>% bind_cols(
  predictions %>%
    setNames(c("predicted_count_rate", "predicted_amount_rate")) %>%
    as.data.frame()
)

metrics(validation_summary, lapse_count_rate, predicted_count_rate)
# # A tibble: 1 x 3
# rmse   rsq   mae
# <dbl> <dbl> <dbl>
#   1 0.263 0.395 0.189

metrics(validation_summary, lapse_amount_rate, predicted_amount_rate)
# # A tibble: 1 x 3
# rmse   rsq   mae
# <dbl> <dbl> <dbl>
#   1 0.265 0.389 0.191

validation_summary %>%
  arrange(predicted_count_rate) %>%
  select(predicted_count_rate, lapse_count_rate) %>%
  ggplot(aes(x = predicted_count_rate, y = lapse_count_rate)) +
  geom_point()

validation_summary %>%
  arrange(predicted_count_rate) %>%
  select(predicted_count_rate, lapse_count_rate) %>%
  mutate(decile = cut(predicted_count_rate, quantile(
    predicted_count_rate, probs = seq(0, 1, 0.1)
  ), include.lowest = TRUE)) %>%
  group_by(decile) %>%
  summarize(mean_predicted = mean(predicted_count_rate),
            mean_actual = mean(lapse_count_rate))


