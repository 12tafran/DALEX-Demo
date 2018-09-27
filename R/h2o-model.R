library(shocklapse)
library(tidyverse)
library(h2o)
training <- training
validation <- validation

predictors <- c(
  "gender", "issue_age", "face_amount", "post_level_premium_structure",
  "premium_jump_ratio", "risk_class", "premium_mode")
responses <- c("lapse_count_rate")
training_h2o <- training %>% mutate(lapse_count_rate = as.double(lapse_count_rate)) %>% 
  select(one_of(predictors), one_of(responses)) %>%
  mutate_at(.vars = vars(one_of(predictors)), .funs = as.factor)
validation_h2o <- validation %>% mutate(lapse_count_rate = as.double(lapse_count_rate)) %>%
  select(one_of(predictors), one_of(responses)) %>%
  mutate_at(.vars = vars(one_of(predictors)), .funs = as.factor)


h2o.init()

df_test <- as.h2o(validation_h2o, destination_frame = "df_test.hex")
df_train <- as.h2o(training_h2o, destination_frame = "df_train.hex")


hyper_parameters =
  list(
    max_depth = seq(2,5,1),
    ntrees=seq(100,500,100),
    nbins = seq(10,50,10)
  )
search_criteria = list(max_runtime_secs =100,
                       max_models = 100,
                       seed = 1234567,
                       strategy = "RandomDiscrete"
)
rf_grid <- h2o.grid("randomForest"
                    ,training_frame = df_train
                    ,validation_frame = df_train
                    ,x = predictors
                    ,y = responses
                    ,grid_id = "samplemodels"
                    ,hyper_params = hyper_parameters
                    ,search_criteria = search_criteria
                    ,stopping_metric="RMSE"
)
sorted_grid = h2o.getGrid("samplemodels",sort_by = "rmse",decreasing = FALSE)
samplemodel = h2o.getModel(sorted_grid@model_ids[[1]])
predict <- as.data.frame(h2o.predict(samplemodel, df_test))

# h2o.shutdown(prompt=FALSE)