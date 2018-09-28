library(DALEX)
source("R/neural-net.R")
source("R/glm.R")
source("R/averages-method.R")
source("R/xgb.R")
source("R/h2o-model.R")

validation <- validation %>% 
  mutate_at(vars(one_of(predictors)), .funs = as.factor)

custom_predict_nn <- function(model, data) {
  keras_data <- bake(rec_nn, data) %>%
    make_keras_data()
  predict(model, keras_data$x)[[1]] %>%
    as.vector()
}


custom_predict_xgb <- function(model, data) {
  data <- bake(rec_xgb, data)
  xgb_data <-   xgb.DMatrix(
    data = Matrix::sparse.model.matrix(
      ~ . - 1,
      data = select(data, predictors)
    )
  )
  predict(model, xgb_data) %>%
    as.vector()
}

custom_predict_h2o <- function(model, newdata)  {
  newdata_h2o <- as.h2o(newdata)
  res <- as.data.frame(h2o.predict(model, newdata_h2o))
  return(as.numeric(res$predict))
}

explainer_h2o_rf <- explain(
  model = samplemodel,
  data = select(validation_h2o, predictors),
  y = validation_h2o$lapse_count_rate,
  predict_function = custom_predict_h2o,
  label = "Random Forest (H2O)")


explainer_nn <- explain(
  model, data = select(validation, predictors),
  y = validation$lapse_count_rate,
  predict_function = custom_predict_nn,
  label = "Neural Network (Keras)"
)

explainer_xgb <- DALEX::explain(
  xgb_model,
  data = select(validation, predictors),
  y = validation$lapse_count_rate,
  predict_function = custom_predict_xgb,
  label = "Gradient boosting (xgboost)"
)

explainer_glm <- DALEX::explain(
  glm1,
  data = select(validation, predictors),
  y = validation$lapse_count_rate,
  label = "GLM"
)


newdata <- validation[25,] %>% select(predictors)# bake(rec, validation[1,]) %>% make_keras_data %>% `[[`("x")
newdata_h2o <- validation_h2o[25,]

pb_xgb <- prediction_breakdown(explainer_xgb, observation = newdata)
pb_nn <- prediction_breakdown(explainer_nn, observation = newdata)
pb_rf <- prediction_breakdown(explainer_h2o_rf, observation = newdata_h2o)
pb_glm <- prediction_breakdown(explainer_glm, observation = newdata)
plot(pb_nn)
plot(pb_xgb)
plot(pb_rf)
plot(pb_glm)
plot(pb_xgb, pb_nn, pb_rf, pb_glm)


mp_nn <- model_performance(explainer_nn)
mp_xgb_model <- model_performance(explainer_xgb)
mp_glm <- model_performance(explainer_glm)
mp_h2o <- model_performance(explainer_h2o_rf)
plot(mp_xgb_model, mp_glm, mp_nn, mp_h2o)
plot(mp_xgb_model, mp_glm, mp_nn, mp_h2o, geom = "boxplot")


vi_nn <- variable_importance(explainer_nn, type = "ratio", n_sample = -1)
vi_xgb <- variable_importance(explainer_xgb, type = "ratio", n_sample = -1)
vi_glm1 <- variable_importance(explainer_glm, type = "ratio", n_sample = -1)
vi_h2o <- variable_importance(explainer_h2o_rf, type = "ratio", n_sample = -1)

# plot(vi_h2o)
# plot(vi_glm1)
plot(vi_xgb, vi_glm1, vi_nn, vi_h2o)

mpp_nn <- variable_response(explainer_nn, "risk_class", type = "factor")
mpp_xgb <- variable_response(explainer_xgb, "risk_class", type = "factor")
mpp_glm <- variable_response(explainer_glm, "risk_class", type = "factor")
mpp_h2o <- variable_response(explainer_h2o_rf, "risk_class", type = "factor")

plot(mpp_nn, mpp_xgb, mpp_glm, mpp_h2o)

plot(mpp_xgb)

h2o.shutdown(prompt=FALSE)