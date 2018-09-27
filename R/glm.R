glm1 <- glm(
  lapse_count_rate ~ gender + issue_age + face_amount + post_level_premium_structure +
    premium_jump_ratio + risk_class + premium_mode,
  family = gaussian,
  data = training
)

glm_summary <- bind_cols(
  validation,
  data.frame(predicted_count_rate = predict(glm1, validation))
)
metrics(glm_summary, lapse_count_rate, predicted_count_rate)
# # A tibble: 1 x 3
# rmse   rsq   mae
# <dbl> <dbl> <dbl>
#1 0.269 0.367 0.199

