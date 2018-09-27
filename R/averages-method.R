# no machine learning here, just taking historical averages
averages_summary <- training %>%
  group_by(gender, issue_age, face_amount,
           post_level_premium_structure, premium_jump_ratio,
           risk_class, premium_mode) %>%
  summarize(predicted_count_rate = sum(lapse_count) / sum(exposure_count),
            predicted_amount_rate = sum(lapse_amount) / sum(exposure_amount)) %>%
  right_join(validation)
glimpse(averages_summary)
metrics(averages_summary, lapse_count_rate, predicted_count_rate)
# # A tibble: 1 x 3
# rmse   rsq   mae
# <dbl> <dbl> <dbl>
#1 0.303 0.282 0.185
