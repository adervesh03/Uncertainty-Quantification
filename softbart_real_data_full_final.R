library(SoftBart)
library(ranger)
library(xgboost)

model_fits_real <- function(data, predictors, target,
                            train_frac = 0.8, B = 1000,
                            calib_frac = 0.2,   # NEW: fraction of training used for calibration
                            verbose = TRUE, seed = 123) {
  if (!is.null(seed)) set.seed(seed)

  # Ensure only target + predictors
  df <- data[, c(target, predictors)]
  df <- na.omit(df)

  y <- df[[target]]
  X <- df[, predictors, drop = FALSE]

  n <- nrow(df)

  ##-----------------------------------------------------------
  ## 1. Train / Test / Calibration split
  ##-----------------------------------------------------------
  # First: train vs test
  train_idx <- sample(1:n, size = floor(train_frac * n), replace = FALSE)
  test_idx  <- setdiff(1:n, train_idx)

  # Second: within training, split into train_main and calibration
  n_train_total <- length(train_idx)
  calib_n <- max(1, floor(calib_frac * n_train_total))  # at least 1 obs

  calib_idx <- sample(train_idx, size = calib_n, replace = FALSE)
  train_main_idx <- setdiff(train_idx, calib_idx)

  # Main training data (used to fit models)
  train_X <- X[train_main_idx, , drop = FALSE]
  train_Y <- y[train_main_idx]

  # Calibration data (used ONLY to estimate residual SD)
  calib_X <- X[calib_idx, , drop = FALSE]
  calib_Y <- y[calib_idx]

  # Test data (for final evaluation and UQ)
  test_X  <- X[test_idx, , drop = FALSE]
  test_Y  <- y[test_idx]

  test_n <- length(test_Y)

  # Matrices for predictive draws (UQ)
  softbart_uq <- matrix(NA, nrow = B, ncol = test_n)
  ranger_uq   <- matrix(NA, nrow = B, ncol = test_n)
  xgboost_uq  <- matrix(NA, nrow = B, ncol = test_n)

  ##-----------------------------------------------------------
  ## SoftBART
  ##-----------------------------------------------------------
  if (verbose) cat("Fitting SoftBART...\n")

  hypers <- Hypers(X = as.matrix(train_X), Y = train_Y, num_tree = 200)
  opts   <- Opts(num_burn = B, num_save = B)

  softbart_fit <- softbart(
    X = as.matrix(train_X),
    Y = train_Y,
    X_test = as.matrix(test_X),
    hypers = hypers,
    opts   = opts,
    verbose = verbose
  )

  # Raw prediction (posterior mean of f(x))
  softbart_raw_pred <- colMeans(softbart_fit$y_hat_test)

  # Posterior predictive draws (f(x) + noise)
  for (b in 1:B) {
    softbart_uq[b, ] <- softbart_fit$y_hat_test[b, ] +
      rnorm(test_n, 0, softbart_fit$sigma[b])
  }

  if (verbose) cat("Finished SoftBART\n")

  ##-----------------------------------------------------------
  ## Ranger (Random Forest) + bootstrap
  ##-----------------------------------------------------------
  if (verbose) cat("Fitting ranger + bootstrap...\n")

  ranger_fit <- ranger(
    formula = y ~ .,
    data = data.frame(y = train_Y, train_X),
    num.trees = 500,
    num.threads = 1
  )

  # Raw prediction
  ranger_raw_pred <- predict(ranger_fit, data = data.frame(test_X))$predictions

  # Residual SD for noise in predictive draws (NOW from calibration set)
  ranger_resid_sd <- sd(
    calib_Y - predict(ranger_fit, data = data.frame(calib_X))$predictions
  )

  for (b in 1:B) {
    n_train <- nrow(train_X)
    boot_idx_local <- sample(1:n_train, replace = TRUE)

    boot_data <- data.frame(
      y = train_Y[boot_idx_local],
      train_X[boot_idx_local, , drop = FALSE]
    )

    boot_fit <- ranger(
      y ~ .,
      data = boot_data,
      num.trees = 500,
      num.threads = 1
    )

    pred <- predict(boot_fit, data = data.frame(test_X))$predictions
    pred <- pred + rnorm(length(pred), 0, ranger_resid_sd)

    ranger_uq[b, ] <- pred
  }

  if (verbose) cat("Finished ranger\n")

  ##-----------------------------------------------------------
  ## XGBoost + bootstrap
  ##-----------------------------------------------------------
  if (verbose) cat("Fitting xgboost + bootstrap...\n")

  dtrain <- xgb.DMatrix(data = as.matrix(train_X), label = train_Y)
  dtest  <- xgb.DMatrix(data = as.matrix(test_X))
  dcalib <- xgb.DMatrix(data = as.matrix(calib_X), label = calib_Y)  # NEW

  params <- list(
    objective = "reg:squarederror",
    eta = 0.1,
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = 1
  )

  xgb_orig <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 100,
    verbose = 0
  )

  # Raw prediction
  xgboost_raw_pred <- predict(xgb_orig, dtest)

  # Residual SD for predictive draws (NOW from calibration set)
  xgb_resid_sd <- sd(calib_Y - predict(xgb_orig, dcalib))

  for (b in 1:B) {
    n_train <- nrow(train_X)
    boot_idx_local <- sample(1:n_train, replace = TRUE)

    dboot <- xgb.DMatrix(
      data  = as.matrix(train_X[boot_idx_local, , drop = FALSE]),
      label = train_Y[boot_idx_local]
    )

    fit <- xgb.train(
      params = params,
      data   = dboot,
      nrounds = 100,
      verbose = 0
    )

    pred <- predict(fit, dtest)
    pred <- pred + rnorm(length(pred), 0, xgb_resid_sd)

    xgboost_uq[b, ] <- pred
  }

  if (verbose) cat("Finished xgboost\n")

  ##===========================================================
  ## 3. Compute metrics on test set
  ##===========================================================

  models <- c("SoftBART", "Ranger", "XGBoost")

  # Posterior / bootstrap mean predictions (UQ)
  softbart_mean_uq <- colMeans(softbart_uq)
  ranger_mean_uq   <- colMeans(ranger_uq)
  xgboost_mean_uq  <- colMeans(xgboost_uq)

  #-----------------------------
  # Raw bias and RMSE
  #-----------------------------
  softbart_bias_raw <- mean(softbart_raw_pred - test_Y)
  ranger_bias_raw   <- mean(ranger_raw_pred   - test_Y)
  xgboost_bias_raw  <- mean(xgboost_raw_pred  - test_Y)

  softbart_rmse_raw <- sqrt(mean((softbart_raw_pred - test_Y)^2))
  ranger_rmse_raw   <- sqrt(mean((ranger_raw_pred   - test_Y)^2))
  xgboost_rmse_raw  <- sqrt(mean((xgboost_raw_pred  - test_Y)^2))

  #-----------------------------
  # UQ bias and RMSE
  #-----------------------------
  softbart_bias_uq <- mean(softbart_mean_uq - test_Y)
  ranger_bias_uq   <- mean(ranger_mean_uq   - test_Y)
  xgboost_bias_uq  <- mean(xgboost_mean_uq  - test_Y)

  softbart_rmse_uq <- sqrt(mean((softbart_mean_uq - test_Y)^2))
  ranger_rmse_uq   <- sqrt(mean((ranger_mean_uq   - test_Y)^2))
  xgboost_rmse_uq  <- sqrt(mean((xgboost_mean_uq  - test_Y)^2))

  #-----------------------------
  # Predictive interval length + coverage (UQ only)
  #-----------------------------
  softbart_lower <- apply(softbart_uq, 2, quantile, 0.025)
  softbart_upper <- apply(softbart_uq, 2, quantile, 0.975)
  softbart_int_length <- mean(softbart_upper - softbart_lower)
  softbart_coverage   <- mean(test_Y >= softbart_lower & test_Y <= softbart_upper)

  ranger_lower <- apply(ranger_uq, 2, quantile, 0.025)
  ranger_upper <- apply(ranger_uq, 2, quantile, 0.975)
  ranger_int_length <- mean(ranger_upper - ranger_lower)
  ranger_coverage   <- mean(test_Y >= ranger_lower   & test_Y <= ranger_upper)

  xgboost_lower <- apply(xgboost_uq, 2, quantile, 0.025)
  xgboost_upper <- apply(xgboost_uq, 2, quantile, 0.975)
  xgboost_int_length <- mean(xgboost_upper - xgboost_lower)
  xgboost_coverage   <- mean(test_Y >= xgboost_lower  & test_Y <= xgboost_upper)

  metrics <- list(
    bias_raw = c(
      SoftBART = softbart_bias_raw,
      Ranger   = ranger_bias_raw,
      XGBoost  = xgboost_bias_raw
    ),
    bias_uq = c(
      SoftBART = softbart_bias_uq,
      Ranger   = ranger_bias_uq,
      XGBoost  = xgboost_bias_uq
    ),
    rmse_raw = c(
      SoftBART = softbart_rmse_raw,
      Ranger   = ranger_rmse_raw,
      XGBoost  = xgboost_rmse_raw
    ),
    rmse_uq = c(
      SoftBART = softbart_rmse_uq,
      Ranger   = ranger_rmse_uq,
      XGBoost  = xgboost_rmse_uq
    ),
    interval_length = c(
      SoftBART = softbart_int_length,
      Ranger   = ranger_int_length,
      XGBoost  = xgboost_int_length
    ),
    coverage = c(
      SoftBART = softbart_coverage,
      Ranger   = ranger_coverage,
      XGBoost  = xgboost_coverage
    )
  )

  list(
    test_idx      = test_idx,
    test_Y        = test_Y,
    softbart_uq   = softbart_uq,
    ranger_uq     = ranger_uq,
    xgboost_uq    = xgboost_uq,
    softbart_raw_pred = softbart_raw_pred,
    ranger_raw_pred   = ranger_raw_pred,
    xgboost_raw_pred  = xgboost_raw_pred,
    metrics       = metrics
  )
}


##=============================================================
## 4. Run the analysis (one split, B = 1000)
##=============================================================



set.seed(2025)
data <- read.csv("concrete_data.csv")


predictors <- c("cement", "blast_furnace_slag", "fly_ash", "water",
                "superplasticizer", "coarse_aggregate", "fine_aggregate", "age")
target <- "concrete_compressive_strength"

data_model <- na.omit(data[, c(target, predictors)])


real_results <- model_fits_real(
  data      = data_model,
  predictors = predictors,
  target     = target,
  train_frac = 0.8,
  B         = 1000,
  verbose   = TRUE,
  seed      = 2025
)

real_results$metrics  # quick numeric summary

##=============================================================
## 5. Plots (similar style to your simulation)
##=============================================================

library(ggplot2)
library(patchwork)

metrics <- real_results$metrics

# Long-format data for bias (absolute bias) and RMSE
bias_df <- data.frame(
  model = rep(names(metrics$bias_raw), 2),
  type  = rep(c("Raw", "UQ"), each = length(metrics$bias_raw)),
  value = c(abs(as.numeric(metrics$bias_raw)),
            abs(as.numeric(metrics$bias_uq)))
)

rmse_df <- data.frame(
  model = rep(names(metrics$rmse_raw), 2),
  type  = rep(c("Raw", "UQ"), each = length(metrics$rmse_raw)),
  value = c(as.numeric(metrics$rmse_raw),
            as.numeric(metrics$rmse_uq))
)

bias_plot <- ggplot(bias_df, aes(x = model, y = value, fill = type)) +
  geom_col(position = position_dodge(width = 0.7)) +
  geom_text(aes(label = round(value, 2)),
            position = position_dodge(width = 0.7),
            vjust = -0.5, size = 3) +
  labs(title = "Mean Absolute Bias (Raw vs UQ)",
       x = "Model", y = "Absolute Bias") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top")

rmse_plot <- ggplot(rmse_df, aes(x = model, y = value, fill = type)) +
  geom_col(position = position_dodge(width = 0.7)) +
  geom_text(aes(label = round(value, 2)),
            position = position_dodge(width = 0.7),
            vjust = -0.5, size = 3) +
  labs(title = "RMSE (Raw vs UQ)",
       x = "Model", y = "RMSE") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top")

# First combined figure: Bias + RMSE
fig1 <- bias_plot + rmse_plot
print(fig1)

##### visual two


interval_df <- data.frame(
  model          = names(metrics$interval_length),
  interval_length = as.numeric(metrics$interval_length)
)

coverage_df <- data.frame(
  model    = names(metrics$coverage),
  coverage = as.numeric(metrics$coverage)
)

interval_plot <- ggplot(interval_df, aes(x = model, y = interval_length)) +
  geom_col(fill = "steelblue") +
  geom_text(aes(label = round(interval_length, 2)),
            vjust = -0.5, size = 3) +
  labs(title = "Mean 95% Predictive Interval Length",
       x = "Model", y = "Interval Length") +
  theme_minimal(base_size = 14)

coverage_plot <- ggplot(coverage_df, aes(x = model, y = coverage)) +
  geom_col(fill = "darkseagreen") +
  geom_text(aes(label = round(coverage, 2)),
            vjust = -0.5, size = 3) +
  labs(title = "95% Predictive Interval Coverage",
       x = "Model", y = "Coverage") +
  theme_minimal(base_size = 14) +
  ylim(0, 1)  # since coverage is a proportion

# Second combined figure: Interval length + Coverage
fig2 <- interval_plot + coverage_plot
print(fig2)

