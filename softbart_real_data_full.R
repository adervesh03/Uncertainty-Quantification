library(SoftBart)
library(ranger)
library(xgboost)
library(ggplot2)
library(patchwork)


##=============================================================
## 1. Load data & specify predictors / target
##=============================================================
data <- read.csv("concrete_data.csv")


predictors <- c("cement", "blast_furnace_slag", "fly_ash", "water",
                "superplasticizer", "coarse_aggregate", "fine_aggregate", "age")
target <- "concrete_compressive_strength"

data_model <- na.omit(data[, c(target, predictors)])
##=============================================================
## 2. Main function: fit models + uncertainty + metrics
##=============================================================

model_fits_real <- function(data, predictors, target,
                            train_frac = 0.8, B = 1000,
                            verbose = TRUE, seed = 123) {
  if (!is.null(seed)) set.seed(seed)

  # Make sure data has only target + predictors
  df <- data[, c(target, predictors)]
  df <- na.omit(df)

  y <- df[[target]]
  X <- df[, predictors, drop = FALSE]

  n <- nrow(df)
  train_idx <- sample(1:n, size = floor(train_frac * n), replace = FALSE)
  test_idx  <- setdiff(1:n, train_idx)

  train_X <- X[train_idx, , drop = FALSE]
  train_Y <- y[train_idx]
  test_X  <- X[test_idx, , drop = FALSE]
  test_Y  <- y[test_idx]

  test_n <- length(test_Y)

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

  ranger_resid_sd <- sd(train_Y - predict(ranger_fit, data = data.frame(train_X))$predictions)

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

  xgb_resid_sd <- sd(train_Y - predict(xgb_orig, dtrain))

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
  # Posterior/ bootstrap mean predictions
  softbart_mean <- colMeans(softbart_uq)
  ranger_mean   <- colMeans(ranger_uq)
  xgboost_mean  <- colMeans(xgboost_uq)

  # Bias (mean error)
  softbart_bias <- mean(softbart_mean - test_Y)
  ranger_bias   <- mean(ranger_mean   - test_Y)
  xgboost_bias  <- mean(xgboost_mean  - test_Y)

  # RMSE
  softbart_rmse <- sqrt(mean((softbart_mean - test_Y)^2))
  ranger_rmse   <- sqrt(mean((ranger_mean   - test_Y)^2))
  xgboost_rmse  <- sqrt(mean((xgboost_mean  - test_Y)^2))

  # 95% predictive interval length
  softbart_lower <- apply(softbart_uq, 2, quantile, 0.025)
  softbart_upper <- apply(softbart_uq, 2, quantile, 0.975)
  softbart_int_length <- mean(softbart_upper - softbart_lower)

  ranger_lower <- apply(ranger_uq, 2, quantile, 0.025)
  ranger_upper <- apply(ranger_uq, 2, quantile, 0.975)
  ranger_int_length <- mean(ranger_upper - ranger_lower)

  xgboost_lower <- apply(xgboost_uq, 2, quantile, 0.025)
  xgboost_upper <- apply(xgboost_uq, 2, quantile, 0.975)
  xgboost_int_length <- mean(xgboost_upper - xgboost_lower)

  # Coverage (does interval contain observed y?)
  softbart_coverage <- mean(test_Y >= softbart_lower & test_Y <= softbart_upper)
  ranger_coverage   <- mean(test_Y >= ranger_lower   & test_Y <= ranger_upper)
  xgboost_coverage  <- mean(test_Y >= xgboost_lower  & test_Y <= xgboost_upper)

  list(
    test_idx = test_idx,
    test_Y   = test_Y,
    softbart_uq = softbart_uq,
    ranger_uq   = ranger_uq,
    xgboost_uq  = xgboost_uq,
    metrics = list(
      bias = c(
        SoftBART = softbart_bias,
        Ranger   = ranger_bias,
        XGBoost  = xgboost_bias
      ),
      rmse = c(
        SoftBART = softbart_rmse,
        Ranger   = ranger_rmse,
        XGBoost  = xgboost_rmse
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
  )
}

##=============================================================
## 4. Run the analysis (one split, B = 1000)
##=============================================================

set.seed(2025)
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

metrics <- real_results$metrics

bias_df <- data.frame(
  model = names(metrics$bias),
  bias  = abs(as.numeric(metrics$bias))
)

rmse_df <- data.frame(
  model = names(metrics$rmse),
  rmse  = as.numeric(metrics$rmse)
)

interval_df <- data.frame(
  model          = names(metrics$interval_length),
  interval_length = as.numeric(metrics$interval_length)
)

coverage_df <- data.frame(
  model    = names(metrics$coverage),
  coverage = as.numeric(metrics$coverage)
)

bias_plot <- ggplot(bias_df, aes(x = model, y = bias)) +
  geom_col(fill = "skyblue") +
  geom_text(aes(label = round(bias, 2)), vjust = -0.5, size = 3) +
  labs(title = "Mean Absolute Bias on Test Set", x = "Model", y = "Bias") +
  theme_minimal(base_size = 14)

rmse_plot <- ggplot(rmse_df, aes(x = model, y = rmse)) +
  geom_col(fill = "orange") +
  geom_text(aes(label = round(rmse, 2)), vjust = -0.5, size = 3) +
  labs(title = "RMSE on Test Set", x = "Model", y = "RMSE") +
  theme_minimal(base_size = 14)

interval_plot <- ggplot(interval_df, aes(x = model, y = interval_length)) +
  geom_col(fill = "seagreen3") +
  geom_text(aes(label = round(interval_length, 2)), vjust = -0.5, size = 3) +
  labs(title = "Mean 95% Predictive Interval Length", x = "Model", y = "Interval Length") +
  theme_minimal(base_size = 14)

coverage_plot <- ggplot(coverage_df, aes(x = model, y = coverage)) +
  geom_col(fill = "plum") +
  geom_text(aes(label = round(coverage, 2)), vjust = -0.5, size = 3) +
  labs(title = "95% Predictive Interval Coverage", x = "Model", y = "Coverage") +
  theme_minimal(base_size = 14)

combined_real_plot <- bias_plot + rmse_plot + interval_plot + coverage_plot
print(combined_real_plot)

