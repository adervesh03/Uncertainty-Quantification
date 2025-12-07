library(SoftBart)
library(ranger)
library(xgboost)



##########################################################################
# function: generate_data
# takes in: sample size n and seed 
# returns: simulated matrix, first column outcome, and the final five are covariates.
generate_data <- function(n, seed=NULL){
  sim_matrix <- matrix(NA, nrow = n, ncol = 6)
  if(!is.null(seed)) set.seed(seed) # if given seed
  
  # creating covariates
  X1 <- rnorm(n , mean = 5, sd = 1)
  X2 <- rnorm(n , mean = 9, sd = 1)
  X3 <- rnorm(n , mean = -3, sd = 1)
  X4 <- rnorm(n , mean = 2, sd = 1)
  X5 <- rnorm(n , mean = 1, sd = 1)
  
  # creating outcome, linear combination of covariates plus random noise
  y <- 2*X1 -3.1 * X2 + 4.7 * X3 + 0.5 * X4 + 1.3 * X5 + rnorm(n, mean = 0, sd = 1)
  
  sim_data <- data.frame(y, X1, X2, X3, X4, X5)
  return(sim_data)
}




##########################################################################
# function: generate_data_2 (nonlinear)
# takes in: sample size n and seed 
# returns: simulated matrix, first column outcome, and the final five are covariates.
generate_data_2 <- function(n, seed=NULL){
  sim_matrix <- matrix(NA, nrow = n, ncol = 6)
  if(!is.null(seed)) set.seed(seed) # if given seed
  
  # creating covariates
  X1 <- rnorm(n , mean = 5, sd = 1)
  X2 <- rnorm(n , mean = 9, sd = 1)
  X3 <- rnorm(n , mean = -3, sd = 1)
  X4 <- rnorm(n , mean = 2, sd = 1)
  X5 <- rnorm(n , mean = 1, sd = 1)
  
  # nonlinear outcome
  y <- 2*X1^3*X4 - 3.1*X2^2 + 4.7*sin(X3) + 0.5*X4*X5 + log(abs(X5)+1) + rnorm(n, 0, 1)
  
  sim_data <- data.frame(y, X1, X2, X3, X4, X5)
  return(sim_data)
}




##########################################################################
# function: split_conformal
# takes in: calibration Y, calibration prediction, test prediciton, alpha
# returns: conformal lower and upper levels and qhat.
split_conformal <- function(calib_Y,
                            calib_pred,
                            test_pred,
                            alpha = 0.05) {
  #absolute residuals
  resid <- abs(calib_Y - calib_pred)
  
  #quantile of residuals (the "correction")
  qhat <- quantile(resid, 1 - alpha, type = 8)
  
  # interval
  lower <- test_pred - qhat
  upper <- test_pred + qhat
  
  return(list(
    lower = lower, 
    upper = upper, 
    qhat = qhat))
}

##########################################################################
# function: model_fits
# takes in: data frame data, train/test/calibration split
# Bootstrap size B 
# returns: matrices containing uncertainty for each of our three models.
# note: first column in data frame data is y
model_fits <- function(data, train_idx, test_idx, calib_idx, B, verbose = FALSE) {
  # we will use B = 2000, and 
  n <- nrow(data)
  y <- data$y
  X <- data[, -1]
  
  # Subset train/test
  train_X <- X[train_idx, ]
  train_Y <- y[train_idx]
  calib_X <- X[calib_idx, ]
  calib_Y <- y[calib_idx]
  test_X  <- X[test_idx, ]
  
  test_n <- nrow(test_X)
  
  # bootstrap data setup
  softbart_uq <- matrix(NA, nrow = B, ncol = test_n)
  ranger_uq <- matrix(NA, nrow = B, ncol = test_n)
  xgboost_uq <- matrix(NA, nrow = B, ncol = test_n)
  
  # conformal data setup
  softbart_conf_lower <- numeric(test_n)
  softbart_conf_upper <- numeric(test_n)
  
  ranger_conf_lower <- numeric(test_n)
  ranger_conf_upper <- numeric(test_n)
  
  xgboost_conf_lower <- numeric(test_n)
  xgboost_conf_upper <- numeric(test_n)
  
  # prediction setup
  softbart_pred <- numeric(test_n)
  ranger_pred <- numeric(test_n)
  xgboost_pred <- numeric(test_n)
  
  ### ---- Softbart ----
  
  # Set up hyperparameters and MCMC options
  hypers <- Hypers(X = train_X, Y = train_Y, num_tree = 200)
  opts <- Opts(num_burn = B, num_save = B)
  
  # Run SoftBART
  softbart_time <- system.time({
    softbart_fit <- softbart(
      X = train_X,
      Y = train_Y,
      X_test = test_X,
      hypers = hypers,
      opts = opts,
      verbose = verbose
    )
    softbart_pred <- softbart_fit$y_hat_test_mean
    # Add residual uncertainty to get prediction draws
    for (b in 1:B) {
      softbart_uq[b, ] <- softbart_fit$y_hat_test[b, ] + rnorm(test_n, 0, softbart_fit$sigma[b])
    }
  })
  
  # calibration fit 
  softbart_fit_cal <- softbart(
    X = train_X,
    Y = train_Y,
    X_test = calib_X,
    hypers = hypers,
    opts = opts,
    verbose = verbose
  )
  
  softbart_pred_calib <- softbart_fit_cal$y_hat_test_mean
  softbart_conformal <- split_conformal(calib_Y = calib_Y,
                                        calib_pred = softbart_pred_calib,
                                        test_pred = softbart_pred)
  softbart_conf_lower <- softbart_conformal$lower
  softbart_conf_upper <- softbart_conformal$upper
  
  cat("Finished SoftBART \n")
  
  
  ### ---- Ranger (Random Forest) ----
  # Fit model
  ranger_time <- system.time({
    # Fit original model
    ranger_fit <- ranger(y ~ ., data = data[train_idx, ], num.trees = 500, num.threads = 1)
    
    ranger_pred <- predict(ranger_fit, data = test_X)$predictions
    
    # calibration fit 
    ranger_pred_calib <- predict(ranger_fit, data = calib_X)$predictions
    
    # Estimate residual standard deviation
    resid_sd <- sd(calib_Y - ranger_pred_calib)
    
    # Bootstrap predictions for uncertainty quantification
    for (b in 1:B) {
      n_train <- nrow(train_X)
      boot_idx <- sample(train_idx, replace = TRUE)
      boot_fit <- ranger(y ~ ., data = data[boot_idx, ], num.trees = 500)
      
      pred <- predict(boot_fit, data = test_X)$predictions
      # add residual noise
      pred <- pred + rnorm(length(pred), mean = 0, sd = resid_sd)
      
      ranger_uq[b, ] <- pred
    }
  })
  
  
  ranger_conformal <- split_conformal(calib_Y = calib_Y,
                                      calib_pred = ranger_pred_calib,
                                      test_pred = ranger_pred)
  ranger_conf_lower <- ranger_conformal$lower
  ranger_conf_upper <- ranger_conformal$upper
  
  cat("Finished ranger \n")
  
  ### ---- XGBoost ----
  xgboost_time <- system.time({
    dtrain <- xgb.DMatrix(data = as.matrix(train_X), label = train_Y)
    dtest  <- xgb.DMatrix(data = as.matrix(test_X))
    dcalib  <- xgb.DMatrix(data = as.matrix(calib_X))
    
    params <- list(
      objective = "reg:squarederror",
      eta = 0.1,
      max_depth = 4,
      subsample = 0.8,
      colsample_bytree = 0.8, 
      nthread = 1
    )
    
    # Fit original model to compute residual sd
    xgb_orig <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
    
    
    xgboost_pred <- predict(xgb_orig, dtest)
    # calibration fit 
    xgboost_pred_calib <- predict(xgb_orig, dcalib)
    
    resid_sd <- sd(calib_Y - xgboost_pred_calib)
    
    for (b in 1:B) {
      # Bootstrap resample
      n_train <- nrow(train_X)
      boot_idx <- sample(train_idx, replace = TRUE)
      dboot <- xgb.DMatrix(data = as.matrix(X[boot_idx, ]), label = y[boot_idx])
      
      fit <- xgb.train(params = params, data = dboot, nrounds = 100, verbose = 0)
      pred <- predict(fit, dtest)
      
      # add residual noise
      pred <- pred + rnorm(length(pred), mean = 0, sd = resid_sd)
      
      xgboost_uq[b, ] <- pred
    }
  })
  
  
  xgboost_conformal <- split_conformal(calib_Y = calib_Y,
                                       calib_pred = xgboost_pred_calib,
                                       test_pred = xgboost_pred)
  xgboost_conf_lower <- xgboost_conformal$lower
  xgboost_conf_upper <- xgboost_conformal$upper
  
  cat("Finished xgboost \n")
  
  return(list(
    softbart_uq = softbart_uq,
    ranger_uq = ranger_uq,
    xgboost_uq = xgboost_uq, 
    softbart_time = softbart_time, 
    ranger_time = ranger_time, 
    xgboost_time = xgboost_time,
    softbart_pred = softbart_pred, 
    ranger_pred = ranger_pred, 
    xgboost_pred = xgboost_pred, 
    softbart_conf_lower = softbart_conf_lower, 
    softbart_conf_upper = softbart_conf_upper, 
    ranger_conf_lower = ranger_conf_lower, 
    ranger_conf_upper = ranger_conf_upper,
    xgboost_conf_lower = xgboost_conf_lower, 
    xgboost_conf_upper = xgboost_conf_upper
  ))
  
}





#####################################################################################
# function: Compute Simulation
# takes in: sample size n, sim number J number of bootstrap replicates B,
# version (0 for linear, 1 for nonlinear), train_frac, calib_frac
# returns: simulation results list model_results
compute_simulation <- function(n, J, B, version, train_frac = 0.8, calib_frac = 0.25) {
  sim_seed = 70 + J
  set.seed(sim_seed)
  if (version == "linear") {
    sim_data  <- generate_data(n = n )
  } else if ( version == "nonlinear" ) {
    sim_data  <- generate_data_2(n = n )
  } else {
    return("choose a correct version")
  }
  
  # initial train/test split
  train_idx_initial <- sample(1:n, size = floor(train_frac * n), replace = FALSE)
  test_idx <- setdiff(1:n, train_idx_initial)
  
  # calibration split is taken from the initial training data
  calib_size <- floor(calib_frac * length(train_idx_initial))
  
  calib_idx <- sample(train_idx_initial, size = calib_size, replace = FALSE)
  
  # actual training indices
  train_idx <- setdiff(train_idx_initial, calib_idx)
  
  
  model_results <- model_fits(data = sim_data, train_idx = train_idx, 
                              test_idx = test_idx, calib_idx = calib_idx,
                              B = B)
  # performance code
  
  y <- sim_data$y[test_idx]
  
  # results extraction
  softbart_uq <- model_results$softbart_uq
  ranger_uq <- model_results$ranger_uq
  xgboost_uq <- model_results$xgboost_uq
  
  softbart_pred <- model_results$softbart_pred
  ranger_pred <- model_results$ranger_pred
  xgboost_pred <- model_results$xgboost_pred
  
  softbart_conf_lower <- model_results$softbart_conf_lower
  softbart_conf_upper <- model_results$softbart_conf_upper
  ranger_conf_lower <- model_results$ranger_conf_lower
  ranger_conf_upper <- model_results$ranger_conf_upper
  xgboost_conf_lower <- model_results$xgboost_conf_lower
  xgboost_conf_upper <- model_results$xgboost_conf_upper
  
  # bias
  softbart_boot_bias <- mean(colMeans(softbart_uq) - y)
  softbart_bias <- mean(softbart_pred - y)
  ranger_boot_bias <- mean(colMeans(ranger_uq) - y)
  ranger_bias <- mean(ranger_pred - y)
  xgboost_boot_bias <- mean(colMeans(xgboost_uq) - y)
  xgboost_bias <- mean(xgboost_pred - y)
  
  bias <- list(
    softbart_boot_bias = softbart_boot_bias, 
    softbart_bias = softbart_bias, 
    ranger_boot_bias = ranger_boot_bias,
    ranger_bias = ranger_bias, 
    xgboost_boot_bias = xgboost_boot_bias,
    xgboost_bias = xgboost_bias
  )
  
  # rmse 
  softbart_boot_rmse <- sqrt(mean((colMeans(softbart_uq) - y)^2))
  softbart_rmse <- sqrt(mean((softbart_pred - y)^2))
  ranger_boot_rmse <- sqrt(mean((colMeans(ranger_uq) - y)^2))
  ranger_rmse <- sqrt(mean((ranger_pred - y)^2))
  xgboost_boot_rmse <- sqrt(mean((colMeans(xgboost_uq) - y)^2))
  xgboost_rmse <- sqrt(mean((xgboost_pred - y)^2))
  
  rmse <- list (
    softbart_boot_rmse = softbart_boot_rmse, 
    softbart_rmse = softbart_rmse, 
    ranger_boot_rmse = ranger_boot_rmse,
    ranger_rmse = ranger_rmse, 
    xgboost_boot_rmse = xgboost_boot_rmse,
    xgboost_rmse = xgboost_rmse
  )
  
  
  # interval length 
  softbart_lower <- apply(softbart_uq, 2, quantile, 0.025)
  softbart_upper <- apply(softbart_uq, 2, quantile, 0.975)
  softbart_boot_int_length <- mean(softbart_upper - softbart_lower)
  softbart_conf_int_length <- mean(softbart_conf_upper - softbart_conf_lower)
  
  ranger_lower <- apply(ranger_uq, 2, quantile, 0.025)
  ranger_upper <- apply(ranger_uq, 2, quantile, 0.975)
  ranger_boot_int_length <- mean(ranger_upper - ranger_lower)
  ranger_conf_int_length <- mean(ranger_conf_upper - ranger_conf_lower)
  
  xgboost_lower <- apply(xgboost_uq, 2, quantile, 0.025)
  xgboost_upper <- apply(xgboost_uq, 2, quantile, 0.975)
  xgboost_boot_int_length <- mean(xgboost_upper - xgboost_lower)
  xgboost_conf_int_length <- mean(xgboost_conf_upper - xgboost_conf_lower)
  
  interval_length <- list (
    softbart_boot_int_length = softbart_boot_int_length,
    softbart_conf_int_length = softbart_conf_int_length,
    ranger_boot_int_length =  ranger_boot_int_length, 
    ranger_conf_int_length =  ranger_conf_int_length, 
    xgboost_boot_int_length = xgboost_boot_int_length,
    xgboost_conf_int_length = xgboost_conf_int_length
  )
  
  
  # coverage 
  softbart_boot_coverage <- mean(y >= softbart_lower & y <= softbart_upper)
  softbart_conf_coverage <- mean(y >= softbart_conf_lower & y <= softbart_conf_upper)
  ranger_boot_coverage <- mean(y >= ranger_lower & y <= ranger_upper)
  ranger_conf_coverage <- mean(y >= ranger_conf_lower & y <= ranger_conf_upper)
  xgboost_boot_coverage <- mean(y >= xgboost_lower & y <= xgboost_upper)
  xgboost_conf_coverage <- mean(y >= xgboost_conf_lower & y <= xgboost_conf_upper)
  
  coverage <- list(
    softbart_boot_coverage = softbart_boot_coverage, 
    softbart_conf_coverage = softbart_conf_coverage, 
    ranger_boot_coverage = ranger_boot_coverage, 
    ranger_conf_coverage = ranger_conf_coverage, 
    xgboost_boot_coverage = xgboost_boot_coverage,
    xgboost_conf_coverage = xgboost_conf_coverage
  )
  
  # time 
  time <- list( 
    softbart_time = model_results$softbart_time,
    ranger_time = model_results$ranger_time,
    xgboost_time = model_results$xgboost_time
  )
  
  return(list(
    bias = bias, 
    rmse = rmse, 
    interval_length = interval_length, 
    coverage = coverage,
    time = time
  ))
}

n = 250
sim_id <- as.numeric(commandArgs(TRUE))
# for local test
sim_id <- 5

linear_results <- compute_simulation(n = n, J = sim_id, B = 1000, version = "linear")
linear_filename <- paste0("linear_sim", sim_id, ".rds")
saveRDS(linear_results, file=linear_filename)

nonlinear_results <- compute_simulation(n = n, J = sim_id, B = 1000, version = "nonlinear")
nonlinear_filename <- paste0("nonlinear_sim", sim_id, ".rds")
saveRDS(nonlinear_results, file=nonlinear_filename)