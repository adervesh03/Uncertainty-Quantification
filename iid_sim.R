library(SoftBart)
library(ranger)
library(xgboost)
library(ggplot2)
library(patchwork)


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


################################################################################################
##***************************************************************************************#######
################################################################################################


run_study <- function(n, B = 1000, J = 50) {
  # initialize "vector of lists"
  sim_results_linear <- vector("list", J)
  sim_results_nonlinear <- vector("list", J)
  
  for (j in seq_len(J)) {
    cat("Running replication", j, "of", J, "\n")
    
    sim_results_linear[[j]] <- compute_simulation(
      n = n, J =j, B = B, version = "linear"
    )
    
    sim_results_nonlinear[[j]] <- compute_simulation(
      n = n, J = j, B = B, version = "nonlinear"
    )
  }
  

  return(list(sim_results_linear = sim_results_linear,
               sim_results_nonlinear = sim_results_nonlinear))
}

####################################################################################
# RESULTS STORAGE AND EXTRACTION (UPDATED)
####################################################################################

plots <- function(results) {
  sim_results_linear <- results$sim_results_linear
  sim_results_nonlinear <- results$sim_results_nonlinear
  
  # --- Initialize storage vectors for all metrics ---

  # Bias
  softbart_bias_linear <- numeric(J); softbart_bias_nonlinear <- numeric(J)
  softbart_boot_bias_linear <- numeric(J); softbart_boot_bias_nonlinear <- numeric(J)
  
  ranger_bias_linear <- numeric(J); ranger_bias_nonlinear <- numeric(J)
  ranger_boot_bias_linear <- numeric(J); ranger_boot_bias_nonlinear <- numeric(J)
  
  xgboost_bias_linear <- numeric(J); xgboost_bias_nonlinear <- numeric(J)
  xgboost_boot_bias_linear <- numeric(J); xgboost_boot_bias_nonlinear <- numeric(J)
  
  # RMSE
  softbart_rmse_linear <- numeric(J); softbart_rmse_nonlinear <- numeric(J)
  softbart_boot_rmse_linear <- numeric(J); softbart_boot_rmse_nonlinear <- numeric(J)
  
  ranger_rmse_linear <- numeric(J); ranger_rmse_nonlinear <- numeric(J)
  ranger_boot_rmse_linear <- numeric(J); ranger_boot_rmse_nonlinear <- numeric(J)
  
  xgboost_rmse_linear <- numeric(J); xgboost_rmse_nonlinear <- numeric(J)
  xgboost_boot_rmse_linear <- numeric(J); xgboost_boot_rmse_nonlinear <- numeric(J)
  
  # Interval lengths
  softbart_boot_int_linear <- numeric(J); softbart_conf_int_linear <- numeric(J)
  softbart_boot_int_nonlinear <- numeric(J); softbart_conf_int_nonlinear <- numeric(J)
  
  ranger_boot_int_linear <- numeric(J); ranger_conf_int_linear <- numeric(J)
  ranger_boot_int_nonlinear <- numeric(J); ranger_conf_int_nonlinear <- numeric(J)
  
  xgboost_boot_int_linear <- numeric(J); xgboost_conf_int_linear <- numeric(J)
  xgboost_boot_int_nonlinear <- numeric(J); xgboost_conf_int_nonlinear <- numeric(J)
  
  # Coverage
  softbart_boot_cov_linear <- numeric(J); softbart_conf_cov_linear <- numeric(J)
  softbart_boot_cov_nonlinear <- numeric(J); softbart_conf_cov_nonlinear <- numeric(J)
  
  ranger_boot_cov_linear <- numeric(J); ranger_conf_cov_linear <- numeric(J)
  ranger_boot_cov_nonlinear <- numeric(J); ranger_conf_cov_nonlinear <- numeric(J)
  
  xgboost_boot_cov_linear <- numeric(J); xgboost_conf_cov_linear <- numeric(J)
  xgboost_boot_cov_nonlinear <- numeric(J); xgboost_conf_cov_nonlinear <- numeric(J)
  
  # Time
  softbart_time_linear <- numeric(J); softbart_time_nonlinear <- numeric(J)
  ranger_time_linear <- numeric(J); ranger_time_nonlinear <- numeric(J)
  xgboost_time_linear <- numeric(J); xgboost_time_nonlinear <- numeric(J)
  
  # --- Extract metrics from results ---
  for (j in 1:J) {
    
    lin <- sim_results_linear[[j]]
    nl  <- sim_results_nonlinear[[j]]
    
    # Bias
    softbart_bias_linear[j] <- lin$bias$softbart_bias
    softbart_boot_bias_linear[j] <- lin$bias$softbart_boot_bias
    softbart_bias_nonlinear[j] <- nl$bias$softbart_bias
    softbart_boot_bias_nonlinear[j] <- nl$bias$softbart_boot_bias
    
    ranger_bias_linear[j] <- lin$bias$ranger_bias
    ranger_boot_bias_linear[j] <- lin$bias$ranger_boot_bias
    ranger_bias_nonlinear[j] <- nl$bias$ranger_bias
    ranger_boot_bias_nonlinear[j] <- nl$bias$ranger_boot_bias
    
    xgboost_bias_linear[j] <- lin$bias$xgboost_bias
    xgboost_boot_bias_linear[j] <- lin$bias$xgboost_boot_bias
    xgboost_bias_nonlinear[j] <- nl$bias$xgboost_bias
    xgboost_boot_bias_nonlinear[j] <- nl$bias$xgboost_boot_bias
    
    # RMSE
    softbart_rmse_linear[j] <- lin$rmse$softbart_rmse
    softbart_boot_rmse_linear[j] <- lin$rmse$softbart_boot_rmse
    softbart_rmse_nonlinear[j] <- nl$rmse$softbart_rmse
    softbart_boot_rmse_nonlinear[j] <- nl$rmse$softbart_boot_rmse
    
    ranger_rmse_linear[j] <- lin$rmse$ranger_rmse
    ranger_boot_rmse_linear[j] <- lin$rmse$ranger_boot_rmse
    ranger_rmse_nonlinear[j] <- nl$rmse$ranger_rmse
    ranger_boot_rmse_nonlinear[j] <- nl$rmse$ranger_boot_rmse
    
    xgboost_rmse_linear[j] <- lin$rmse$xgboost_rmse
    xgboost_boot_rmse_linear[j] <- lin$rmse$xgboost_boot_rmse
    xgboost_rmse_nonlinear[j] <- nl$rmse$xgboost_rmse
    xgboost_boot_rmse_nonlinear[j] <- nl$rmse$xgboost_boot_rmse
    
    # Interval lengths
    softbart_boot_int_linear[j] <- lin$interval_length$softbart_boot_int_length
    softbart_conf_int_linear[j] <- lin$interval_length$softbart_conf_int_length
    softbart_boot_int_nonlinear[j] <- nl$interval_length$softbart_boot_int_length
    softbart_conf_int_nonlinear[j] <- nl$interval_length$softbart_conf_int_length
    
    ranger_boot_int_linear[j] <- lin$interval_length$ranger_boot_int_length
    ranger_conf_int_linear[j] <- lin$interval_length$ranger_conf_int_length
    ranger_boot_int_nonlinear[j] <- nl$interval_length$ranger_boot_int_length
    ranger_conf_int_nonlinear[j] <- nl$interval_length$ranger_conf_int_length
    
    xgboost_boot_int_linear[j] <- lin$interval_length$xgboost_boot_int_length
    xgboost_conf_int_linear[j] <- lin$interval_length$xgboost_conf_int_length
    xgboost_boot_int_nonlinear[j] <- nl$interval_length$xgboost_boot_int_length
    xgboost_conf_int_nonlinear[j] <- nl$interval_length$xgboost_conf_int_length
    
    # Coverage
    softbart_boot_cov_linear[j] <- lin$coverage$softbart_boot_coverage
    softbart_conf_cov_linear[j] <- lin$coverage$softbart_conf_coverage
    softbart_boot_cov_nonlinear[j] <- nl$coverage$softbart_boot_coverage
    softbart_conf_cov_nonlinear[j] <- nl$coverage$softbart_conf_coverage
    
    ranger_boot_cov_linear[j] <- lin$coverage$ranger_boot_coverage
    ranger_conf_cov_linear[j] <- lin$coverage$ranger_conf_coverage
    ranger_boot_cov_nonlinear[j] <- nl$coverage$ranger_boot_coverage
    ranger_conf_cov_nonlinear[j] <- nl$coverage$ranger_conf_coverage
    
    xgboost_boot_cov_linear[j] <- lin$coverage$xgboost_boot_coverage
    xgboost_conf_cov_linear[j] <- lin$coverage$xgboost_conf_coverage
    xgboost_boot_cov_nonlinear[j] <- nl$coverage$xgboost_boot_coverage
    xgboost_conf_cov_nonlinear[j] <- nl$coverage$xgboost_conf_coverage
    
    # Time
    softbart_time_linear[j] <- lin$time$softbart_time[["elapsed"]]
    softbart_time_nonlinear[j] <- nl$time$softbart_time[["elapsed"]]
    
    ranger_time_linear[j] <- lin$time$ranger_time[["elapsed"]]
    ranger_time_nonlinear[j] <- nl$time$ranger_time[["elapsed"]]
    
    xgboost_time_linear[j] <- lin$time$xgboost_time[["elapsed"]]
    xgboost_time_nonlinear[j] <- nl$time$xgboost_time[["elapsed"]]
  }
  
  ####################################################################################
  # CREATE DATA FRAMES FOR PLOTTING
  ####################################################################################
  
  # Bias
  bias_linear_df <- data.frame(
    model = c("SoftBART","Ranger","XGBoost"),
    bias = c(mean(abs(softbart_bias_linear)), mean(abs(ranger_bias_linear)), mean(abs(xgboost_bias_linear))),
    boot_bias = c(mean(abs(softbart_boot_bias_linear)), mean(abs(ranger_boot_bias_linear)), mean(abs(xgboost_boot_bias_linear)))
  )
  bias_nonlinear_df <- data.frame(
    model = c("SoftBART","Ranger","XGBoost"),
    bias = c(mean(abs(softbart_bias_nonlinear)), mean(abs(ranger_bias_nonlinear)), mean(abs(xgboost_bias_nonlinear))),
    boot_bias = c(mean(abs(softbart_boot_bias_nonlinear)), mean(abs(ranger_boot_bias_nonlinear)), mean(abs(xgboost_boot_bias_nonlinear)))
  )
  
  # RMSE
  rmse_linear_df <- data.frame(
    model = c("SoftBART","Ranger","XGBoost"),
    rmse = c(mean(softbart_rmse_linear), mean(ranger_rmse_linear), mean(xgboost_rmse_linear)),
    boot_rmse = c(mean(softbart_boot_rmse_linear), mean(ranger_boot_rmse_linear), mean(xgboost_boot_rmse_linear))
  )
  rmse_nonlinear_df <- data.frame(
    model = c("SoftBART","Ranger","XGBoost"),
    rmse = c(mean(softbart_rmse_nonlinear), mean(ranger_rmse_nonlinear), mean(xgboost_rmse_nonlinear)),
    boot_rmse = c(mean(softbart_boot_rmse_nonlinear), mean(ranger_boot_rmse_nonlinear), mean(xgboost_boot_rmse_nonlinear))
  )
  
  # Interval lengths
  interval_linear_df <- data.frame(
    model = rep(c("SoftBART","Ranger","XGBoost"), each = 2),
    type = rep(c("Bootstrap","Conformal"), times = 3),
    interval = c(
      mean(softbart_boot_int_linear), mean(softbart_conf_int_linear),
      mean(ranger_boot_int_linear), mean(ranger_conf_int_linear),
      mean(xgboost_boot_int_linear), mean(xgboost_conf_int_linear)
    )
  )
  interval_nonlinear_df <- data.frame(
    model = rep(c("SoftBART","Ranger","XGBoost"), each = 2),
    type = rep(c("Bootstrap","Conformal"), times = 3),
    interval = c(
      mean(softbart_boot_int_nonlinear), mean(softbart_conf_int_nonlinear),
      mean(ranger_boot_int_nonlinear), mean(ranger_conf_int_nonlinear),
      mean(xgboost_boot_int_nonlinear), mean(xgboost_conf_int_nonlinear)
    )
  )
  
  # Coverage
  coverage_linear_df <- data.frame(
    model = rep(c("SoftBART","Ranger","XGBoost"), each = 2),
    type = rep(c("Bootstrap","Conformal"), times = 3),
    coverage = c(
      mean(softbart_boot_cov_linear), mean(softbart_conf_cov_linear),
      mean(ranger_boot_cov_linear), mean(ranger_conf_cov_linear),
      mean(xgboost_boot_cov_linear), mean(xgboost_conf_cov_linear)
    )
  )
  coverage_nonlinear_df <- data.frame(
    model = rep(c("SoftBART","Ranger","XGBoost"), each = 2),
    type = rep(c("Bootstrap","Conformal"), times = 3),
    coverage = c(
      mean(softbart_boot_cov_nonlinear), mean(softbart_conf_cov_nonlinear),
      mean(ranger_boot_cov_nonlinear), mean(ranger_conf_cov_nonlinear),
      mean(xgboost_boot_cov_nonlinear), mean(xgboost_conf_cov_nonlinear)
    )
  )
  
  # Time
  time_linear_df <- data.frame(
    model = c("SoftBART","Ranger","XGBoost"),
    time = c(mean(softbart_time_linear), mean(ranger_time_linear), mean(xgboost_time_linear))
  )
  time_nonlinear_df <- data.frame(
    model = c("SoftBART","Ranger","XGBoost"),
    time = c(mean(softbart_time_nonlinear), mean(ranger_time_nonlinear), mean(xgboost_time_nonlinear))
  )
  
  ### === BIAS PLOTS === ###
  bias_linear_plot <- ggplot(bias_linear_df, aes(x = model, y = bias)) +
    geom_col(fill = "skyblue") +
    geom_text(aes(label = round(bias, 2)), vjust = -0.5, size = 4) +
    labs(title = "Mean Absolute Bias (Linear DGP)", x = "Model", y = "Bias") +
    theme_minimal(base_size = 14)
  
  bias_linear_boot_plot <- ggplot(bias_linear_df, aes(x = model, y = boot_bias)) +
    geom_col(fill = "deepskyblue4") +
    geom_text(aes(label = round(boot_bias, 2)), vjust = -0.5, size = 4) +
    labs(title = "Mean Absolute Bias (Linear DGP) — Bootstrap", x = "Model", y = "Bootstrap Bias") +
    theme_minimal(base_size = 14)
  
  bias_nonlinear_plot <- ggplot(bias_nonlinear_df, aes(x = model, y = bias)) +
    geom_col(fill = "skyblue") +
    geom_text(aes(label = round(bias, 2)), vjust = -0.5, size = 4) +
    labs(title = "Mean Absolute Bias (Nonlinear DGP)", x = "Model", y = "Bias") +
    theme_minimal(base_size = 14)
  
  bias_nonlinear_boot_plot <- ggplot(bias_nonlinear_df, aes(x = model, y = boot_bias)) +
    geom_col(fill = "deepskyblue4") +
    geom_text(aes(label = round(boot_bias, 2)), vjust = -0.5, size = 4) +
    labs(title = "Mean Absolute Bias (Nonlinear DGP) — Bootstrap", x = "Model", y = "Bootstrap Bias") +
    theme_minimal(base_size = 14)
  
  ### === RMSE PLOTS === ###
  rmse_linear_plot <- ggplot(rmse_linear_df, aes(x = model, y = rmse)) +
    geom_col(fill = "orange") +
    geom_text(aes(label = round(rmse, 2)), vjust = -0.5, size = 4) +
    labs(title = "RMSE (Linear DGP)", x = "Model", y = "RMSE") +
    theme_minimal(base_size = 14)
  
  rmse_linear_boot_plot <- ggplot(rmse_linear_df, aes(x = model, y = boot_rmse)) +
    geom_col(fill = "darkorange3") +
    geom_text(aes(label = round(boot_rmse, 2)), vjust = -0.5, size = 4) +
    labs(title = "RMSE (Linear DGP) — Bootstrap", x = "Model", y = "Bootstrap RMSE") +
    theme_minimal(base_size = 14)
  
  rmse_nonlinear_plot <- ggplot(rmse_nonlinear_df, aes(x = model, y = rmse)) +
    geom_col(fill = "orange") +
    geom_text(aes(label = round(rmse, 2)), vjust = -0.5, size = 4) +
    labs(title = "RMSE (Nonlinear DGP)", x = "Model", y = "RMSE") +
    theme_minimal(base_size = 14)
  
  rmse_nonlinear_boot_plot <- ggplot(rmse_nonlinear_df, aes(x = model, y = boot_rmse)) +
    geom_col(fill = "darkorange3") +
    geom_text(aes(label = round(boot_rmse, 2)), vjust = -0.5, size = 4) +
    labs(title = "RMSE (Nonlinear DGP) — Bootstrap", x = "Model", y = "Bootstrap RMSE") +
    theme_minimal(base_size = 14)
  
  ### === INTERVAL LENGTH PLOTS === ###
  interval_linear_plot <- ggplot(interval_linear_df, aes(x = model, y = interval, fill = type)) +
    geom_col(position = position_dodge(width = 0.7)) +
    geom_text(aes(label = round(interval, 2)), position = position_dodge(width = 0.7), vjust = -0.5, size = 4) +
    labs(title = "Interval Length (Linear DGP)", x = "Model", y = "Mean Interval Length") +
    scale_fill_brewer(palette = "Set2") +
    theme_minimal(base_size = 14)
  
  interval_nonlinear_plot <- ggplot(interval_nonlinear_df, aes(x = model, y = interval, fill = type)) +
    geom_col(position = position_dodge(width = 0.7)) +
    geom_text(aes(label = round(interval, 2)), position = position_dodge(width = 0.7), vjust = -0.5, size = 4) +
    labs(title = "Interval Length (Nonlinear DGP)", x = "Model", y = "Mean Interval Length") +
    scale_fill_brewer(palette = "Set2") +
    theme_minimal(base_size = 14)
  
  ### === COVERAGE PLOTS === ###
  coverage_linear_plot <- ggplot(coverage_linear_df, aes(x = model, y = coverage, fill = type)) +
    geom_col(position = position_dodge(width = 0.7)) +
    geom_text(aes(label = round(coverage, 2)), position = position_dodge(width = 0.7), vjust = -0.5, size = 4) +
    labs(title = "Coverage (Linear DGP)", x = "Model", y = "Coverage") +
    scale_fill_brewer(palette = "Set2") +
    theme_minimal(base_size = 14)
  
  coverage_nonlinear_plot <- ggplot(coverage_nonlinear_df, aes(x = model, y = coverage, fill = type)) +
    geom_col(position = position_dodge(width = 0.7)) +
    geom_text(aes(label = round(coverage, 2)), position = position_dodge(width = 0.7), vjust = -0.5, size = 4) +
    labs(title = "Coverage (Nonlinear DGP)", x = "Model", y = "Coverage") +
    scale_fill_brewer(palette = "Set2") +
    theme_minimal(base_size = 14)
  
  ### === COMPUTATION TIME PLOTS === ###
  time_linear_plot <- ggplot(time_linear_df, aes(x = model, y = time)) +
    geom_col(fill = "grey50") +
    labs(x = "n = 500", y = "Time (s)") +
    theme_minimal(base_size = 14)
  
  time_nonlinear_plot <- ggplot(time_nonlinear_df, aes(x = model, y = time)) +
    geom_col(fill = "grey50") +
    labs(x = "n = 500", y = "Time (s)") +
    theme_minimal(base_size = 14)
  
  ### === COMBINE PLOTS === ###
  combined_plot_bias <- bias_linear_plot + bias_linear_boot_plot + bias_nonlinear_plot + bias_nonlinear_boot_plot
  combined_plot_rmse <- rmse_linear_plot + rmse_linear_boot_plot + rmse_nonlinear_plot + rmse_nonlinear_boot_plot
  combined_plot_interval <- interval_linear_plot + interval_nonlinear_plot 
  combined_plot_coverage <- coverage_linear_plot + coverage_nonlinear_plot
  time_plots <- time_linear_plot + time_nonlinear_plot
  
  return(list(
    combined_plot_bias = combined_plot_bias,
    combined_plot_rmse = combined_plot_rmse,
    combined_plot_interval = combined_plot_interval,
    combined_plot_coverage = combined_plot_coverage,
    time_plots = time_plots
  ))
}


####################################################################################
# ******************************************************************************
####################################################################################
#end of funcitons 
J = 20
n500_results <- run_study(n = 500, B = 1000, J = J)

# change to your own
setwd("/Users/tylerschmidt/Work/School/STAT5400/Uncertainty-Quantification")
saveRDS(n500_results, "n500_results.rds")

# loading results in 
n100 <- readRDS("n100_results.rds")
n250 <- readRDS("n250_results.rds")
n500 <- readRDS("n500_results.rds")
#

sim_plots <- plots(n500)
### === PRINT PLOTS === ###
print(sim_plots$combined_plot_bias)
print(sim_plots$combined_plot_rmse)
print(sim_plots$combined_plot_interval)
print(sim_plots$combined_plot_coverage)
print(sim_plots$time_plots)

n100_time <- sim_plots$time_plots
n250_time <- sim_plots$time_plots
n500_time <- sim_plots$time_plots

n100_time / n250_time / n500_time
