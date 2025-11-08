library(SoftBart)
library(ranger)
library(xgboost)
library(ggplot2)

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
# function: model_fits
# takes in: data frame data, 
# folds, # splits K, Bootstrap size B 
# returns: matrices containing uncertainty for each of our three models.
model_fits <- function( data, folds, K, B, verbose = TRUE) {
  # we will use B = 2000, and 
  n <- nrow(data)
  y <- data$y
  X <- data[, -1]
  
  softbart_uq <- matrix(NA, nrow = B, ncol = n)
  ranger_uq <- matrix(NA, nrow = B, ncol = n)
  xgboost_uq <- matrix(NA, nrow = B, ncol = n)
  
  for (k in 1:K) {
    train <- which(folds != k) 
    test <- which(folds ==k)
    
    train_X <- X[train, ]
    train_Y <- y[train]
    test_X  <- X[test, ]
    
    # softbart fit
    
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
    # Add residual uncertainty to get prediction draws
    n_test <- ncol(softbart_fit$y_hat_test)
    B_iter <- nrow(softbart_fit$y_hat_test)
    
    y_pred <- matrix(NA, nrow = B_iter, ncol = n_test)
    for (b in 1:B_iter) {
      # add a draw from N(0, sigma[b]^2) for each test point
      y_pred[b, ] <- softbart_fit$y_hat_test[b, ] + rnorm(n_test, mean = 0, sd = softbart_fit$sigma[b])
    }
    
    softbart_uq[, test] <- y_pred
    })
    cat("Finished SoftBART on fold", k, "\n")
    
    ### ---- Ranger (Random Forest) ----
    # Fit model
    ranger_time <- system.time({
      # Fit original model
      ranger_fit <- ranger(y ~ ., data = data[train, ], num.trees = 500, num.threads = 1)
      
      # Estimate residual standard deviation
      resid_sd <- sd(train_Y - predict(ranger_fit, data[train, ])$predictions)
      
      # Bootstrap predictions for uncertainty quantification
      for (b in 1:B) {
        n_train <- nrow(train_X)
        boot_idx <- sample(1:n_train, replace = TRUE)
        boot_fit <- ranger(y ~ ., data = data[boot_idx, ], num.trees = 500)
        
        pred <- predict(boot_fit, data = test_X)$predictions
        # add residual noise
        pred <- pred + rnorm(length(pred), mean = 0, sd = resid_sd)
        
        ranger_uq[b, test] <- pred
      }
    })
    cat("Finished ranger on fold", k, "\n")
    
    ### ---- XGBoost ----
    ### ---- XGBoost ----
    xgboost_time <- system.time({
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
      
      # Fit original model to compute residual sd
      xgb_orig <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
      resid_sd <- sd(train_Y - predict(xgb_orig, dtrain))
      
      for (b in 1:B) {
        # Bootstrap resample
        n_train <- nrow(train_X)
        boot_idx <- sample(1:n_train, replace = TRUE)
        dboot <- xgb.DMatrix(data = as.matrix(train_X[boot_idx, ]), label = train_Y[boot_idx])
        
        fit <- xgb.train(params = params, data = dboot, nrounds = 100, verbose = 0)
        pred <- predict(fit, dtest)
        
        # add residual noise
        pred <- pred + rnorm(length(pred), mean = 0, sd = resid_sd)
        
        xgboost_uq[b, test] <- pred
      }
    })
    cat("Finished xgboost on fold", k, "\n")
  }
  
  return(list(
    softbart_uq = softbart_uq,
    ranger_uq = ranger_uq,
    xgboost_uq = xgboost_uq, 
    softbart_time = softbart_time, 
    ranger_time = ranger_time, 
    xgboost_time = xgboost_time
  ))
  
}





#####################################################################################
# function: Compute Simulation
# takes in: sample size n, sim number J, number of splits K, number of bootstrap replicates B,
# version (0 for linear, 1 for nonlinear)
# returns: simulation results list model_results
compute_simulation <- function(n, J, K, B, version) {
  sim_seed = 70 + J
  set.seed(sim_seed)
  if (version == "linear") {
    sim_data  <- generate_data(n = n )
  } else if ( version == "nonlinear" ) {
    sim_data  <- generate_data_2(n = n )
  } else {
    return("choose a correct version")
  }
 
  folds <- sample(rep(1:5, length.out = n))
  
  model_results <- model_fits(data = sim_data, folds = folds, K = K, B = B)
  
  # performance code
  
  y <- sim_data$y
  
  softbart_uq = model_results$softbart_uq
  ranger_uq = model_results$ranger_uq
  xgboost_uq = model_results$xgboost_uq
  
  # bias
  softbart_bias <- mean(colMeans(softbart_uq) - y)
  ranger_bias <- mean(colMeans(ranger_uq) - y)
  xgboost_bias <- mean(colMeans(xgboost_uq) - y)
  
  bias <- list(
    softbart_bias = softbart_bias, 
    ranger_bias = ranger_bias, 
    xgboost_bias = xgboost_bias
  )
  
  # rmse 
  softbart_rmse <- sqrt(mean((colMeans(softbart_uq) - y)^2))
  ranger_rmse <- sqrt(mean((colMeans(ranger_uq) - y)^2))
  xgboost_rmse <- sqrt(mean((colMeans(xgboost_uq) - y)^2))
  
  rmse <- list (
    softbart_rmse = softbart_rmse, 
    ranger_rmse = ranger_rmse, 
    xgboost_rmse = xgboost_rmse
  )
  
  
  # interval length 
  softbart_lower <- apply(softbart_uq, 2, quantile, 0.025)
  softbart_upper <- apply(softbart_uq, 2, quantile, 0.975)
  softbart_int_length <- mean(softbart_upper - softbart_lower)
  
  ranger_lower <- apply(ranger_uq, 2, quantile, 0.025)
  ranger_upper <- apply(ranger_uq, 2, quantile, 0.975)
  ranger_int_length <- mean(ranger_upper - ranger_lower)
  
  xgboost_lower <- apply(xgboost_uq, 2, quantile, 0.025)
  xgboost_upper <- apply(xgboost_uq, 2, quantile, 0.975)
  xgboost_int_length <- mean(xgboost_upper - xgboost_lower)
  
  interval_length <- list (
    softbart_int_length = softbart_int_length, 
    ranger_int_length =  ranger_int_length, 
    xgboost_int_length = xgboost_int_length
  )
  
  
  # coverage 
  softbart_coverage <- mean(y >= softbart_lower & y <= softbart_upper)
  ranger_coverage <- mean(y >= ranger_lower & y <= ranger_upper)
  xgboost_coverage <- mean(y >= xgboost_lower & y <= xgboost_upper)
  
  coverage <- list(
    softbart_coverage = softbart_coverage, 
    ranger_coverage = ranger_coverage, 
    xgboost_coverage = xgboost_coverage
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

J <- 5

# initialize "vector of lists"
sim_results_linear <- vector("list", J)
sim_results_nonlinear <- vector("list", J)

for (j in seq_len(J)) {
  cat("Running replication", j, "of", J, "\n")
  
  sim_results_linear[[j]] <- compute_simulation(
    n = 100, J =j, K = 5, B = 1000, version = "linear"
  )
  
  sim_results_nonlinear[[j]] <- compute_simulation(
    n = 100, J = j, K = 5, B = 1000, version = "nonlinear"
  )
}

# results storage

# bias
softbart_bias_linear <- numeric(J)
softbart_bias_nonlinear <- numeric(J)

ranger_bias_linear <- numeric(J)
ranger_bias_nonlinear <- numeric(J)

xgboost_bias_linear <- numeric(J)
xgboost_bias_nonlinear <- numeric(J)

# rmse
softbart_rmse_linear <- numeric(J)
softbart_rmse_nonlinear <- numeric(J)

ranger_rmse_linear <- numeric(J)
ranger_rmse_nonlinear <- numeric(J)

xgboost_rmse_linear <- numeric(J)
xgboost_rmse_nonlinear <- numeric(J)

# interval length
softbart_interval_length_linear <- numeric(J)
softbart_interval_length_nonlinear <- numeric(J)

ranger_interval_length_linear <- numeric(J)
ranger_interval_length_nonlinear <- numeric(J)

xgboost_interval_length_linear <- numeric(J)
xgboost_interval_length_nonlinear <- numeric(J)

# coverage
softbart_coverage_linear <- numeric(J)
softbart_coverage_nonlinear <- numeric(J)

ranger_coverage_linear <- numeric(J)
ranger_coverage_nonlinear <- numeric(J)

xgboost_coverage_linear <- numeric(J)
xgboost_coverage_nonlinear <- numeric(J)


# time 
softbart_time_linear <- numeric(J)
softbart_time_nonlinear <- numeric(J)

ranger_time_linear <- numeric(J)
ranger_time_nonlinear <- numeric(J)

xgboost_time_linear <- numeric(J)
xgboost_time_nonlinear <- numeric(J)


# results extraction

for (j in 1:J) {
  # bias
  softbart_bias_linear[j] <- sim_results_linear[[j]]$bias$softbart_bias
  softbart_bias_nonlinear[j] <- sim_results_nonlinear[[j]]$bias$softbart_bias
  
  ranger_bias_linear[j] <- sim_results_linear[[j]]$bias$ranger_bias
  ranger_bias_nonlinear[j] <- sim_results_nonlinear[[j]]$bias$ranger_bias
  
  xgboost_bias_linear[j] <- sim_results_linear[[j]]$bias$xgboost_bias
  xgboost_bias_nonlinear[j] <- sim_results_nonlinear[[j]]$bias$xgboost_bias
  
  # rmse
  softbart_rmse_linear[j] <- sim_results_linear[[j]]$rmse$softbart_rmse
  softbart_rmse_nonlinear[j] <- sim_results_nonlinear[[j]]$rmse$softbart_rmse
  
  ranger_rmse_linear[j] <- sim_results_linear[[j]]$rmse$ranger_rmse
  ranger_rmse_nonlinear[j] <- sim_results_nonlinear[[j]]$rmse$ranger_rmse
  
  xgboost_rmse_linear[j] <- sim_results_linear[[j]]$rmse$xgboost_rmse
  xgboost_rmse_nonlinear[j] <- sim_results_nonlinear[[j]]$rmse$xgboost_rmse
  
  # interval length
  softbart_interval_length_linear[j] <- sim_results_linear[[j]]$interval_length$softbart_int_length
  softbart_interval_length_nonlinear[j] <- sim_results_nonlinear[[j]]$interval_length$softbart_int_length
  
  ranger_interval_length_linear[j] <-sim_results_linear[[j]]$interval_length$ranger_int_length
  ranger_interval_length_nonlinear[j] <- sim_results_nonlinear[[j]]$interval_length$ranger_int_length
  
  xgboost_interval_length_linear[j] <- sim_results_linear[[j]]$interval_length$xgboost_int_length
  xgboost_interval_length_nonlinear[j] <- sim_results_nonlinear[[j]]$interval_length$xgboost_int_length
  
  # coverage
  softbart_coverage_linear[j] <- sim_results_linear[[j]]$coverage$softbart_coverage
  softbart_coverage_nonlinear[j] <- sim_results_nonlinear[[j]]$coverage$softbart_coverage
  
  ranger_coverage_linear[j] <- sim_results_linear[[j]]$coverage$ranger_coverage
  ranger_coverage_nonlinear[j] <- sim_results_nonlinear[[j]]$coverage$ranger_coverage
  
  xgboost_coverage_linear[j] <- sim_results_linear[[j]]$coverage$xgboost_coverage
  xgboost_coverage_nonlinear[j] <- sim_results_nonlinear[[j]]$coverage$xgboost_coverage
  
  
  # time 
  softbart_time_linear[j] <- sim_results_linear[[j]]$time$softbart_time[["elapsed"]]
  softbart_time_nonlinear[j] <- sim_results_nonlinear[[j]]$time$softbart_time[["elapsed"]]
  
  ranger_time_linear[j] <- sim_results_linear[[j]]$time$ranger_time[["elapsed"]]
  ranger_time_nonlinear[j] <- sim_results_nonlinear[[j]]$time$ranger_time[["elapsed"]]
  
  xgboost_time_linear[j] <- sim_results_linear[[j]]$time$xgboost_time[["elapsed"]]
  xgboost_time_nonlinear[j] <- sim_results_nonlinear[[j]]$time$xgboost_time[["elapsed"]]
  
}


# Bias
bias_linear_df <- data.frame(
  model = c("SoftBART", "Ranger", "XGBoost"),
  bias = c(
    mean(abs(softbart_bias_linear)),
    mean(abs(ranger_bias_linear)),
    mean(abs(xgboost_bias_linear))
  )
)

bias_nonlinear_df <- data.frame(
  model = c("SoftBART", "Ranger", "XGBoost"),
  bias = c(
    mean(abs(softbart_bias_nonlinear)),
    mean(abs(ranger_bias_nonlinear)),
    mean(abs(xgboost_bias_nonlinear))
  )
)

# RMSE
rmse_linear_df <- data.frame(
  model = c("SoftBART", "Ranger", "XGBoost"),
  rmse = c(
    mean(softbart_rmse_linear),
    mean(ranger_rmse_linear),
    mean(xgboost_rmse_linear)
  )
)

rmse_nonlinear_df <- data.frame(
  model = c("SoftBART", "Ranger", "XGBoost"),
  rmse = c(
    mean(softbart_rmse_nonlinear),
    mean(ranger_rmse_nonlinear),
    mean(xgboost_rmse_nonlinear)
  )
)

# Interval length
interval_length_linear_df <- data.frame(
  model = c("SoftBART", "Ranger", "XGBoost"),
  interval_length = c(
    mean(softbart_interval_length_linear),
    mean(ranger_interval_length_linear),
    mean(xgboost_interval_length_linear)
  )
)

interval_length_nonlinear_df <- data.frame(
  model = c("SoftBART", "Ranger", "XGBoost"),
  interval_length = c(
    mean(softbart_interval_length_nonlinear),
    mean(ranger_interval_length_nonlinear),
    mean(xgboost_interval_length_nonlinear)
  )
)

# Coverage
coverage_linear_df <- data.frame(
  model = c("SoftBART", "Ranger", "XGBoost"),
  coverage = c(
    mean(softbart_coverage_linear),
    mean(ranger_coverage_linear),
    mean(xgboost_coverage_linear)
  )
)

coverage_nonlinear_df <- data.frame(
  model = c("SoftBART", "Ranger", "XGBoost"),
  coverage = c(
    mean(softbart_coverage_nonlinear),
    mean(ranger_coverage_nonlinear),
    mean(xgboost_coverage_nonlinear)
  )
)

# Time
time_linear_df <- data.frame(
  model = c("SoftBART", "Ranger", "XGBoost"),
  time = c(
    mean(softbart_time_linear),
    mean(ranger_time_linear),
    mean(xgboost_time_linear)
  )
)

time_nonlinear_df <- data.frame(
  model = c("SoftBART", "Ranger", "XGBoost"),
  time = c(
    mean(softbart_time_nonlinear),
    mean(ranger_time_nonlinear),
    mean(xgboost_time_nonlinear)
  )
  
  
  
)



### === BIAS PLOTS === ###
bias_linear_plot <- ggplot(bias_linear_df, aes(x = model, y = bias)) +
  geom_col(fill = "skyblue") +
  geom_text(aes(label = round(bias, 2)), vjust = -0.5, size = 4) +
  labs(title = "Mean Absolute Bias (Linear DGP)", x = "Model", y = "Bias") +
  theme_minimal(base_size = 14)

bias_nonlinear_plot <- ggplot(bias_nonlinear_df, aes(x = model, y = bias)) +
  geom_col(fill = "skyblue") +
  geom_text(aes(label = round(bias, 2)), vjust = -0.5, size = 4) +
  labs(title = "Mean Absolute Bias (Nonlinear DGP)", x = "Model", y = "Bias") +
  theme_minimal(base_size = 14)

### === RMSE PLOTS === ###
rmse_linear_plot <- ggplot(rmse_linear_df, aes(x = model, y = rmse)) +
  geom_col(fill = "orange") +
  geom_text(aes(label = round(rmse, 1)), vjust = -0.5, size = 4) +
  labs(title = "RMSE (Linear DGP)", x = "Model", y = "RMSE") +
  theme_minimal(base_size = 14)

rmse_nonlinear_plot <- ggplot(rmse_nonlinear_df, aes(x = model, y = rmse)) +
  geom_col(fill = "orange") +
  geom_text(aes(label = round(rmse, 1)), vjust = -0.5, size = 4) +
  labs(title = "RMSE (Nonlinear DGP)", x = "Model", y = "RMSE") +
  theme_minimal(base_size = 14)

### === INTERVAL LENGTH PLOTS === ###
interval_length_linear_plot <- ggplot(interval_length_linear_df, aes(x = model, y = interval_length)) +
  geom_col(fill = "seagreen3") +
  geom_text(aes(label = round(interval_length, 1)), vjust = -0.5, size = 4) +
  labs(title = "Interval Length (Linear DGP)", x = "Model", y = "Mean Interval Length") +
  theme_minimal(base_size = 14)

interval_length_nonlinear_plot <- ggplot(interval_length_nonlinear_df, aes(x = model, y = interval_length)) +
  geom_col(fill = "seagreen3") +
  geom_text(aes(label = round(interval_length, 1)), vjust = -0.5, size = 4) +
  labs(title = "Interval Length (Nonlinear DGP)", x = "Model", y = "Mean Interval Length") +
  theme_minimal(base_size = 14)

### === COVERAGE PLOTS === ###
coverage_linear_plot <- ggplot(coverage_linear_df, aes(x = model, y = coverage)) +
  geom_col(fill = "plum") +
  geom_text(aes(label = round(coverage, 2)), vjust = -0.5, size = 4) +
  labs(title = "Coverage (Linear DGP)", x = "Model", y = "Coverage") +
  theme_minimal(base_size = 14)

coverage_nonlinear_plot <- ggplot(coverage_nonlinear_df, aes(x = model, y = coverage)) +
  geom_col(fill = "plum") +
  geom_text(aes(label = round(coverage, 2)), vjust = -0.5, size = 4) +
  labs(title = "Coverage (Nonlinear DGP)", x = "Model", y = "Coverage") +
  theme_minimal(base_size = 14)

### === COMPUTATION TIME PLOTS === ###
time_linear_plot <- ggplot(time_linear_df, aes(x = model, y = time)) +
  geom_col(fill = "grey50") +
  geom_text(aes(label = round(time, 1)), vjust = -0.5, size = 4) +
  labs(title = "Computation Time (Linear DGP)", x = "Model", y = "Time (s)") +
  theme_minimal(base_size = 14)

time_nonlinear_plot <- ggplot(time_nonlinear_df, aes(x = model, y = time)) +
  geom_col(fill = "grey50") +
  geom_text(aes(label = round(time, 1)), vjust = -0.5, size = 4) +
  labs(title = "Computation Time (Nonlinear DGP)", x = "Model", y = "Time (s)") +
  theme_minimal(base_size = 14)

combined_plot_linear <- bias_linear_plot + rmse_linear_plot + 
  interval_length_linear_plot + coverage_linear_plot 

combined_plot_nonlinear <- bias_nonlinear_plot + rmse_nonlinear_plot + 
  interval_length_nonlinear_plot + coverage_nonlinear_plot 

time_plots <- time_linear_plot + time_nonlinear_plot

print(combined_plot_linear)
print(combined_plot_nonlinear)
print(time_plots)

