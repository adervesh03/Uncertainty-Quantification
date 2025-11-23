library(SoftBart)
library(ranger)
library(xgboost)
library(ggplot2)
library(patchwork)

##########################################################################
# function: simulate_one_series
# takes in: # Time T, ar2 coefficients phi, covariate betas, sigma
# returns: list of vector of outcomes y, matrix of covariates X.
simulate_one_series <- function(T = 5, phi = c(0.5, -0.3),
                                beta = c(2,-3.1,4.7,0.5,1.3), sigma = 1) {
  K <- length(beta)            # number of covariates (5)
  X <- matrix(rnorm(T*K), nrow = T, ncol = K)  # X[t, ]
  y <- numeric(T)
  eps <- rnorm(T, 0, sigma)
  
  # initialize y1,y2 using covariates (or any baseline)
  y[1] <- sum(X[1,]*beta) + eps[1]
  y[2] <- sum(X[2,]*beta) + eps[2]
  
  for (t in 3:T) {
    fx <- sum(X[t,] * beta)          # f(X_t) = linear here; replace with nonlinear f if desired
    y[t] <- phi[1]*y[t-1] + phi[2]*y[t-2] + fx + eps[t]
  }
  return(list(y = y, X = X))
}


build_lagged <- function(sim) {
  y <- sim$y
  X <- sim$X
  T <- length(y)
  
  Z <- list()
  Y <- numeric()
  
  for (t in 3:T) {
    x_t <- X[t, ]
    z_t <- c(x_t, y[t-1], y[t-2])   # features
    Z[[t]] <- z_t
    Y[t] <- y[t]
  }
  
  Zmat <- do.call(rbind, Z[3:T])
  Yvec <- Y[3:T]
  
  return(list(Z = Zmat, Y = Yvec))
}




simulate_many_series <- function(M = 100, T = 5,
                                 phi = c(0.5, -0.3),
                                 beta = c(2,-3.1,4.7,0.5,1.3),
                                 sigma = 1) {
  
  Z_list <- list()
  Y_list <- list()
  
  for (i in 1:M) {
    dat <- simulate_one_series(T=T, phi=phi, beta=beta, sigma=sigma)
    lagged <- build_lagged(dat$y, dat$X)
    
    Z_list[[i]] <- lagged$Zmat
    Y_list[[i]] <- lagged$Yvec
  }
  
  # Combine all series
  Z_all <- do.call(rbind, Z_list)
  Y_all <- unlist(Y_list)
  
  list(Z=Z_all, Y=Y_all)
}




simulate_UQ <- function(M = 50, T = 200, phi = c(0.5,-0.3), beta = c(2,-3.1,4.7,0.5,1.3),
                        sigma = 1, train_frac = 0.8, B = 200) {
  
  # simulate multiple series
  sim_list <- lapply(1:M, function(i) simulate_one_series(T, phi, beta, sigma))
  lagged_list <- lapply(sim_list, build_lagged)
  
  # stack all series
  Z_all <- do.call(rbind, lapply(lagged_list, `[[`, "Z"))
  Y_all <- unlist(lapply(lagged_list, `[[`, "Y"))
  
  # train/test split (pooled)
  n <- nrow(Z_all)
  train_idx <- 1:floor(train_frac*n)
  test_idx <- (floor(train_frac*n)+1):n
  
  train_X <- Z_all[train_idx, ]
  train_Y <- Y_all[train_idx]
  test_X <- Z_all[test_idx, ]
  test_Y <- Y_all[test_idx]
  
  #### SoftBART
  hypers <- Hypers(X = train_X, Y = train_Y, num_tree = 200)
  opts <- Opts(num_burn = B, num_save = B)
  
  softbart_fit <- softbart(X = train_X, Y = train_Y, X_test = test_X,
                           hypers = hypers, opts = opts)
  
  softbart_uq <- sapply(1:B, function(b) softbart_fit$y_hat_test[b,] +
                          rnorm(nrow(test_X), 0, softbart_fit$sigma[b]))
  

  
  # return list of matrices
  return(list(softbart = softbart_uq,
              test_Y = test_Y))
}

##############################################
res <- simulate_UQ(M = 50, T = 6, B = 200)
