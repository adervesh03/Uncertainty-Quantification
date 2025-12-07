

clean_study <- function(n, J = 50) {
  # initialize "vector of lists"
  sim_results_linear <- vector("list", J)
  sim_results_nonlinear <- vector("list", J)
  
  
  
  for (i in 1:J) {
    linear_filename <- paste0("linear_sim", sim_id, ".rds")
    sim_results_linear[[i]] <- readRDS(linear_filename)
    
    nonlinear_filename <- paste0("nonlinear_sim", sim_id, ".rds")
    sim_results_nonlinear[[i]] <- readRDS(nonlinear_filename)
  }
  
  
  return(list(sim_results_linear = sim_results_linear,
              sim_results_nonlinear = sim_results_nonlinear))
}


results <- clean_study(n =n, J = 50)

filename <-"n2_data.rds"
saveRDS(results, file=filename)