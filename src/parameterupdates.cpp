#include "RcppArmadillo.h"
#include "leapandshift.h"
#include "distances.h"
#include "partitionfuns.h"
#include <cmath>


// [[Rcpp::depends(RcppArmadillo)]]

// Initialize latent ranks as provided by rho_init, or randomly:
arma::mat initialize_rho(Rcpp::Nullable<arma::mat> rho_init, int n_items, int n_clusters){
  if(rho_init.isNotNull()){
    return arma::repmat(Rcpp::as<arma::mat>(rho_init), 1, n_clusters);
    // Rcpp facilitates data interchange between R and C++ through the templated functions `Rcpp::as` (for conversion of objects from R to C++) and `Rcpp::wrap` (for conversion from C++ to R).
  } else {
    return arma::shuffle(arma::repmat(arma::regspace<arma::mat>(1, 1, n_items), 1, n_clusters));
  }
}
// Note : `shuffle(V)` for vector V, generates a copy of the vector with the elements shuffled. `shuffle(X, dim)` for matrix X generates a copy of the matrix with the elemtns shuffled in each column(dim=0), or each row(dim=1) where the default is dim=0.
// Note : `regspace< mat >(1, n)` generates a vector same as 1:n in R and then converts it to a matrix
// 

double update_alpha(arma::vec& alpha_acceptance,
                  const double& alpha_old,
                  const arma::mat& rankings,
                  const arma::vec& obs_freq,
                  const int& cluster_index,
                  const arma::vec& rho_old,
                  const double& alpha_prop_sd,
                  const std::string& metric,
                  const double& lambda,
                  const Rcpp::Nullable<arma::vec> cardinalities = R_NilValue,
                  const Rcpp::Nullable<arma::vec> logz_estimate = R_NilValue,
                  double alpha_max = 1e6) {
  // Note : R_NilValue is equivalent to NULL value in R.

  // Set the number of assessors. Not using the variable from run_mcmc because
  // here we want the number of assessors in this cluster
  //int n_assessors = rankings.n_cols; (This was the code for the number of assessors in run_mcmc)
  int n_items = rho_old.n_elem;

  double alpha_proposal = std::exp(arma::randn<double>() * alpha_prop_sd +
                              std::log(alpha_old));
   // Note : `randn` is a random number generated from standard normal distrn. By using simga_alpha and log(alpha_old), we can consider it as a random number generated from normal with parameter log(alpha_old) and sigma_alpha.
   // Note : Y=log(X)~N(u, s) --> X=exp(Y) where Y~N(u, s)

  double rank_dist = rank_dist_sum(rankings, rho_old, metric, obs_freq);


  // Difference between current and proposed alpha
  double alpha_diff = alpha_old - alpha_proposal;

  // Compute the Metropolis-Hastings ratio
  double ratio =
    alpha_diff / n_items * rank_dist +
    lambda * alpha_diff +
    arma::sum(obs_freq) * (
        get_partition_function(n_items, alpha_old, cardinalities, logz_estimate, metric) -
          get_partition_function(n_items, alpha_proposal, cardinalities, logz_estimate, metric)
    ) + std::log(alpha_proposal) - std::log(alpha_old);
  // Note : Here `sum(obs_freq)` plays a rold of the number of assessors N. But I think it is better to replace it with n_assessors=ranking.n_cols because sub(obs_freq) is the total N while ranking.n_cols here is the number of assessors in this cluster.
  
  // Draw a uniform random number
  double u = std::log(arma::randu<double>());

  if(ratio > u && alpha_proposal < alpha_max){
    ++alpha_acceptance(cluster_index);
    return alpha_proposal;
  } else {
    return alpha_old;
  }
}


void update_rho(arma::cube& rho, arma::vec& rho_acceptance, arma::mat& rho_old,
                int& rho_index, const int& cluster_index, const int& rho_thinning,
                const double& alpha_old, const int& leap_size, const arma::mat& rankings,
                const std::string& metric, const int& n_items, const int& t,
                const arma::uvec& element_indices, const arma::vec& obs_freq) {

  arma::vec rho_cluster = rho_old.col(cluster_index);

  // Sample a rank proposal
  arma::vec rho_proposal;
  arma::uvec indices;
  double prob_backward, prob_forward;

  leap_and_shift(rho_proposal, indices, prob_backward, prob_forward,
                 rho_cluster, leap_size, !((metric == "cayley") || (metric == "ulam")));
  // ! `leap_and_shift` is defined in leapandshift.cpp
  // Note : then we get new "rho_proposal". Also, "indices" is the index where rho_old and rho_proposal are different. "prob_backward" and "prob_forward" are probabilities associated to the transition.

  // Compute the distances to current and proposed ranks
  double dist_new = rank_dist_sum(rankings.rows(indices), rho_proposal(indices), metric, obs_freq);
  double dist_old = rank_dist_sum(rankings.rows(indices), rho_cluster(indices), metric, obs_freq);
  // ! `rank_dist_sum` is defined in distances.cpp
  // `X.rows( vector_of_row_indices)` is a way of subsetting matrix
  // "dist_new" yields sum_{j=1}^N d(R_j, rho_proposal), where distance is caculated only on the indices where rho_proposal and rho_old are different.

  // Metropolis-Hastings ratio
  double ratio = - alpha_old / n_items * (dist_new - dist_old) +
    std::log(prob_backward) - std::log(prob_forward);
  // Note : Here a "ratio" is equal to log(r) in the paper.

  // Draw a uniform random number
  double u = std::log(arma::randu<double>());

  if(ratio > u){
    rho_old.col(cluster_index) = rho_proposal;
    ++rho_acceptance(cluster_index);
    // Note : At first, rho_old was initially given. Now, rho_old has changed. 
    // Note : rho_accpetance holds the number of acceptance. Here, increment it by one.
  }

  // Save rho if appropriate
  if(t % rho_thinning == 0){
    if(cluster_index == 0) ++rho_index;
    // Note : "cluster_index ==0 " implies that it is time for new slice for rho cube.
    rho.slice(rho_index).col(cluster_index) = rho_old.col(cluster_index);
  }

}



