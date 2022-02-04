#include <cmath>
#include "RcppArmadillo.h"
#include "mixtures.h"
#include "distances.h"
#include "missing_data.h"
#include "pairwise_comparisons.h"
#include "parameterupdates.h"

// [[Rcpp::depends(RcppArmadillo)]]

//' Worker function for computing the posterior distribution.
//'
//' @param rankings A set of complete rankings, with one sample per column.
//' With n_assessors samples and n_items items, rankings is n_items x n_assessors.
//' @param obs_freq  A vector of observation frequencies (weights) to apply to the rankings.
//' @param nmc Number of Monte Carlo samples.
//' @param constraints List of lists of lists, returned from `generate_constraints`.
//' @param cardinalities Used when metric equals \code{"footrule"} or
//' \code{"spearman"} for computing the partition function. Defaults to
//' \code{R_NilValue}.
//' @param logz_estimate Estimate of the log partition function.
//' @param metric The distance metric to use. One of \code{"spearman"},
//' \code{"footrule"}, \code{"kendall"}, \code{"cayley"}, or
//' \code{"hamming"}.
//' @param error_model Error model to use.
//' @param Lswap Swap parameter used by Swap proposal for proposing rank augmentations in the case of non-transitive pairwise comparisons.
//' @param n_clusters Number of clusters. Defaults to 1.
//' @param include_wcd Boolean defining whether or
//' not to store the within-cluster distance.
//' @param leap_size Leap-and-shift step size.
//' @param alpha_prop_sd Standard deviation of proposal distribution for alpha.
//' @param alpha_init Initial value of alpha.
//' @param alpha_jump How many times should we sample \code{rho} between
//' each time we sample \code{alpha}. Setting \code{alpha_jump} to a high
//' number can significantly speed up computation time, since we then do not
//' have to do expensive computation of the partition function.
//' @param lambda Parameter of the prior distribution.
//' @param alpha_max Maximum value of \code{alpha}, used for truncating the exponential prior distribution.
//' @param psi Hyperparameter for the Dirichlet prior distribution used in clustering.
//' @param rho_thinning Thinning parameter. Keep only every \code{rho_thinning} rank
//' sample from the posterior distribution.
//' @param aug_thinning Integer specifying the thinning for data augmentation.
//' @param clus_thin Integer specifying the thinning for saving cluster assignments.
//' @param save_aug Whether or not to save the augmented data every
//' \code{aug_thinning}th iteration.
//' @param verbose Logical specifying whether to print out the progress of the
//' Metropolis-Hastings algorithm. If \code{TRUE}, a notification is printed every
//' 1000th iteration.
//' @param kappa_1 Hyperparameter for \eqn{theta} in the Bernoulli error model. Defaults to 1.0.
//' @param kappa_2 Hyperparameter for \eqn{theta} in the Bernoulli error model. Defaults to 1.0.
//' @param save_ind_clus Whether or not to save the individual cluster probabilities in each step,
//' thinned as specified in argument \code{clus_thin}. This results in csv files \code{cluster_probs1.csv},
//' \code{cluster_probs2.csv}, ..., being saved in the calling directory. This option may slow down the code
//' considerably, but is necessary for detecting label switching using Stephen's algorithm.
//' @keywords internal
//'
// [[Rcpp::export]]
Rcpp::List run_mcmc(arma::mat rankings, arma::vec obs_freq, int nmc,
                    Rcpp::List constraints,
                    Rcpp::Nullable<arma::vec> cardinalities,
                    Rcpp::Nullable<arma::vec> logz_estimate,
                    Rcpp::Nullable<arma::mat> rho_init,
                    std::string metric = "footrule",
                    std::string error_model = "none",
                    int Lswap = 1,
                    int n_clusters = 1,
                    bool include_wcd = false,
                    int leap_size = 1,
                    double alpha_prop_sd = 0.5,
                    double alpha_init = 5,
                    int alpha_jump = 1,
                    double lambda = 0.1,
                    double alpha_max = 1e6,
                    int psi = 10,
                    int rho_thinning = 1,
                    int aug_thinning = 1,
                    int clus_thin = 1,
                    bool save_aug = false,
                    bool verbose = false,
                    double kappa_1 = 1.0,
                    double kappa_2 = 1.0,
                      bool save_ind_clus = false
                      ){

  // The number of items ranked
  int n_items = rankings.n_rows;

  // The number of assessors
  int n_assessors = rankings.n_cols;

  bool augpair = (constraints.length() > 0);
  bool any_missing = !arma::is_finite(rankings); 
  // Note : `is_finite` checks whether all elements are finite.

  arma::umat missing_indicator;
  arma::uvec assessor_missing;
  // Note : `umat` is matrix class consisting of uword, which is shorthand for unsigned integers. uvec is similar vector class.

  if(any_missing){
    // Converting to umat will convert NA to 0, but might cause clang-UBSAN error, so converting explicitly.
    rankings.replace(arma::datum::nan, 0);
    // Note : `.replace(old_value, new_value)`    `nan` stands for not a number
    missing_indicator = arma::conv_to<arma::umat>::from(rankings);
    // Note : `conv_to<type>::from(X)`  converts from one matrix(cube) type of X to another. In this case, from `mat` to `umat`
    missing_indicator.transform( [](int val) { return (val == 0) ? 1 : 0; } );
    // Note : `.transform(lambda_function)` transforms each element using lambda function [](){}
    // Note : `?` is conditional operator -> `condition ? expression1 : expression 2` works like `ifelse` in R
    assessor_missing = arma::conv_to<arma::uvec>::from(arma::sum(missing_indicator, 0));
    // Note : For matrix M, `sum(M, dim)` returns the sums of elements in each column (dim=0) or each row (dim=1)
    // Note : "assessor_missing" becomes N dim vector representing number of not answered item for each assessor
    initialize_missing_ranks(rankings, missing_indicator, assessor_missing);
    // ! `initialize_missing_ranks` is defined in missing_data.cpp 
  } else {
    missing_indicator.reset();
    assessor_missing.reset();
    // Note : `.reset()` reset the size to zero so that the object will have no elements.
  }

  // Declare the cube to hold the latent ranks
  arma::cube rho(n_items, n_clusters, std::ceil(static_cast<double>(nmc * 1.0 / rho_thinning)));
  // Note : `static_cast<new_type>(target)` convert the type of target variable to the new type
  rho.slice(0) = initialize_rho(rho_init, n_items, n_clusters);
  // ! `initialize_rho` is defined in parameterupdates.cpp
  arma::mat rho_old = rho(arma::span::all, arma::span::all, arma::span(0));
  // Note : `Q( span(first_row, last_row), span(first_col, last_col), span(first_slice, last_slice) )` provides subcube view.  `span(start,end)` can be replaced by `span::all` to indicate the entire range
  

  // Declare the vector to hold the scaling parameter alpha
  arma::mat alpha(n_clusters, std::ceil(static_cast<double>(nmc * 1.0 / alpha_jump)));
  alpha.col(0).fill(alpha_init);
  // `.fill(value)` is a member function of Mat, Col, Row, Cube, which sets the elemnts to a specified value.

  // If the user wants to save augmented data, we need a cube
  arma::cube augmented_data;
  if(save_aug){
    augmented_data.set_size(n_items, n_assessors, std::ceil(static_cast<double>(nmc * 1.0 / aug_thinning)));
    augmented_data.slice(0) = rankings;
  }

  // Clustering
  bool clustering = n_clusters > 1;
  int n_cluster_assignments = n_clusters > 1 ? std::ceil(static_cast<double>(nmc * 1.0 / clus_thin)) : 1;
  // Note : conditional operator hexpressions have the form `E1? E2 : E3` which is similar to ifelse function in R
  arma::mat cluster_probs(n_clusters, n_cluster_assignments);
  cluster_probs.col(0).fill(1.0 / n_clusters);
  arma::vec current_cluster_probs = cluster_probs.col(0);
  arma::umat cluster_assignment(n_assessors, n_cluster_assignments);
  cluster_assignment.col(0) = arma::randi<arma::uvec>(n_assessors, arma::distr_param(0, n_clusters - 1));
  // Note : `randi(n_elem, distr_param(a,b))` generates a vector with the elements set to random integer values in the [a,b] interval and the syntax is `vector_type v = randi<vector_type>(n_elem, distr_param(a,b))`

  arma::uvec current_cluster_assignment = cluster_assignment.col(0);

  // Matrix with precomputed distances d(R_j, \rho_j), used to avoid looping during cluster assignment`
  arma::mat dist_mat(n_assessors, n_clusters);
  update_dist_mat(dist_mat, rankings, rho_old, metric, obs_freq);
  // ! `update_dist_mat` is defined in mixtures.cpp
  arma::mat within_cluster_distance(n_clusters, include_wcd ? nmc : 1);
  within_cluster_distance.col(0) = update_wcd(current_cluster_assignment, dist_mat);
  // ! `update_wcd` is defined in mixtures.cpp


  // Declare indicator vectors to hold acceptance or not
  arma::vec alpha_acceptance = arma::ones(n_clusters);
  arma::vec rho_acceptance = arma::ones(n_clusters);

  arma::vec aug_acceptance;
  if(any_missing | augpair){
    // Note : `|` is logical symbol for OR
    aug_acceptance = arma::ones<arma::vec>(n_assessors);
  } else {
    aug_acceptance.reset();
    // Note : `.reset()` resets the size to zero so that the object will have no elements.
    // Note : If there is missing data or the given data is preferences, we should generate augmented rank in each mcmc step so that accpetance indicator vector for N assessors are necessary. Otherwise, we don't need augmented rank.
  }

  // Declare vector with Bernoulli parameter for the case of intransitive preferences
  arma::vec theta, shape_1, shape_2;
  if(error_model == "bernoulli"){
    theta = arma::zeros<arma::vec>(nmc);
    shape_1 = arma::zeros<arma::vec>(nmc);
    shape_2 = arma::zeros<arma::vec>(nmc);
    shape_1(0) = kappa_1;
    shape_2(0) = kappa_2;
  } else {
    theta.reset();
    shape_1.reset();
    shape_2.reset();
  }

  // Other variables used
  int alpha_index = 0, rho_index = 0, aug_index = 0, cluster_assignment_index = 0;
  arma::vec alpha_old = alpha.col(0);

  arma::uvec element_indices = arma::regspace<arma::uvec>(0, rankings.n_rows - 1);
  // Note : "element_indices" is same as 1:n in R where n is the number of items



  // This is the Metropolis-Hastings loop

  // Starting at t = 1, meaning that alpha and rho must be initialized at index 0,
  // and this has been done above
  for(int t = 1; t < nmc; ++t){

    // Check if the user has tried to interrupt.
    if (t % 1000 == 0) {
      Rcpp::checkUserInterrupt();
      if(verbose){
        Rcpp::Rcout << "First " << t << " iterations of Metropolis-Hastings algorithm completed." << std::endl;
      }

    }

    if(error_model == "bernoulli"){

      update_shape_bernoulli(shape_1(t), shape_2(t), kappa_1, kappa_2,
                             rankings, constraints);

      // Update the theta parameter for the error model, which is independent of cluster
      theta(t) = rtruncbeta(shape_1(t), shape_2(t), 0.5);
    }

    for(int i = 0; i < n_clusters; ++i){
      update_rho(rho, rho_acceptance, rho_old, rho_index, i,
                 rho_thinning, alpha_old(i), leap_size,
                 clustering ? rankings.submat(element_indices, arma::find(current_cluster_assignment == i)) : rankings,
                 metric, n_items, t, element_indices, obs_freq);
       // ! `update_rho` is defined in parameterupdates.cpp
    }

    if(t % alpha_jump == 0) {
      ++alpha_index;
      for(int i = 0; i < n_clusters; ++i){
        alpha(i, alpha_index) = update_alpha(alpha_acceptance, alpha_old(i),
              clustering ? rankings.submat(element_indices, arma::find(current_cluster_assignment == i)) : rankings,
              obs_freq, i, rho_old.col(i), alpha_prop_sd, metric, lambda, cardinalities, logz_estimate, alpha_max);
      }
      // ! `update_alpha` is defined in parameterupdates.cpp
      // Update alpha_old
      alpha_old = alpha.col(alpha_index);
    }

  if(clustering){
    current_cluster_probs = update_cluster_probs(current_cluster_assignment, n_clusters, psi);
    // ! `update_cluster_probs" is defined in mixtures.cpp

    // Note : I think that before updating cluster labels, we should update distance matrix since current rho matrix is updated
    current_cluster_assignment = update_cluster_labels(dist_mat, current_cluster_probs,
                                                       alpha_old, n_items, t, metric, cardinalities,
                                                       logz_estimate, save_ind_clus);

    if(t % clus_thin == 0){
      ++cluster_assignment_index;
      cluster_assignment.col(cluster_assignment_index) = current_cluster_assignment;
      cluster_probs.col(cluster_assignment_index) = current_cluster_probs;
    }
  }

  if(include_wcd){
    // Update within_cluster_distance
    within_cluster_distance.col(t) = update_wcd(current_cluster_assignment, dist_mat);
  }

  // Perform data augmentation of missing ranks, if needed
  if(any_missing){
    update_missing_ranks(rankings, current_cluster_assignment, aug_acceptance, missing_indicator,
                         assessor_missing, alpha_old, rho_old, metric);
  }

  // Perform data augmentation of pairwise comparisons, if needed
  if(augpair){
    augment_pairwise(rankings, current_cluster_assignment, alpha_old, 0.1, rho_old,
                     metric, constraints, aug_acceptance, clustering, error_model, Lswap);

  }

  // Save augmented data if the user wants this. Uses the same index as rho.
  if(save_aug & (t % aug_thinning == 0)){
    ++aug_index;
    augmented_data.slice(aug_index) = rankings;
  }

  if(clustering | include_wcd){
    update_dist_mat(dist_mat, rankings, rho_old, metric, obs_freq);
    }
  }



  // Return everything that might be of interest
  return Rcpp::List::create(
    Rcpp::Named("rho") = rho,
    Rcpp::Named("rho_acceptance") = rho_acceptance / nmc,
    Rcpp::Named("alpha") = alpha,
    Rcpp::Named("alpha_acceptance") = alpha_acceptance / nmc,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("shape1") = shape_1,
    Rcpp::Named("shape2") = shape_2,
    Rcpp::Named("cluster_assignment") = cluster_assignment + 1,
    Rcpp::Named("cluster_probs") = cluster_probs,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("within_cluster_distance") = within_cluster_distance,
    Rcpp::Named("augmented_data") = augmented_data,
    Rcpp::Named("any_missing") = any_missing,
    Rcpp::Named("augpair") = augpair,
    Rcpp::Named("aug_acceptance") = aug_acceptance / nmc,
    Rcpp::Named("n_assessors") = n_assessors,
    Rcpp::Named("obs_freq") = obs_freq
  );


}



