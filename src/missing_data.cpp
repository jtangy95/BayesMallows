#include <RcppArmadillo.h>
#include <cmath>
#include "distances.h"
#include "misc.h"

// [[Rcpp::depends(RcppArmadillo)]]

arma::vec propose_augmentation(const arma::vec& ranks, const arma::uvec& indicator){
  arma::vec proposal = ranks;
  proposal(arma::find(indicator == 1)) = arma::shuffle(ranks(arma::find(indicator == 1)));
  return(proposal);
}
// Note : In general there are two ways that a computer language can pass an argument to a subroutine. The first is called call-by-value and the second is Call-by-reference.
// Note : A reference parameter is declared by preceding the parameter name in the function's declaration with an `&`
// Note : Declaring a parameter as a reference allows a function to modify the original argument, rather than being passed a local copy that is restricted to the scope of the function.
// Note : Declaring function parameters `const` indicates that the function promises not to change these values.
void initialize_missing_ranks(arma::mat& rankings, const arma::umat& missing_indicator,
                              const arma::uvec& assessor_missing) {

  int n_assessors = rankings.n_cols;

  for(int i = 0; i < n_assessors; ++i){
    if(assessor_missing(i) == 0) {
      continue;
    } else {
      arma::vec rank_vector = rankings.col(i);
      arma::uvec present_inds = arma::find(missing_indicator.col(i) == 0);
      arma::uvec missing_inds = arma::find(missing_indicator.col(i) == 1);
      // Note : `find(condition)` returns a column vector containing the indices of elements which satisfies a relational condition
      // Find the available ranks and permute them
      arma::uvec new_ranks = arma::shuffle(arma_setdiff(
        // ! `setdiff` is defined in misc.cpp, which is an analogue of R's setdiff() function.
        arma::linspace<arma::uvec>(1, rank_vector.size()),
        // Note : angle brackets < > is used to specify a type
        // Note : `linspace(start, end, k)` generates a vector with linearly spaced k elements, where default is k=100.  
        // Note : `regspace(start, delta, end)` generates a vector with regularly spaced elements, where default is delta = 1 given start < end.
        // *** I think that `linspace` in this code should be replaced with `regspace`
        arma::conv_to<arma::uvec>::from(rank_vector(present_inds))
      ));

      for(int j = 0; j < missing_inds.size(); ++j){
        rank_vector(missing_inds(j)) = static_cast<double>(arma::as_scalar(new_ranks(j)));
      }
      rankings.col(i) = rank_vector;
    }
  }
}

void update_missing_ranks(arma::mat& rankings, const arma::uvec& current_cluster_assignment,
                          arma::vec& aug_acceptance,
                          const arma::umat& missing_indicator,
                          const arma::uvec& assessor_missing,
                          const arma::vec& alpha, const arma::mat& rho,
                          const std::string& metric){

  int n_items = rankings.n_rows;
  int n_assessors = rankings.n_cols;

  for(int i = 0; i < n_assessors; ++i){
    if(assessor_missing(i) == 0){
      ++aug_acceptance(i);
      continue;
    }

    // Sample an augmentation proposal
    arma::vec proposal = propose_augmentation(rankings.col(i), missing_indicator.col(i));

    // Draw a uniform random number
    double u = std::log(arma::randu<double>());

    // Find which cluster the assessor belongs to
    int cluster = current_cluster_assignment(i);

    double ratio = -alpha(cluster) / n_items *
      (get_rank_distance(proposal, rho.col(cluster), metric) -
      get_rank_distance(rankings.col(i), rho.col(cluster), metric));

    if(ratio > u){
      rankings.col(i) = proposal;
      ++aug_acceptance(i);
    }
  }
}
