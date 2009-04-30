/*!
  \file fff_onesample_stat.h
  \brief One-sample test statistics
  \author Alexis Roche
  \date 2004-2008

*/

 
#ifndef FFF_ONESAMPLE_STAT
#define FFF_ONESAMPLE_STAT

#ifdef __cplusplus
extern "C" {
#endif

#include "fff_vector.h"
  
  /*!
    \typedef fff_onesample_stat_flag
    \brief Decision statistic for one-sample tests

    \c FFF_ONESAMPLE_MEAN is the sample mean. In permutation testing
    context, it is equivalent to \c FFF_ONESAMPLE_STUDENT (see below). 

    \c FFF_ONESAMPLE_MEDIAN is the sample median. 
    
    \c FFF_ONESAMPLE_STUDENT is the one-sample Student statistic
    defined as \f$ t = \frac{\hat{m}-m}{\hat{\sigma}/\sqrt{n}} \f$,
    where \a n is the sample size, \f$\hat{m}\f$ is the sample mean,
    and \f$\hat{\sigma}\f$ is the sample standard deviation normalized
    by \a n-1.

    \c FFF_ONESAMPLE_LAPLACE is a robust version of Student's \a t
    based on the Laplace likelihood ratio. The statistic is defined
    by: \f$ t = {\rm sign}(med-m) \sqrt{2n\log(\frac{s_0}{s})}\f$,
    where \a n is the sample size, \f$med\f$ is the sample median, and
    \f$s, s_0\f$ are the mean absolute deviations wrt the median and
    the baseline, respectively. Owing to Wilks's theorem, \a t is an
    approximate Z-statistic under the null assumption \a m=base.

    \c FFF_ONESAMPLE_TUKEY is similar to Laplace's \a t except the
    scale estimates are computed using the median of absolute
    deviations (MAD) rather than the average absolute deviation. This
    provides an even more robust statistic, which we term Tukey's \a t
    as Tukey appears to be the first author who proposed MAD as a
    scale estimator.

    \c FFF_ONESAMPLE_SIGN_STAT is the simple sign statistic, \f$ t =
    (n_+ - n_-)/n \f$ where \f$ n_+ \f$ (resp. \f$ n_- \f$) is the
    number of sample values greater than (resp. lower than) the
    baseline, and \a n is the total sample size.

    \c FFF_ONESAMPLE_SIGNED_RANK is Wilcoxon's signed rank statistic,
    \f$ t = \frac{2}{n(n+1)} \sum_i {\rm rank}(|x_i-m|) {\rm sign}(x_i-m)
    \f$, where rank values range from 1 to \a n, the sample size. Using
    this definition, \a t ranges from -1 to 1.

    \c FFF_ONESAMPLE_ELR implements the empirical likelihood ratio for
    a univariate mean (see Owen, 2001). The one-tailed statistic is
    defined as: \f$ t = {\rm sign}(\hat{\mu}-m) \sqrt{-2\log\lambda}
    \f$, where \a n is the sample size, \f$\hat{\mu}\f$ is the
    empirical mean, and \f$\lambda\f$ is the empirical likelihood
    ratio. The latter is given by \f$ \lambda = \prod_{i=1}^n nw_i\f$
    where \f$ w_i \f$ are nonnegative weights assessing the
    "probability" of each datapoint under the null assumption that the
    population mean equals \a m.

    \c FFF_ONESAMPLE_GRUBB is the Grubb's statistic for normality
    testing. It is defined as \f$ t = \max_i
    \frac{|x_i-\hat{m}|}{\hat{\sigma}} \f$ where \f$\hat{m}\f$ is the
    sample mean, and \f$\hat{\sigma}\f$ is the sample standard
    deviation. 
  */
  typedef enum {
    FFF_ONESAMPLE_EMPIRICAL_MEAN = 0,
    FFF_ONESAMPLE_EMPIRICAL_MEDIAN = 1,
    FFF_ONESAMPLE_STUDENT = 2,
    FFF_ONESAMPLE_LAPLACE = 3,
    FFF_ONESAMPLE_TUKEY = 4,
    FFF_ONESAMPLE_SIGN_STAT = 5,
    FFF_ONESAMPLE_WILCOXON = 6,
    FFF_ONESAMPLE_ELR = 7, 
    FFF_ONESAMPLE_GRUBB = 8,
    FFF_ONESAMPLE_EMPIRICAL_MEAN_MFX = 10,
    FFF_ONESAMPLE_EMPIRICAL_MEDIAN_MFX = 11,
    FFF_ONESAMPLE_STUDENT_MFX = 12,
    FFF_ONESAMPLE_SIGN_STAT_MFX = 15,
    FFF_ONESAMPLE_WILCOXON_MFX = 16, 
    FFF_ONESAMPLE_ELR_MFX = 17, 
    FFF_ONESAMPLE_GAUSSIAN_MEAN_MFX = 19
  } fff_onesample_stat_flag;
  
  /*!
    \struct fff_onesample_stat
    \brief General structure for one-sample test statistics
  */
  typedef struct{
    fff_onesample_stat_flag flag; /*!< statistic's identifier */
    double base; /*!< baseline for mean-value testing */
    unsigned int constraint; /* non-zero for statistics computed from maximum likelihood under the null hypothesis */ 
    void* params; /*!< other auxiliary parameters */
    double (*compute_stat)(void*, const fff_vector*, double); /*!< actual statistic implementation */
  } fff_onesample_stat;
  
  
  /*!
    \struct fff_onesample_stat_mfx
    \brief General structure for one-sample test statistics with mixed-effects

    Tests statistics corrected for mixed effects, i.e. eliminates the
    influence of heteroscedastic measurement errors. The classical
    Student statistic is generalized from the likelihood ratio of the
    model including heteroscedastic first-level errors. More comments
    to come.
  */
  typedef struct{
    fff_onesample_stat_flag flag; /*!< MFX statistic's identifier */
    double base; /*!< baseline for mean-value testing */
    int empirical; /*!< boolean, tells whether MFX statistic is nonparametric or not */ 
    unsigned int niter; /* non-zero for statistics based on iterative algorithms */ 
    unsigned int constraint; /* non-zero for statistics computed from maximum likelihood under the null hypothesis */ 
    void* params; /*!< auxiliary parameters */
    double (*compute_stat)(void*, const fff_vector*, const fff_vector*, double); /*!< actual statistic implementation */
  } fff_onesample_stat_mfx;
  
  /*!
    \brief Constructor for the \c fff_onesample_stat structure 
    \param n sample size
    \param flag statistic identifier 
    \param base baseline value for mean-value testing
   */ 
  extern fff_onesample_stat* fff_onesample_stat_new(unsigned int n, fff_onesample_stat_flag flag, double base); 
  /*!
    \brief Destructor for the \c fff_onesample_stat structure 
    \param thisone instance to be deleted 
   */ 
  extern void fff_onesample_stat_delete(fff_onesample_stat* thisone); 
  /*!
    \brief Compute a one-sample test statistic
    \param thisone already created one-sample stat structure 
    \param x input vector 
  */  
  extern double fff_onesample_stat_eval(fff_onesample_stat* thisone, const fff_vector* x);


  /** MFX **/ 
  extern fff_onesample_stat_mfx* fff_onesample_stat_mfx_new(unsigned int n, fff_onesample_stat_flag flag, double base); 
  extern void fff_onesample_stat_mfx_delete(fff_onesample_stat_mfx* thisone); 
  extern double fff_onesample_stat_mfx_eval(fff_onesample_stat_mfx* thisone, const fff_vector* x, const fff_vector* vx);

  extern void fff_onesample_stat_mfx_pdf_fit(fff_vector* w, fff_vector* z,
					     fff_onesample_stat_mfx* thisone, 
					     const fff_vector* x, const fff_vector* vx);

  extern void fff_onesample_stat_gmfx_pdf_fit(double* mu, double* v, 
					      fff_onesample_stat_mfx* thisone, 
					      const fff_vector* x, const fff_vector* vx);

  /** Sign permutations **/
  extern void fff_onesample_permute_signs(fff_vector* xx, const fff_vector* x, double magic);  

#ifdef __cplusplus
}
#endif

#endif

