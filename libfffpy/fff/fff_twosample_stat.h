/*!
  \file fff_twosample_stat.h
  \brief One-sample test statistics
  \author Alexis Roche
  \date 2008

*/

 
#ifndef FFF_TWOSAMPLE_STAT
#define FFF_TWOSAMPLE_STAT

#ifdef __cplusplus
extern "C" {
#endif

#include "fff_vector.h"
  
  /* Two-sample stat flag */
  typedef enum {
    FFF_TWOSAMPLE_STUDENT = 2,
    FFF_TWOSAMPLE_WILCOXON = 6,
    FFF_TWOSAMPLE_STUDENT_MFX = 12
  } fff_twosample_stat_flag;
 
  
  /*!
    \struct fff_twosample_stat
    \brief General structure for two-sample test statistics
  */
  typedef struct{
    unsigned int n1; /*!< number of subjects in first group */
    unsigned int n2; /*!< number of subjects in second group */
    fff_twosample_stat_flag flag; /*!< statistic's identifier */
    void* params; 
    double (*compute_stat)(void*, const fff_vector*, unsigned int); /*!< actual statistic implementation */
  } fff_twosample_stat;
  

  extern fff_twosample_stat* fff_twosample_stat_new(unsigned int n1, unsigned int n2, fff_twosample_stat_flag flag); 
  extern void fff_twosample_stat_delete(fff_twosample_stat* thisone); 
  extern double fff_twosample_stat_eval(fff_twosample_stat* thisone, const fff_vector* x);



  /** MFX **/ 


  /*!
    \struct fff_twosample_stat_mfx
    \brief General structure for two-sample test statistics
  */
  typedef struct{
    unsigned int n1; /*!< number of subjects in first group */
    unsigned int n2; /*!< number of subjects in second group */
    fff_twosample_stat_flag flag; /*!< statistic's identifier */
    unsigned int niter; 
    void* params; /*! auxiliary structures */ 
    double (*compute_stat)(void*, const fff_vector*, const fff_vector*, unsigned int); /*!< actual statistic implementation */
  } fff_twosample_stat_mfx;

  
  extern fff_twosample_stat_mfx* fff_twosample_stat_mfx_new(unsigned int n1, unsigned int n2, 
							    fff_twosample_stat_flag flag); 
  extern void fff_twosample_stat_mfx_delete(fff_twosample_stat_mfx* thisone); 
  extern double fff_twosample_stat_mfx_eval(fff_twosample_stat_mfx* thisone, 
					    const fff_vector* x, const fff_vector* vx);


  /** Label permutations **/
  extern unsigned int fff_twosample_permutation(unsigned int* idx1, unsigned int* idx2, 
						unsigned int n1, unsigned int n2, double* magic);
  
 
  extern void fff_twosample_apply_permutation(fff_vector* px, fff_vector* pv, 
					      const fff_vector* x1, const fff_vector* v1, 
					      const fff_vector* x2, const fff_vector* v2,
					      unsigned int i, 
					      const unsigned int* idx1, const unsigned int* idx2); 
  

#ifdef __cplusplus
}
#endif

#endif

