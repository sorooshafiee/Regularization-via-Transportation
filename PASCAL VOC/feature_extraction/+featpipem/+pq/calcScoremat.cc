#include <stdint.h>
#include <string.h>
#include "mex.h"
#include "matrix.h"

template <class uintType>
void calcScoreMat(const mwSize n, const mwSize class_count, const mwSize pq_dim,
		  const mxArray* LUT, const float* biasterms, const uintType* pqcodes,
		  float** scoremat) {
  float feat_score;
  mxArray* LUTcell;
  float* LUTarr;
  
  for (size_t fi = 0; fi < n; ++fi) {
    for (size_t ci = 0; ci < class_count; ++ci) {
      feat_score = 0.0;
      LUTcell = mxGetCell(LUT, ci);
      LUTarr = (float*)mxGetPr(LUTcell);
      for (size_t qi = 0; qi < pq_dim; ++qi) {
	feat_score = feat_score + LUTarr[qi + (pqcodes[qi + fi*pq_dim]-1)*pq_dim];
      }
      (*scoremat)[ci + fi*class_count] = feat_score + biasterms[ci];
    }
  }
}

// scoremat = f(lut, biasterms, pqcodes)
void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray *prhs[]) {

  // get input parameters
  mwSize class_count = mxGetNumberOfElements(prhs[0]);
  
  float* biasterms = (float*)mxGetData(prhs[1]);

  mwSize pq_dim = mxGetM(prhs[2]);
  mwSize n = mxGetN(prhs[2]);

  // prepare output parameters
  mwSize scoremat_dims[] = {class_count, n};
  plhs[0] = mxCreateNumericArray(2, scoremat_dims, mxSINGLE_CLASS, mxREAL);
  float* scoremat = (float*)mxGetPr(plhs[0]);

  // processing

  const char* class_name = mxGetClassName(prhs[2]);
  if (~strcmp(class_name, "uint8")) {
    uint8_t* pqcodes = (uint8_t*)mxGetData(prhs[2]);
    calcScoreMat<uint8_t>(n, class_count, pq_dim, prhs[0], biasterms, pqcodes, &scoremat);
  } else if (~strcmp(class_name, "uint16")) {
    uint16_t* pqcodes = (uint16_t*)mxGetData(prhs[2]);
    calcScoreMat<uint16_t>(n, class_count, pq_dim, prhs[0], biasterms, pqcodes, &scoremat);
  } else if (~strcmp(class_name, "uint32")) {
    uint32_t* pqcodes = (uint32_t*)mxGetData(prhs[2]);
    calcScoreMat<uint32_t>(n, class_count, pq_dim, prhs[0], biasterms, pqcodes, &scoremat);
  } else {
    printf("Type is: %s\n", mxGetClassName(prhs[2]));
    mexErrMsgTxt("Type of pqcodes could not be identified!");
  }
  
}
