#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#include <fstream>
#include <iostream>

#include <vector>
#include <algorithm>

#include "svqp2/svqp2.h"
#include "svqp2/vector.h"

#include "mex.h"

using namespace std;

int cache_size=1024;              // 1024Mb cache size as default
double epsgr=1e-3;                // tolerance on gradients
int verbosity=1;                  // verbosity level, 0=off
vector<double> kparam;           // kernel parameters
int use_b0=1;                     // use threshold via constraint \sum a_i y_i =0

int n;
double *A;
double *x;
double *b;
double *cmin;
double *cmax;

//******************* Kernel functions ***********************
double kernel(int i, int j, void *kparam)
{
    return A[j * n + i];
}

void mexFunction(
        int nlhs,              // Number of left hand side (output) arguments
        mxArray *plhs[],       // Array of left hand side arguments
        int nrhs,              // Number of right hand side (input) arguments
        const mxArray *prhs[]  // Array of right hand side arguments
        )
{
    n  = mxGetM (prhs[0]);
    
    A = mxGetPr(prhs[0]);       
    b = mxGetPr(prhs[1]);
    cmin = mxGetPr(prhs[2]);
    cmax = mxGetPr(prhs[3]);
    verbosity = (int) (mxGetPr(prhs[4])[0]);
    if(nrhs > 4)
        x = mxGetPr(prhs[4]);
        
    // train
    SVQP2* sv = new SVQP2(n);
    
    (*sv).verbosity=verbosity;
    (*sv).Afunction=kernel;
    (*sv).Aclosure=(void*) &kparam;
    (*sv).maxcachesize=(long int) 1024*1024*cache_size;
    (*sv).sumflag=use_b0;
    (*sv).epsgr=epsgr;
    
    for(int j=0;j<n;++j){
        sv->Aperm[j] = j;
        sv->b[j] = b[j];
        sv->cmin[j] = cmin[j]; 
        sv->cmax[j] = cmax[j];
        
//         if(nrhs > 4)
//             sv->x[j] = x[j];
    }
    
    sv->run(true,true);
    
    plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL); // array for alphas
    double *alpha = mxGetPr(plhs[0]);
    for (int i=0; i<n; i++)
        alpha[i]= sv->x[i];
    
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    double b0 = (sv->gmin+ sv->gmax)/2.0;
    *mxGetPr(plhs[1]) = b0;
    
    plhs[2] = mxCreateDoubleMatrix(n,1,mxREAL);
    double *fval = mxGetPr(plhs[2]);
    vector<int> ip(n);
    for(int i=0; i<n; i++)
        ip[sv->pivot[i]] = i;
    for(int i=0; i<n; i++){
        fval[i] = sv->b[ip[i]] - sv->g[ip[i]] + b0; // y = f(x) 
    }
    
    delete sv;
}
