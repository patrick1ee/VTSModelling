#include "mex.h"
#include "math.h"
  /*  13-12-18		Created form doEulerNoiseX0Mex.c to be the same without the noise param (initially for use with isFP) */

void doEulerNoise(double wIE, double wEI, double wEE, double beta, double Tau,
                  double thetaE, double thetaI, double nMax,
                  double dt, double E0, double I0, double *E, double *I)
{
  E[0] = E0;
  I[0] = I0;

  /*  Euler method */
  
  mwSize i;
  double Edot;
  double Idot;

  for (i = 0; i < (int)(nMax - 1.0); i++) {
    
    Edot = (-E[i] + 1.0 / (1.0 + exp(-beta * (((thetaE - wIE * I[i]) + wEE * E[i]) - 1.0)))) / Tau;
    Idot = (-I[i] + 1.0 / (1.0 + exp(-beta * ((thetaI + wEI * E[i]) - 1.0)))) / Tau;
    E[i+1] = E[i] + Edot * dt;
    I[i+1] = I[i] + Idot * dt;
  }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
				double wIE;
				double wEI;
				double wEE;
                double beta;
                double Tau;
                double thetaE;
                double thetaI;
                double nMax;
                double dt;
				double E0;
				double I0;
                double *E;
                double *I;

				wIE = mxGetScalar(prhs[0]);
				wEI = mxGetScalar(prhs[1]);
				wEE = mxGetScalar(prhs[2]);
				beta = mxGetScalar(prhs[3]);
				Tau = mxGetScalar(prhs[4]);
				thetaE = mxGetScalar(prhs[5]);
				thetaI = mxGetScalar(prhs[6]);
				nMax = mxGetScalar(prhs[7]);
				dt = mxGetScalar(prhs[8]);
				E0 = mxGetScalar(prhs[9]);
				I0 = mxGetScalar(prhs[10]);

				plhs[0] = mxCreateDoubleMatrix(1,(mwSize)nMax,mxREAL);
				E = mxGetPr(plhs[0]);
				plhs[1] = mxCreateDoubleMatrix(1,(mwSize)nMax,mxREAL);
				I = mxGetPr(plhs[1]);
				
				doEulerNoise(wIE,wEI,wEE,beta,Tau,thetaE,thetaI,nMax,dt,E0,I0,E,I);
}

