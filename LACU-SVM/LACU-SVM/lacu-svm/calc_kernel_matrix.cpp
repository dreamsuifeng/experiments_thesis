#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "mex.h"

/* mex calc_kernel_matrix.cpp -largeArrayDims */

double linear_distance(int m_1, int *ind_1, double *val_1,
	int m_2, int *ind_2, double *val_2);

double Euclid_distance(int m_1, int *ind_1, double *val_1,
	int m_2, int *ind_2, double *val_2);

void mexFunction
        (
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]
        )
{
    const mxArray *x_array = prhs[0];
    const mxArray *y_array = prhs[1];
    const mxArray *kerneltype_array = prhs[2];
    const mxArray *parameter_array = prhs[3];
    
    double *x_data = mxGetPr(x_array);
    double *y_data = mxGetPr(y_array);
    double *kerneltype_data = mxGetPr(kerneltype_array);
    double *parameter_data = mxGetPr(parameter_array);
    double *kv_data;
    
    int num_x, num_y, dim, kernel_type, i, j, k, kv_index, low, high, m_x, m_y;
    
    mwIndex *x_ir, *x_jc, *y_ir, *y_jc;
    
    int *x_index, *y_index;
    double *x_value, *y_value;
    
    kernel_type = (int) kerneltype_data[0];
    num_x = (int) mxGetN(x_array);
    num_y = (int) mxGetN(y_array);
    dim = (int) mxGetM(x_array);
    
    x_index = new int[dim];
    y_index = new int[dim];
    x_value = new double[dim];
    y_value = new double[dim];
    
    plhs[0] = mxCreateDoubleMatrix(num_x, num_y, mxREAL);
    kv_data = mxGetPr(plhs[0]);
    
    x_ir = mxGetIr(x_array);
    x_jc = mxGetJc(x_array);
    y_ir = mxGetIr(y_array);
    y_jc = mxGetJc(y_array);
    
    for(i=0;i<num_x;i++)
    {
        low = (int) x_jc[i];
        high = (int) x_jc[i+1];
        m_x = 0;
        for(k=low; k<high; k++, m_x++)
        {
            x_index[m_x] = (int) x_ir[k];
            x_value[m_x] = x_data[k];
        }
        
        for(j=0;j<num_y;j++)
        {
            low = (int) y_jc[j];
            high = (int) y_jc[j+1];
            m_y = 0;
            for(k=low; k<high; k++, m_y++)
            {
                y_index[m_y] = (int) y_ir[k];
                y_value[m_y] = y_data[k];
            }
            
            kv_index = j * num_x + i;
            switch(kernel_type)
            {
                case 0:
                    kv_data[kv_index] = linear_distance(m_x, x_index, x_value, m_y, y_index, y_value);
                    break;
                case 1:
                    kv_data[kv_index] = pow(parameter_data[0] * linear_distance(m_x, x_index, x_value, m_y, y_index, y_value)+parameter_data[1], parameter_data[2]);
                    break;
                case 2:
                    kv_data[kv_index] = exp(-parameter_data[0] * Euclid_distance(m_x, x_index, x_value, m_y, y_index, y_value));
                    break;
            }
        }
    }
    
    delete[] x_index;
    delete[] y_index;
    delete[] x_value;
    delete[] y_value;
}

double linear_distance(int m_1, int *ind_1, double *val_1,
	int m_2, int *ind_2, double *val_2)
{	
	double distance = 0;
	int p=0, q=0;
	
	while(p < m_1 && q < m_2)
	{
		if(ind_1[p] == ind_2[q])
			distance += val_1[p++] * val_2[q++];
		else if(ind_1[p] < ind_2[q])
			p++;
		else
			q++;
	}
	
	return distance;
}

double Euclid_distance(int m_1, int *ind_1, double *val_1,
	int m_2, int *ind_2, double *val_2)
{	
	double distance = 0, tmp;
	int p=0, q=0;
	
	while(p < m_1 && q < m_2)
	{
		if(ind_1[p] == ind_2[q])
		{
			tmp = val_1[p++] - val_2[q++];
			distance += tmp * tmp;
		}
		else if(ind_1[p] < ind_2[q])
		{
			distance += val_1[p] * val_1[p];
			p++;
		}
		else
		{	
			distance += val_2[q] * val_2[q];
			q++;
		}
	}
	
	while(p < m_1)
	{
		distance += val_1[p] * val_1[p];
		p++;
	}
		
	while(q < m_2)
	{	
		distance += val_2[q] * val_2[q];
		q++;
	}
			
	return distance;
}