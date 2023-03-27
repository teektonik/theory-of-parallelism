#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <string.h>

#include <cublas_v2.h>

int main(int argc, char** argv) {

       	int iter_max;
	int sizeofnet1;
	double accuracy;

	accuracy = atof(argv[1]);
	sizeofnet1 = atoi(argv[2]);
	int n = sizeofnet1;
	iter_max = atoi(argv[3]);
	double err = 1;

	double* A = (double*)calloc(n * n, sizeof(double));
	double* anew = (double*)calloc(n * n, sizeof(double));

     	cublasHandle_t handler;
	cublasStatus_t status = cublasCreate(&handler);
	A[0] = 10;
	A[0 + n - 1] = 20;
	A[0 + n * (n - 1)] = 20;
	A[(n - 1) * n + n - 1] = 30;
	anew[0] = 10;
	anew[0 + n - 1] = 20;
	anew[0 + n * (n - 1)] = 20;
	anew[(n - 1) * n + n - 1] = 30;

	double dif1 = (double)10 / (n - 1);
	for (int i = 1; i < n - 1; i++) {
		A[0 + i * n] = A[0 + i * n - n] + dif1;
		A[(n - 1) + i * n] = A[(n - 1) + i * n - n] + dif1;
		A[i] = A[i - 1] + dif1;
		A[i + (n - 1) * (n)] = A[i + (n - 1) * (n)-1] + dif1;
		anew[0 + i * n] = anew[0 + i * n - n] + dif1;
		anew[(n - 1) + i * n] = anew[(n - 1) + i * n - n] + dif1;
		anew[i] = anew[i - 1] + dif1;
		anew[i + (n - 1) * (n)] = anew[i + (n - 1) * (n)-1] + dif1;
	}

	int iter = 0;
	int N=n*n;
//	const double alpha = -1;
//	int idx = 0;
	memcpy(anew, A, n * n * sizeof(double));
	#pragma acc enter data copyin(A[0:N], anew[0:N])
	{
	//	const double alpha = -1;
	//	                         int idx = 0;
			
	for (iter = 0; iter < iter_max && err>accuracy; iter++) {
	 const double alpha = -1;
         int idx = 0;	 
	#pragma acc data present(A, anew)		
	#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(128) async
			for (int j = 1; j < n - 1; j++) {
				for (int i = 1; i < n - 1; i++) {
					anew[i + j * n] = 0.25 * (A[i + j * n - n] + A[i + j * n + n] + A[i + j * n - 1] + A[i + j * n + 1]);
																					                
				}
			}
		
	
		 if (iter % 100 == 0) {	 		
			#pragma acc data present(A, anew) wait
			#pragma acc host_data use_device(A, anew)
			 {
				 cublasDaxpy(handler, N, &alpha, anew, 1, A, 1);
				 cublasIdamax(handler, N, A, 1, &idx);
			 }
			#pragma acc update host(A[idx-1])
			 err = fabs(A[idx - 1]);
			#pragma acc host_data use_device(A, anew)
			 cublasDcopy(handler, n*n, anew, 1, A, 1);
			 printf("%d %lf\n", iter, err);
		 }

		double* tmp = A;
		A = anew;
		anew = tmp;
	}
	}
	// free(A);
	//
	cublasDestroy(handler);
	printf("%d\n%lf", iter, err);

	return 0;
}
