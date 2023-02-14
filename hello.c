#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <time.h>

#define n 10000000
#define pi 3.14159265358979323846


int main() {
	

	float* arr0 = (float*)malloc(sizeof(float)*n);
	double* arr1 = (double*)malloc(sizeof(double)*n);
	float a0 = (2 * pi) / (n - 1);
	double a1 = (2 * pi) / (n - 1);
	
	float s0 = 0;
	double s1 = 0;
	
	#pragma acc kernels
	for (int i = 0; i < n; i++) {
		arr0[i] = sin(s0);
		arr1[i] = sin(s1);
		s0 += a0;
		s1 += a1;

	}
	
	float sum0 = 0;
	double sum1 = 0;
	#pragma acc kernels
	for (int i = 0; i < n; i++) {
		sum0 += arr0[i];
		sum1 += arr1[i];
	}
	printf("%f %lf", sum0, sum1);

	return 0;



}
