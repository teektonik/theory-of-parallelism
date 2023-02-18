#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <time.h>

#define n 10000000
#define pi 3.14159265358979323846


int main() {
    clock_t start, end;
    start = clock();

    double* arr0 = (double*)malloc(sizeof(double) * n);

    double a0 = (2 * pi) / (n - 1);

    double sum0 = 0;

#pragma acc data create(arr0[:n])
    {
#pragma acc parallel loop vector vector_length(32) gang
        for (int i = 0; i < n; i++) {
            arr0[i] = sin(a0 * i);

        }



#pragma acc parallel loop reduction(+:sum0)
        for (int i = 0; i < n; i++) {
            sum0 += arr0[i];

        }
    }
    printf("%-32.25f\n", sum0);
    end = clock();

    printf("%.4f second(s)\n", ((float)end - start) / ((float)CLOCKS_PER_SEC));
    return 0;



}
