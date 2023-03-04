#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <string.h>

double A[1025*1025];
double anew[1025*1025];

int main(int argc, char** argv) {
                int iter_max;
                int sizeofnet1;

                double accuracy;
                accuracy =  atof(argv[1]);

                sizeofnet1 = atoi(argv[2]);

                int n = sizeofnet1;

                iter_max = atoi(argv[3]);
                double err = 100;
                A[0] = 10;
                A[0 + n-1] = 20;
                A[0+n*(n-1)] = 20;
                A[(n - 1) * n + n-1] = 30;
                anew[0] = 10;
                anew[0 + n - 1] = 20;      
                anew[0 + n * (n - 1)] = 20;
                anew[(n - 1) * n + n - 1] = 30;

                double dif1 =(double) 10 / (n - 1);
                #pragma acc parallel loop vector vector_length(128) gang
                for (int i = 1; i < n-1; i++) {
                         A[0+i*n] = A[0+i*n - n] + dif1;
                         A[(n-1)+i*n] = A[(n - 1) + i * n - n] + dif1;
                         A[i] = A[i - 1] + dif1;
                         A[i+(n-1)*(n)] = A[i + (n - 1) * (n)-1] + dif1;
                         anew[0 + i * n] = anew[0 + i * n - n] + dif1;
                         anew[(n - 1) + i * n] = anew[(n - 1) + i * n - n] + dif1;
                         anew[i] = anew[i - 1] + dif1;
                         anew[i + (n - 1) * (n)] = anew[i + (n - 1) * (n)-1] + dif1;
                }

                int iter = 0;




        for (iter = 0; iter < iter_max && err>accuracy; iter++) {
                err = 0;

                #pragma acc parallel loop seq vector vector_length(256) gang num_gangs(256)  copy(err)
                for (int j = 1; j < n - 1; j++) {
                        for (int i = 1; i < n - 1; i++) {
                                anew[i+j*n] = 0.25 * (A[i + j * n -n] + A[i + j * n+n] + A[i + j * n -1] + A[i + j * n +1]);
                                err = fmax(err, anew[i + j * n] - A[i + j * n]);
                                }
                        }
                        memcpy(A, anew, sizeof(A));
                }
        printf("%d\n%lf", iter, err);

                return 0;
}
