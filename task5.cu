//include libries
// mpic++ tp5.cu
//mpiexec -m "кол-во процессов"
//nsys profile —trace=cuda, mpi a.out 0.000001 512 301
//mpirun -n 2 a.out 0.000001 128 1000000
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>
#include <time.h>


#define CALCULATE(matrixA, matrixB, size, i, j) \
	matrixB[i * size + j] = 0.25 * (matrixA[i * size + j - 1] + matrixA[(i - 1) * size + j] + \
			matrixA[(i + 1) * size + j] + matrixA[i * size + j + 1]);	

__global__ void calculateBoundaries(double* A1, double* anew1, int n, int sizegroup)
{
	//расчет границ
	int idxUp = blockIdx.x * blockDim.x + threadIdx.x;
	int idxDown = blockIdx.x * blockDim.x + threadIdx.x;

	if (idxUp == 0 || idxUp > n - 2) return;
	
	if(idxUp < n)
	{
		/*sizegroup-2 - строка*/
		anew1[1 * n + idxUp] = 0.25 * (A1[1 * n + idxUp - 1] + A1[(1 - 1) * n + idxUp] + \
			A1[(1 + 1) * n + idxUp] + A1[1 * n + idxUp + 1]);
		anew1[(sizegroup - 2) * n + idxDown] = 0.25 * (A1[(sizegroup - 2) * n + idxDown - 1] + A1[((sizegroup - 2) - 1) * n + idxDown] + \
			A1[((sizegroup - 2) + 1) * n + idxDown] + A1[(sizegroup - 2) * n + idxDown + 1]);
	}
}

//calculate error
__global__ void error(double * A1, double *anew1 , double * res, int n, int sizegroup){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i * n + j; //find index
	if(!(j == 0 || i == 0 || j == n - 1 || i == sizegroup - 1)){
		res[idx] = fabs(anew1[idx]-A1[idx]);
	}
}
//calculate new value of an array cell 
__global__ void compute(double * A1, double * anew1,int n, int sizegroup){
	int j= blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
	if(!(j < 1 || i < 2 || j > n - 2 || i > sizegroup - 2))
		anew1[i * n + j] = 0.25 * (A1[i * n + j - 1] + A1[(i - 1) * n + j] + \
			A1[(i + 1) * n + j] + A1[i * n + j + 1]);
}

int findNearestPowerOfTwo(int num) {
    int power = 1;
    while (power < num) {
        power <<= 1;
    }
    return power;
}
int main(int argc, char** argv) {
	int rank,size_group; //номер текущего процесса, кол-во процессов
	MPI_Init(&argc, &argv);
	//узнаем номер процесса и кол-во процессов
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size_group);
	// каждому процессу по видеокарте
	cudaSetDevice(rank);
	
	//parametrs from command line
	int iter_max;
	int sizeofnet1;
	double accuracy;

	
	if (argc<4){
		printf("not enough arguments");
		return 0;
	}
	accuracy = atof(argv[1]);
	sizeofnet1 = atoi(argv[2]);
	int n = sizeofnet1;
	iter_max = atoi(argv[3]);
	//check
	if ( accuracy<0 || n<0 || iter_max<0){
		printf("wrong arguments");
		return 0;
	}

	//allocate memory for arrays
	double* A = (double*)calloc(n * n, sizeof(double));
	double* anew = (double*)calloc(n * n, sizeof(double));
	//calculate boundary conditions
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

	//потоки для поиска границ и расчета матриц
	cudaStream_t stream, matrixCalculationStream;
	cudaStreamCreate ( &stream);
	cudaStreamCreate(&matrixCalculationStream);
	// ищем границы между устройствами
	int sizeofArrayProcces = n/size_group;
	//индекс элемента, с которого начинается блок данных, который мы отправим на конкретную видеокарту
	int start_cell = sizeofArrayProcces*rank;
	
	int iter = 0;
	void* t_memory = NULL;
	size_t t_memory_size = 0;
	double* A1 = NULL;
	double* anew1 = NULL;
	double* tmp_arr = NULL;
	double err = 100;
	double* err1 = 0;

	// Расчитываем, сколько памяти требуется процессу
    //необходимо знать границы предыдущего и следующего блоков, так как вычисления, по сути, идут крестиком.
	if (rank != 0 && rank != size_group - 1)
	{
		sizeofArrayProcces += 2;
	}
	else 
	{
		sizeofArrayProcces += 1;
	}

	//allocate array with cuda function
	cudaMalloc((&err1), sizeof(double));
	cudaMalloc((&A1), sizeof(double) * sizeofArrayProcces * n);
	cudaMalloc((&anew1), sizeof(double) * sizeofArrayProcces * n);
	cudaMalloc((&tmp_arr), sizeof(double) * sizeofArrayProcces * n);

	// Копируем часть заполненной матрицы в выделенную память, начиная с 1 строки
	int move;
	if (rank==0)
		move = 0;
	else
		move = n;

	//Если процесс не первый, то нам нужно вести расчеты вместе с предыдущей строкой
	cudaMemcpy(A1, A + (start_cell*n) - move, sizeof(double) * sizeofArrayProcces * n, cudaMemcpyHostToDevice);
	cudaMemcpy(anew1, anew + (start_cell*n) - move, sizeof(double) * sizeofArrayProcces * n, cudaMemcpyHostToDevice);

	//кол-во потоков в процессе - кратно 32, ориентируясь на размер матрицы
	int threads_x = min(findNearestPowerOfTwo(n), 1024);
	//столько же, сколько строк в матрице
    int blocks_y = sizeofArrayProcces ;
	//Если сетка меньше 1024, то блок - занимает всю строку матрицы, если больше, то часть строки
    int blocks_x = n / threads_x;

    dim3 blockDim(threads_x, 1); // кол-во потоков в блоке
    dim3 gridDim(blocks_x, blocks_y); //кол-во блоков на сетке

	//// t_memory = NULL, при первом вызове возвращает нужный размер, нужно выделить память, чтобы функция работала
	cub::DeviceReduce::Max(t_memory, t_memory_size, tmp_arr, err1, n*sizeofArrayProcces);
	cudaMalloc((&t_memory), t_memory_size);
	//запускаем отсчет времени
	clock_t begin = clock();
	//main algorithm
	for (iter = 0; iter < iter_max && err>accuracy; iter++) {
		// Расчитываем границы, которые потом будем отправлять другим процессам
		calculateBoundaries<<<n, 1, 0, stream>>>(A1, anew1, 
										n, sizeofArrayProcces);
		// ждём, пока закончим рассчитывать границы, чтобы иметь возвожность отправлять результаты расчётов границ
		cudaStreamSynchronize(stream);
		// Расчет матрицы
		compute<<<gridDim, blockDim, 0, matrixCalculationStream>>>(A1, anew1, n, sizeofArrayProcces);
		//every 100 iterations calculate error
		if (iter%100==0){
			error<<<gridDim, blockDim, 0, matrixCalculationStream>>>(A1, anew1, tmp_arr, n, sizeofArrayProcces);
			cub::DeviceReduce::Max(t_memory, t_memory_size, tmp_arr, err1, n*sizeofArrayProcces, matrixCalculationStream);
			cudaStreamSynchronize(matrixCalculationStream);
			// Находим максимальную ошибку среди всех и передаём её всем процессам
			MPI_Allreduce((void*)err1, (void*)err1, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			//чтобы начать передавать в новом потоке
			cudaMemcpyAsync(&err, err1, sizeof(double), cudaMemcpyDeviceToHost, matrixCalculationStream);
		}
		// Обмен "граничными" условиями каждой области
		// Обмен верхней границей
		if (rank != 0)
		{
		    MPI_Sendrecv(anew1 + n + 1, n - 2, MPI_DOUBLE, rank - 1, 0, 
			anew1 + 1, n - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		// Обмен нижней границей
		if (rank != size_group - 1)
		{
		   MPI_Sendrecv(anew1 + (sizeofArrayProcces - 2) * n + 1, n - 2, MPI_DOUBLE, rank + 1, 0,
							anew1 + (sizeofArrayProcces - 1) * n + 1, 
							n - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	
		cudaStreamSynchronize(matrixCalculationStream);//дожидаемся окончания расчёта матрицы
		//change arrays
		double* tmp = A1;
		A1 = anew1;
		anew1 = tmp;
	}
	clock_t end = clock();
	//print answer
	if (rank == 0)
	{
		printf("%d\n%lf\n", iter, err);
		printf("%lf\n", 1.0*(end-begin)/CLOCKS_PER_SEC);
	}
	//free memory
	cudaFree(err1);
	cudaFree(t_memory);
	cudaFree(A1);
	cudaFree(anew1);
	free(A);
	free(anew);
	MPI_Finalize();
	return 0;
}
