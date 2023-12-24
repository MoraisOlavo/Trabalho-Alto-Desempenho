/* Membros do Grupo
 * 	-Eduardo Figueiredo Freire Andrade 11232820
 * 	-Milena Correa da Silva  11795401
 * 	-Olavo Morais Borges Pereira 11297792
 * 	-Pedro Henrique Conrado F de Oliveira 11819091
 * Compilação: mpicc p_distance.c -o par.out -lm -fopenmp
 * Execução: mpirun -np x -bind-to socket ./par.out n seed t
 * Exemplo: mpirun -np 1 -bind-to socket ./par.out 100 42 1
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <mpi.h>
#include <omp.h>

#define MAX_COORD 100.0

//Função que calcula distância de manhattan entre dois pontos
int manhattan_distance(int x1, int y1, int z1, int x2, int y2, int z2) {
	return abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2);
}

//Função que calcula distância euclidiana entre dois pontos
double euclidean_distance(int x1, int y1, int z1, int x2, int y2, int z2) {
	return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
}

int main(int argc, char *argv[]) {
	//Obtém parâmetros para execução do programa
	if (argc != 4) {
		printf("Uso: %s <N> <seed> <t>\n", argv[0]);
		return 1;
	}

	int N = atoi(argv[1]);
	int seed = atoi(argv[2]);
	int t = atoi(argv[3]);


	int rank, size, workers;

	//Cria o comunicador, cada processo recebe seu rank e o número total de processos
	MPI_Status status;
	int provided;
	MPI_Init_thread(&argc, &argv,MPI_THREAD_MULTIPLE,&provided);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//Aloca espaço para as matrizes
	int *x = (int *)malloc(N*N * sizeof(int));
	int *y = (int *)malloc(N*N * sizeof(int));
	int *z = (int *)malloc(N*N * sizeof(int));

	//Variáveis usadas para armazenar as menores distâncias, maiores distâncias e somatórios 
	//Essas variáveis são referentes apenas aos pontos atribuídos à máquina, ao final da execução essas variáveis serão enviadas para
	//a máquina 0, que vai fazer a redução (somar os somatórios, encontrar o menor dos mínimos, etc)
	int min_manhattan = INT_MAX, max_manhattan = 0;
	double min_euclidean = DBL_MAX, max_euclidean = 0.0;
	int sum_min_manhattan = 0, sum_max_manhattan = 0;
	double sum_min_euclidean = 0.0, sum_max_euclidean = 0.0;

	//Máquina 0 cria as matrizes e faz broadcast 
	if(rank == 0){
		srand(seed); 
		for(int i = 0; i < N*N; i++){
			x[i] = rand() % (int)MAX_COORD;
		}
		for(int i = 0; i < N*N; i++){
			y[i] = rand() % (int)MAX_COORD;
		}
		for(int i = 0; i < N*N; i++){
			z[i] = rand() % (int)MAX_COORD;
		}

		MPI_Bcast(&(x[0]), N*N, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(y[0]), N*N, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(z[0]), N*N, MPI_INT, 0, MPI_COMM_WORLD);

	}else{
		//Demais máquinas recebem a matriz da máquina 0
		MPI_Bcast(&(x[0]), N*N, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(y[0]), N*N, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(z[0]), N*N, MPI_INT, 0, MPI_COMM_WORLD);
	}

	//Variáveis usadas nos loops 
	int ij;
	int k;

	//Variáveis usadas para armazenar as menores e maiores distâncias do ponto ij, são redefinidas a cada iteração do for externo
	int local_max_manhattan, local_min_manhattan;
	double local_max_euclidean, local_min_euclidean;

	//Variáveis usadas para armazenar a distância calculada
	int manhattan_dist;
	double euclidean_dist;

	//Cria a região paralela com t threads
#pragma omp parallel num_threads(t) \
	private(ij,k,local_max_euclidean,local_min_euclidean,local_max_manhattan,local_min_manhattan,manhattan_dist,euclidean_dist) \
	reduction(min: min_manhattan, min_euclidean) \
	reduction (max: max_euclidean, max_manhattan) \
	reduction(+: sum_max_euclidean, sum_min_euclidean, sum_max_manhattan, sum_min_manhattan)
	{
		//Divide as iterações do for entre as threads
#pragma omp for schedule(dynamic,1)
		for (ij = rank; ij < N*N; ij+=size) {
			local_min_manhattan = INT_MAX;
			local_max_manhattan = 0;
			local_min_euclidean = DBL_MAX;
			local_max_euclidean = 0.0;


			//Usa a extensão SIMD
#pragma omp simd reduction(min : local_min_manhattan, local_min_euclidean) reduction(max : local_max_manhattan, local_max_euclidean)
			for (k = ij+1; k < N*N; k++) {

				manhattan_dist = manhattan_distance(x[ij], y[ij], z[ij], x[k], y[k], z[k]);
				euclidean_dist = euclidean_distance(x[ij], y[ij], z[ij], x[k], y[k], z[k]);

				if (manhattan_dist < local_min_manhattan) { 
					local_min_manhattan = manhattan_dist;
				}
				if (manhattan_dist > local_max_manhattan) {
					local_max_manhattan = manhattan_dist;
				}

				if (euclidean_dist < local_min_euclidean) {
					local_min_euclidean = euclidean_dist;
				}
				if (euclidean_dist > local_max_euclidean) {
					local_max_euclidean = euclidean_dist;
				}
			}//Fim do for interno


			if (local_min_manhattan < min_manhattan)
				min_manhattan = local_min_manhattan;
			if (local_max_manhattan > max_manhattan)
				max_manhattan = local_max_manhattan;

			if (local_min_euclidean < min_euclidean)
				min_euclidean = local_min_euclidean;
			if (local_max_euclidean > max_euclidean)
				max_euclidean = local_max_euclidean;

			if (local_min_manhattan != INT_MAX && local_min_euclidean != DBL_MAX){
				sum_max_manhattan += local_max_manhattan;
				sum_max_euclidean += local_max_euclidean;
				sum_min_manhattan += local_min_manhattan;
				sum_min_euclidean += local_min_euclidean;
			}

		}//Fim do for externo
	}//Fim da região paralela

	//Variáveis usadas para armazenar o resultado das operações de redução entre as máquinas
	int global_sum_min_manhattan, global_sum_max_manhattan;
	double global_sum_min_euclidean, global_sum_max_euclidean;
	int global_min_manhattan, global_max_manhattan;
	double global_min_euclidean, global_max_euclidean;

	//Faz a redução das variáveis
	MPI_Reduce(&min_manhattan, &global_min_manhattan, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&max_manhattan, &global_max_manhattan, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&min_euclidean, &global_min_euclidean, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&max_euclidean, &global_max_euclidean, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&sum_min_manhattan, &global_sum_min_manhattan, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&sum_max_manhattan, &global_sum_max_manhattan, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&sum_min_euclidean, &global_sum_min_euclidean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&sum_max_euclidean, &global_sum_max_euclidean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	//Máquina 0 exibe o resultado
	if (rank == 0) {
		printf("Distância de Manhattan mínima: %d (soma min: %d) e máxima: %d (soma max: %d).\n", global_min_manhattan, global_sum_min_manhattan, global_max_manhattan, global_sum_max_manhattan);
		printf("Distância Euclidiana mínima: %.2lf (soma min: %.2lf) e máxima: %.2lf (soma max: %.2lf).\n", global_min_euclidean, global_sum_min_euclidean, global_max_euclidean, global_sum_max_euclidean);
	}

	//Libera espaço alocado
	free(x);
	free(y);
	free(z);

	MPI_Finalize();

	return 0;
}
