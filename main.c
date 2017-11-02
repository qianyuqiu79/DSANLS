#include "common.h"


void random_initialization(Matrix A, FLOAT upper)
{
	VSLStreamStatePtr random_stream;
	vslNewStream(&random_stream, VSL_BRNG_SFMT19937, (unsigned int)time(NULL));
	vRngUniform(VSL_RNG_METHOD_UNIFORM_STD, random_stream, A.numRow * A.numCol, A.values, 0, upper);
}


int main(int argc, char **argv) {
	int k = 0;
	float row_ratio = 0.1, col_ratio = 0.1;
	float alpha = 0.0, beta = 1.0;
	float upper_bound = 1.0; 
	int max_iter = MAX_ITER;
	bool use_gd = false;
	int verbose_interval = 1;
	SketchMethod sketch_method = SUBSAMPLE;

	if (argc < 3) {
		printf("Error: input file and k must be specified\n");
		return -1;
	}

	sscanf(argv[2], "%d", &k);
	if (k <= 0) {
		printf("Error: k must be an positive integer\n");
		return -1;
	}

	// parsing extra options
	int i = 3;
	while (i < argc) {
		if (argv[i][0] != '-') {
			printf("Error: wrong option format \"%s\"\n", argv[i]);
			return -1;
		}

		switch (argv[i][1])
		{
			// number of total iteration
		case 'i':
			max_iter = atoi(argv[i + 1]);
			if (max_iter <= 0) {
				printf("Error: iteration number must be an positive integer\n");
				return -1;
			}
			i += 2;
			break;

			// sketch ratio
		case 's':
			sscanf(argv[i + 1], "%f", &row_ratio);
			sscanf(argv[i + 2], "%f", &col_ratio);
			if (row_ratio <= 0 || row_ratio > 1 || col_ratio <= 0 || col_ratio > 1) {
				printf("Error: proportion size must be within (0, 1]\n");
				return -1;
			}
			i += 3;
			break;

			// verbose frequency
		case 'o':
			verbose_interval = atoi(argv[i + 1]);
			i += 2;
			break;

			// sketch method
		case 'm':
			sketch_method = atoi(argv[i + 1]);
			i += 2;
			break;

			// step size
		case 't':
			sscanf(argv[i + 1], "%f", &alpha);
			sscanf(argv[i + 2], "%f", &beta);
			i += 3;
			break;

			// use gradient tdescent to solve sub-problem
		case 'g':
			use_gd = true;
			i += 1;
			break;

			// the upper bound of the random intialization
		case 'u':
			sscanf(argv[i + 2], "%f", &beta);
			i += 2;
			break;

		default:
			printf("Error: unkown option \"%s\"\n", argv[i]);
			return -1;
			break;
		}
	}

	// initialize MPI
	LocalData data;
	MPI_Init(NULL, NULL);
	data.comm = MPI_COMM_WORLD;
	MPI_Comm_size(MPI_COMM_WORLD, &(data.size));
	MPI_Comm_rank(MPI_COMM_WORLD, &(data.rank));

	// read data and distribute to other nodes
	read_and_distribute(argv[1], &data);
	if (data.rank == MPI_ROOT_RANK)
		printf("Problem size: %d * %d\n", data.numRow, data.numCol);

	// initialize U and V
	Matrix U = create_dense_matrix(data.rowPartition[data.rank + 1] - data.rowPartition[data.rank], k);
	Matrix V = create_dense_matrix(data.colPartition[data.rank + 1] - data.colPartition[data.rank], k);

#if DEBUG_MODE
	for (int i = 0; i < U.numNonzero; i++)
		U.values[i] = 1.0;
	for (int i = 0; i < V.numNonzero; i++)
		V.values[i] = 1.0;
#else
	random_initialization(U, upper_bound);
	random_initialization(V, upper_bound);
#endif

	dsanls(data, sketch_method, row_ratio, col_ratio, max_iter, alpha, beta, use_gd, verbose_interval, U, V);

	return 0;
}