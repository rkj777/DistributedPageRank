/* This is the main code for Lab4 MPI Pagerank algorithm
*	Rajan Jassal
*	Eric Smith
*/
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "Lab4_IO.h"
#include "timer.h"

#define CENTRAL_PROC 0

int main() {
	
	int num_processes, myRank;

	/* Loop Variables */
	int i, j;
	double start, end;

	/* Begin MPI */
	MPI_Init(NULL, NULL);

	/* Get number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

	/* Get myRank */
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	struct node* nodes;
	int* num_in_links;
	int* num_out_links;
	int N;

	get_node_stat(&N, &num_in_links, &num_out_links);

	int local_N = (N/num_processes);
	int lowerBound = local_N * (myRank);
	int upperBound = local_N * (myRank+1);

	node_init(&nodes, num_in_links, num_out_links, lowerBound, upperBound);

	/* Variables for page rank calculation */
	bool error_check_passed;
	double bootstrap_sum;
	double error;

	//Fun fact: page rank is not actually named for the the ranking of the page but for Google co-founder Larry Page
	double * page_rank = malloc(sizeof(double) * (N/num_processes));
	double * collected_rank = malloc(sizeof(double) * N);
	double * old_rank = NULL;

	//Constants
	const double init_odds = 1.0/N;
	const double damping_constant = 0.85;
	const double static_odds = (1.0 - damping_constant) * init_odds;
	const double max_error = 0.00001;

	if(myRank == 0) {
		old_rank = malloc(sizeof(double) * N);
		for(i=0; i<N; i++) {
			old_rank[i] = init_odds;
		}
		
	}

	// Sets initial values of rank to the odds of picking one page randomly
	for(i=0; i< local_N; i++){
		page_rank[i] = init_odds;
	}
	for(i=0; i<N; i++){
		collected_rank[i] = init_odds;
	}

	int iterationcount = 0;

	error_check_passed = false;
	GET_TIME(start);
	while(!error_check_passed) {

		++iterationcount;
		// Update Page-Ranks
		for(i=0; i < local_N; i++){ //only works in the case that N is divisible by the number of processes
			bootstrap_sum = 0;
			for(j=0; j < nodes[i].num_in_links; j++) {
				bootstrap_sum += collected_rank[nodes[i].inlinks[j]] / num_out_links[nodes[i].inlinks[j]];
			}
			page_rank[i] = static_odds + (damping_constant * bootstrap_sum);
			//printf("Old: %f, New: %f\n", old_rank[i], page_rank[i]);
		}

		//Check for error
		double top_error = 0;
		double bottom_error = 0;
		error = 0;

		// Gather updated Page rank in central process
		MPI_Allgather(page_rank, local_N, MPI_DOUBLE, collected_rank, local_N, MPI_DOUBLE, MPI_COMM_WORLD);

		// If I am central process check error
		if(myRank == 0) {
			for(i=0; i < N; i++) {
				top_error += (collected_rank[i] - old_rank[i]) * (collected_rank[i] - old_rank[i]);
				bottom_error += old_rank[i] * old_rank[i];
			}
			error = sqrt(top_error) / sqrt(bottom_error);
			if(error < max_error) {
				error_check_passed = true;
			}
		}
		
		// If send out signal to each process so they know if they should continue looping
		MPI_Bcast(&error_check_passed, 1, MPI_C_BOOL, CENTRAL_PROC, MPI_COMM_WORLD);
		
		// Update old_rank so error check next time is accurate
		if(myRank == 0) {
			for(i=0; i<N; i++) {
				old_rank[i] = collected_rank[i];
			}
		}
	}

	// Save final Pagerank to output file.
	if(myRank == 0) {
		Lab4_saveoutput(collected_rank, N, 0.0);
		GET_TIME(end);
		printf("Running time: %f\n", end - start);
		free(old_rank);
	}

	// Free data
	node_destroy(nodes, local_N);
	free(page_rank);
	free(collected_rank);
	free(num_in_links);
	free(num_out_links);

	MPI_Finalize();
	return 0;
}
