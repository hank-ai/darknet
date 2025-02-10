#include "darknet_internal.hpp"

void free_matrix(matrix & m)
{
	TAT_REVIEWED(TATPARMS, "2024-04-04");

	for (int i = 0; i < m.rows; ++i)
	{
		free(m.vals[i]);
		m.vals[i] = nullptr;
	}
	free(m.vals);
	m.vals = nullptr;
}

float matrix_topk_accuracy(matrix truth, matrix guess, int k)
{
	TAT(TATPARMS);

	int* indexes = (int*)xcalloc(k, sizeof(int));
	int n = truth.cols;
	int i,j;
	int correct = 0;
	for(i = 0; i < truth.rows; ++i){
		top_k(guess.vals[i], n, k, indexes);
		for(j = 0; j < k; ++j){
			int class_id = indexes[j];
			if(truth.vals[i][class_id]){
				++correct;
				break;
			}
		}
	}
	free(indexes);
	return (float)correct/truth.rows;
}

matrix make_matrix(int rows, int cols)
{
	TAT(TATPARMS);

	int i;
	matrix m;
	m.rows = rows;
	m.cols = cols;
	m.vals = (float**)xcalloc(m.rows, sizeof(float*));
	for(i = 0; i < m.rows; ++i){
		m.vals[i] = (float*)xcalloc(m.cols, sizeof(float));
	}
	return m;
}


//matrix make_matrix(int rows, int cols);

void copy(float *x, float *y, int n);
float dist(float *x, float *y, int n);
int *sample(int n);

int closest_center(float *datum, matrix centers)
{
	TAT(TATPARMS);

	int j;
	int best = 0;
	float best_dist = dist(datum, centers.vals[best], centers.cols);
	for (j = 0; j < centers.rows; ++j) {
		float new_dist = dist(datum, centers.vals[j], centers.cols);
		if (new_dist < best_dist) {
			best_dist = new_dist;
			best = j;
		}
	}
	return best;
}

float dist_to_closest_center(float *datum, matrix centers)
{
	TAT(TATPARMS);

	int ci = closest_center(datum, centers);
	return dist(datum, centers.vals[ci], centers.cols);
}

int kmeans_expectation(matrix data, int *assignments, matrix centers)
{
	TAT(TATPARMS);

	int i;
	int converged = 1;
	for (i = 0; i < data.rows; ++i) {
		int closest = closest_center(data.vals[i], centers);
		if (closest != assignments[i]) converged = 0;
		assignments[i] = closest;
	}
	return converged;
}

void kmeans_maximization(matrix data, int *assignments, matrix centers)
{
	TAT(TATPARMS);

	matrix old_centers = make_matrix(centers.rows, centers.cols);

	int i, j;
	int *counts = (int*)xcalloc(centers.rows, sizeof(int));
	for (i = 0; i < centers.rows; ++i) {
		for (j = 0; j < centers.cols; ++j) {
			old_centers.vals[i][j] = centers.vals[i][j];
			centers.vals[i][j] = 0;
		}
	}
	for (i = 0; i < data.rows; ++i) {
		++counts[assignments[i]];
		for (j = 0; j < data.cols; ++j) {
			centers.vals[assignments[i]][j] += data.vals[i][j];
		}
	}
	for (i = 0; i < centers.rows; ++i) {
		if (counts[i]) {
			for (j = 0; j < centers.cols; ++j) {
				centers.vals[i][j] /= counts[i];
			}
		}
	}

	for (i = 0; i < centers.rows; ++i) {
		for (j = 0; j < centers.cols; ++j) {
			if(centers.vals[i][j] == 0) centers.vals[i][j] = old_centers.vals[i][j];
		}
	}
	free(counts);
	free_matrix(old_centers);
}



void random_centers(matrix data, matrix centers)
{
	TAT(TATPARMS);

	int i;
	int *s = sample(data.rows);
	for (i = 0; i < centers.rows; ++i) {
		copy(data.vals[s[i]], centers.vals[i], data.cols);
	}
	free(s);
}

int *sample(int n)
{
	TAT(TATPARMS);

	int i;
	int* s = (int*)xcalloc(n, sizeof(int));
	for (i = 0; i < n; ++i) s[i] = i;
	for (i = n - 1; i >= 0; --i) {
		int swap = s[i];
		int index = rand() % (i + 1);
		s[i] = s[index];
		s[index] = swap;
	}
	return s;
}

float dist(float *x, float *y, int n)
{
	TAT(TATPARMS);

	float mw = (x[0] < y[0]) ? x[0] : y[0];
	float mh = (x[1] < y[1]) ? x[1] : y[1];
	float inter = mw*mh;
	float sum = x[0] * x[1] + y[0] * y[1];
	float un = sum - inter;
	float iou = inter / un;
	return 1 - iou;
}

void copy(float *x, float *y, int n)
{
	TAT(TATPARMS);

	int i;
	for (i = 0; i < n; ++i) y[i] = x[i];
}

model do_kmeans(matrix data, int k)
{
	TAT(TATPARMS);

	matrix centers = make_matrix(k, data.cols);
	int* assignments = (int*)xcalloc(data.rows, sizeof(int));
	random_centers(data, centers);  // IoU = 67.31% after kmeans

	// range centers [min - max] using exp graph or Pyth example
	if (k == 1) kmeans_maximization(data, assignments, centers);
	int i;
	for(i = 0; i < 1000 && !kmeans_expectation(data, assignments, centers); ++i) {
		kmeans_maximization(data, assignments, centers);
	}
	printf("\n iterations = %d \n", i);
	model m;
	m.assignments = assignments;
	m.centers = centers;
	return m;
}
