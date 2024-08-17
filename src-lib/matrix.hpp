#pragma once

#include "darknet_internal.hpp"

//typedef struct matrix{
//    int rows, cols;
//    float **vals;
//} matrix;

typedef struct {
    int *assignments;
    matrix centers;
} model;

model do_kmeans(matrix data, int k);
matrix make_matrix(int rows, int cols);
void free_matrix(matrix & m);

float matrix_topk_accuracy(matrix truth, matrix guess, int k);
