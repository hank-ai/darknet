#pragma once

//#include <pthread.h>

#include "darknet.h"
#include "matrix.hpp"
#include "list.hpp"
#include "tree.hpp"

#ifdef __cplusplus
extern "C" {
#endif

static inline float distance_from_edge(int x, int max)
{
    int dx = (max/2) - x;
    if (dx < 0) dx = -dx;
    dx = (max/2) + 1 - dx;
    dx *= 2;
    float dist = (float)dx/max;
    if (dist > 1) dist = 1;
    return dist;
}

void print_letters(float *pred, int n);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h, int c);
data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int truth_size, int classes, int use_flip, int gaussian_noise, int use_blur, int use_mixup,
    float jitter, float resize, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int mosaic_bound, int contrastive, int contrastive_jit_flip, int contrastive_color, int show_imgs);
data load_data_tag(char **paths, int n, int m, int k, int use_flip, int min, int max, int w, int h, int c, float angle, float aspect, float hue, float saturation, float exposure);
matrix load_image_augment_paths(char **paths, int n, int use_flip, int min, int max, int w, int h, int c, float angle, float aspect, float hue, float saturation, float exposure, int contrastive);
data load_data_super(char **paths, int n, int m, int w, int h, int c, int scale);
data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int use_flip, int min, int max, int w, int h, int c, float angle,
    float aspect, float hue, float saturation, float exposure, int use_mixup, int use_blur, int show_imgs, float label_smooth_eps, int contrastive);
data load_go(char *filename);

box_label *read_boxes(char *filename, int *n);
data load_cifar10_data(char *filename);
data load_all_cifar10();

data load_data_writing(char** paths, int n, int m, int w, int h, int c, int out_w, int out_h);
list *get_paths(char *filename);
char **get_labels(char *filename);
char **get_labels_custom(char *filename, int *size);
void get_random_batch(data d, int n, float *X, float *y);
data get_data_part(data d, int part, int total);
data get_random_data(data d, int num);
void get_next_batch(data d, int n, int offset, float *X, float *y);
data load_categorical_data_csv(char *filename, int target, int k);
void normalize_data_rows(data d);
void scale_data_rows(data d, float s);
void translate_data_rows(data d, float s);
void randomize_data(data d);
data *split_data(data d, int part, int total);
data concat_data(data d1, data d2);
data concat_datas(data *d, int n);
void fill_truth(char *path, char **labels, int k, float *truth);
void fill_truth_smooth(char *path, char **labels, int k, float *truth, float label_smooth_eps);

#ifdef __cplusplus
}
#endif
