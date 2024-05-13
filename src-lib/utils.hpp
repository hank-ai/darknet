#pragma once

#include "darknet.h"
#include "list.hpp"

#include <errno.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <limits>
#ifdef __cplusplus
extern "C" {
#endif

#define DARKNET_LOC __FILE__, __func__, __LINE__

void free_ptrs(void **ptrs, int n);
void top_k(float *a, int n, int k, int *index);

/* The "location" is the file, function, and line as defined by the DARKNET_LOC macro.
 * This is then printed when darknet_fatal_error() is called to terminate the instance of darknet.
 */
void *xmalloc_location(const size_t size, const char * const filename, const char * const funcname, const int line);
void *xcalloc_location(const size_t nmemb, const size_t size, const char * const filename, const char * const funcname, const int line);
void *xrealloc_location(void *ptr, const size_t size, const char * const filename, const char * const funcname, const int line);

#define xmalloc(s)      xmalloc_location(s, DARKNET_LOC)
#define xcalloc(m, s)   xcalloc_location(m, s, DARKNET_LOC)
#define xrealloc(p, s)  xrealloc_location(p, s, DARKNET_LOC)

/// Calling this function ends the application.  This function will @em never return control back to the caller.  @see @ref DARKNET_LOC
void darknet_fatal_error(const char * const filename, const char * const funcname, const int line, const char * const msg, ...);

/// Convert the given size to a human-readable string.  This uses 1024 as a divider, so 1 KiB == 1024 bytes.
const char * size_to_IEC_string(const size_t size);

/** Convert the current time -- as seconds -- into a double.  Precision is microseconds (10^-6).
 * It takes 1000 microseconds to make 1 millisecond.
 */
double what_time_is_it_now();

int *read_map(char *filename);
void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
char *basecfg(char *cfgfile);
int alphanum_to_int(char c);
char int_to_alphanum(int i);
int read_int(int fd);
void write_int(int fd, int n);
void read_all(int fd, char *buffer, size_t bytes);
void write_all(int fd, char *buffer, size_t bytes);
int read_all_fail(int fd, char *buffer, size_t bytes);
int write_all_fail(int fd, char *buffer, size_t bytes);
void find_replace(const char* str, char* orig, char* rep, char* output);
void replace_image_to_label(const char* input_path, char* output_path);
void malloc_error(const size_t size, const char * const filename, const char * const funcname, const int line);
void calloc_error(const size_t size, const char * const filename, const char * const funcname, const int line);
void realloc_error(const size_t size, const char * const filename, const char * const funcname, const int line);
void file_error(const char * const s, const char * const filename, const char * const funcname, const int line);
void strip(char *s);
void strip_args(char *s);
void strip_char(char *s, char bad);
list *split_str(char *s, char delim);
char *fgetl(FILE *fp);
list *parse_csv_line(char *line);
char *copy_string(char *s);
int count_fields(char *line);
float *parse_fields(char *line, int n);
void normalize_array(float *a, int n);
void scale_array(float *a, int n, float s);
void translate_array(float *a, int n, float s);
int max_index(float *a, int n);
int top_max_index(float *a, int n, int k);
float constrain(float min, float max, float a);
int constrain_int(int a, int min, int max);
float mse_array(float *a, int n);
float rand_normal();
size_t rand_size_t();
float rand_uniform(float min, float max);
float rand_scale(float s);
int rand_int(int min, int max);
float sum_array(float *a, int n);
float mean_array(float *a, int n);
void mean_arrays(float **a, int n, int els, float *avg);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
float mag_array_skip(float *a, int n, int * indices_to_skip);
float dist_array(float *a, float *b, int n, int sub);
float **one_hot_encode(float *a, int n, int k);
float sec(clock_t clocks);
int find_int_arg(int argc, char **argv, const char * const arg, int def);
float find_float_arg(int argc, char **argv, const char * const arg, float def);
int find_arg(int argc, char* argv[], const char * const arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
int sample_array(float *a, int n);
int sample_array_custom(float *a, int n);
void print_statistics(float *a, int n);
unsigned int random_gen_fast(void);
float random_float_fast();
int rand_int_fast(int min, int max);
unsigned int random_gen(unsigned int min=0, unsigned int max=std::numeric_limits<unsigned int>::max());
float random_float();
float rand_uniform_strong(float min, float max);
float rand_precalc_random(float min, float max, float random_part);
double double_rand(void);
unsigned int uint_rand(unsigned int less_than);
int check_array_is_nan(float *arr, int size);
int check_array_is_inf(float *arr, int size);
int int_index(int *a, int val, int n);
int *random_index_order(int min, int max);
int max_int_index(int *a, int n);
boxabs box_to_boxabs(const box* b, const int img_w, const int img_h, const int bounds_check);
int make_directory(char *path, int mode);
unsigned long custom_hash(char *str);
bool is_live_stream(const char * path);

#define max_val_cmp(a,b) (((a) > (b)) ? (a) : (b))
#define min_val_cmp(a,b) (((a) < (b)) ? (a) : (b))

#ifdef __cplusplus
}
#endif
