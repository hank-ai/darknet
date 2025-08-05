#pragma once

#include "darknet_internal.hpp"

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
[[noreturn]] void darknet_fatal_error(const char * const filename, const char * const funcname, const int line, const char * const msg, ...);

/// Convert the given size to a human-readable string.  This uses 1024 as a divider, so 1 KiB == 1024 bytes.
const char * size_to_IEC_string(const size_t size);

int *read_map(const char *filename);
void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
const char * basecfg(const char * cfgfile);
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
char *copy_string(char *s);
void normalize_array(float *a, int n);
void scale_array(float *a, int n, float s);
void translate_array(float *a, int n, float s);
int max_index(float *a, int n);
int top_max_index(float *a, int n, int k);
float constrain(float min, float max, float a);

float mse_array(float *a, int n);

float sum_array(float *a, int n);
float mean_array(float *a, int n);
void mean_arrays(float **a, int n, int els, float *avg);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
float dist_array(float *a, float *b, int n, int sub);
float sec(clock_t clocks);
int find_int_arg(int argc, char **argv, const char * const arg, int def);
float find_float_arg(int argc, char **argv, const char * const arg, float def);
int find_arg(int argc, char* argv[], const char * const arg);
const char * find_char_arg(int argc, char **argv, const char *arg, const char *def);
void print_statistics(float *a, int n);

/** In V5 this was modified to use @p std::uniform_real_distribution to return proper C++ pseudo random float values.
 * The @p "min" is inclusive, and @p "max" is exclusive, so @p rand_uniform(0.0f, 5.0f) will return @p 0.0f but never
 * @p 5.0f.  The @p "min" and @p "max" values will automatically be swapped if necessary.  This will never return
 * @p NaN or infinite values.
 */
float rand_uniform(float min, float max);

/** Somewhat similar to @ref rand_uniform(), but will initialize many values at once instead of having to repeatedly
 * call @ref rand_uniform().  Especially useful when initializing a large number of weights on startup.
 *  @since 2025-07-16
 */
void rand_uniform_many(float * x, const size_t n, float min, float max, const float scale = 1.0f);

float rand_scale(float s);

/// The @p "min" and @p "max" values are inclusive.  For example, @p rand_uint(1, 6) can return 6 possible values.
unsigned int rand_uint(unsigned int min=0, unsigned int max=std::numeric_limits<unsigned int>::max());

/// The @p "min" and @p "max" values are inclusive.  For example, @p rand_int(1, 6) can return 6 possible values.
int rand_int(int min=0, int max=std::numeric_limits<int>::max());

float rand_precalc_random(float min, float max, float random_part);

static inline float rand_normal()
{
	TAT(TATPARMS);
	return rand_uniform(-5.0f, 5.0f);
}

static inline float rand_float()
{
	TAT(TATPARMS);
	return rand_uniform(0.0f, 1.0f);
}

static inline bool rand_bool()
{
	TAT(TATPARMS);
	return rand_int(0, 1) ? false : true;
}

int int_index(int *a, int val, int n);
int make_directory(char *path, int mode);

unsigned long custom_hash(const std::string & str);

#define max_val_cmp(a,b) (((a) > (b)) ? (a) : (b))
#define min_val_cmp(a,b) (((a) < (b)) ? (a) : (b))
