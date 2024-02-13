#pragma once

#include "darknet.h"
#include "list.hpp"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
	char *key;
	char *val;
	int used;
} kvp;

/// Read the .data file.
list *read_data_cfg(char *filename);

/** Parse a key-value pair from a single line of text that came from a @p .cfg or @p .data file.
 *
 * @returns @p 0 if the line does not contain a key-value pair.
 * @returns @p 1 if a key-value pair was parsed and stored in @p options.
 */
int read_option(char *s, list *options);

void option_insert(list *l, char *key, char *val);
char *option_find(list *l, const char *key);
char *option_find_str(list *l, char *key, char *def);
char *option_find_str_quiet(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);
float option_find_float(list *l, char *key, float def);
float option_find_float_quiet(list *l, char *key, float def);
void option_unused(list *l);

#ifdef __cplusplus
}
#endif
