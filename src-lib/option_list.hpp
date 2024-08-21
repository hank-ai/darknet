#pragma once

#include "darknet_internal.hpp"

typedef struct
{
	const char *key;
	const char *val;
	int used;
} kvp;

/// Read the .data file.
list *read_data_cfg(const char *filename);

/** Parse a key-value pair from a single line of text that came from a @p .cfg or @p .data file.
 *
 * @returns @p 0 if the line does not contain a key-value pair.
 * @returns @p 1 if a key-value pair was parsed and stored in @p options.
 */
int read_option(char *s, list *options);

void option_insert(list *l, const char *key, const char *val);
const char *option_find(list *l, const char *key);
const char *option_find_str(list *l, const char *key, const char *def);
const char *option_find_str_quiet(list *l, const char *key, const char *def);
int option_find_int(list *l, const char *key, int def);
int option_find_int_quiet(list *l, const char *key, int def);
float option_find_float(list *l, const char *key, float def);
float option_find_float_quiet(list *l, const char *key, float def);
void option_unused(list *l);
