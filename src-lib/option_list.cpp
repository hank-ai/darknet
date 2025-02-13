#include "option_list.hpp"
#include "darknet_internal.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


list *read_data_cfg(const char *filename)
{
	TAT(TATPARMS);

	FILE * file = fopen(filename, "r");
	if (file == nullptr)
	{
		file_error(filename, DARKNET_LOC);
	}

	char *line;
	int line_number = 0;
	list *options = make_list();
	while ((line=fgetl(file)) != 0)
	{
		++line_number;
		strip(line);
		switch(line[0])
		{
			case '\0':
			case '#':
			case ';':
			{
				free(line);
				break;
			}
			default:
			{
				if (not read_option(line, options))
				{
					Darknet::display_warning_msg("WARNING: failed to parse line #" + std::to_string(line_number) + " in " + filename + ": " + line + "\n");
					free(line);
				}
				break;
			}
		}
	}

	fclose(file);

	/* There is a limited number of options that typically exists in the .data files.  We should expect the following:
	 *
	 *		classes = <number>
	 *		train = <filename>
	 *		valid = <filename>
	 *		names = <filename>
	 *		backup = <directory>
	 */

	const char * str = option_find(options, "classes");
	if (str == nullptr)
	{
		Darknet::display_warning_msg("WARNING: expected to find \"classes=...\" in " + std::string(filename) + "\n");
	}
	else
	{
		const int classes = atoi(str);
		if (classes <= 0 or classes >= 50)
		{
			Darknet::display_warning_msg("WARNING: unusual number of classes (" + std::to_string(classes) + ") in " + filename + "\n");
		}
	}

	for (const std::string fn : {"train", "valid", "names"})
	{
		str = option_find(options, fn.c_str());
		if (str == nullptr)
		{
			Darknet::display_warning_msg("WARNING: expected to find \"" + fn + "=...\" in " + filename + "\n");
		}
		else
		{
			// does this file actually exist?
			if (std::filesystem::exists(str) == false)
			{
				Darknet::display_warning_msg("WARNING: file " + std::string(str) + " does not seem to exist (\"" + fn + "=...\") in " + filename + "\n");
			}
		}
	}

	str = option_find(options, "backup");
	if (str == nullptr)
	{
		Darknet::display_warning_msg("WARNING: expected to find \"backup=...\" in " + std::string(filename) + "\n");
	}
	else
	{
		if (std::filesystem::is_directory(str) == false)
		{
			Darknet::display_warning_msg("WARNING: \"" + std::string(str) + "\" does not seem to be a valid directory for \"backup=...\" in " + filename + "\n");
		}
	}

	// see if there are options we don't recognize
	node * n = options->front;
	while (n)
	{
		kvp * p = (kvp *)n->val;
		if (p->used == 0)
		{
			Darknet::display_warning_msg("WARNING: unexpected option \"" + std::string(p->key) + "=" + p->val + "\" in " + filename + "\n");
		}
		n = n->next;
	}

	return options;
}


int read_option(char *s, list *options)
{
	TAT(TATPARMS);

	size_t len = strlen(s);
	char * val = nullptr;
	bool found = false;

	// expect the line to be KEY=VAL so look for the "="
	for (auto i = 0; i < len; ++i)
	{
		if (s[i] == '=')
		{
			found = true;
			s[i] = '\0';
			val = s + i + 1; // point to the character that comes immediately after "="
			break;
		}
	}

	if (not found)
	{
		return 0;
	}

	char *key = s;
	option_insert(options, key, val);

	return 1;
}


void option_insert(list *l, const char *key, const char *val)
{
	TAT(TATPARMS);

	kvp* p = (kvp*)xmalloc(sizeof(kvp));
	p->key = key;
	p->val = val;
	p->used = 0;
	list_insert(l, p);
}

void option_unused(list *l)
{
	TAT(TATPARMS);

	kvp * previous_kvp = NULL;

	node *n = l->front;
	while(n)
	{
		kvp *p = (kvp *)n->val;
		if (!p->used)
		{
			if (previous_kvp)
			{
				// attempt to give some context as to where the error is happening in the .cfg file
				*cfg_and_state.output														<< std::endl
					<< "Last option was: " << previous_kvp->key	<< "=" << previous_kvp->val << std::endl
					<< "Unused option is " << p->key			<< "=" << p->val			<< std::endl;
			}
			darknet_fatal_error(DARKNET_LOC, "invalid, unused, or unrecognized option: %s=%s", p->key, p->val);
		}
		previous_kvp = p;
		n = n->next;
	}
}

const char *option_find(list *l, const char *key)
{
	TAT(TATPARMS);

	node *n = l->front;
	while(n)
	{
		kvp *p = (kvp *)n->val;
		if(strcmp(p->key, key) == 0)
		{
			p->used = 1;
			return p->val;
		}
		n = n->next;
	}
	return nullptr;
}

const char *option_find_str(list *l, const char *key, const char *def)
{
	TAT(TATPARMS);

	const char *v = option_find(l, key);
	if(v)
	{
		return v;
	}

	if (def)
	{
		*cfg_and_state.output << key << ": Using default \"" << def << "\"" << std::endl;
	}

	return def;
}

const char *option_find_str_quiet(list *l, const char *key, const char *def)
{
	TAT(TATPARMS);

	const char *v = option_find(l, key);
	if (v)
	{
		return v;
	}
	return def;
}

int option_find_int(list *l, const char *key, int def)
{
	TAT(TATPARMS);

	const char *v = option_find(l, key);
	if (v)
	{
		return atoi(v);
	}

	*cfg_and_state.output << key << ": Using default \"" << def << "\"" << std::endl;

	return def;
}

int option_find_int_quiet(list *l, const char *key, int def)
{
	TAT(TATPARMS);

	const char *v = option_find(l, key);
	if (v)
	{
		return atoi(v);
	}

	return def;
}

float option_find_float_quiet(list *l, const char *key, float def)
{
	TAT(TATPARMS);

	const char *v = option_find(l, key);
	if (v)
	{
		return atof(v);
	}

	return def;
}

float option_find_float(list *l, const char *key, float def)
{
	TAT(TATPARMS);

	const char *v = option_find(l, key);
	if(v)
	{
		return atof(v);
	}

	*cfg_and_state.output << key << ": Using default \"" << def << "\"" << std::endl;

	return def;
}
