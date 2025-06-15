#define _CRT_RAND_S

#include <csignal>
#include <regex>
#include "darknet_internal.hpp"
#include "darkunistd.hpp"

#ifdef WIN32
#include <dbghelp.h>
#pragma comment(lib, "DbgHelp.lib")
#else
#include <sys/time.h>
#include <sys/stat.h>
#include <execinfo.h>
#endif
#include <random>


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


void *xmalloc_location(const size_t size, const char * const filename, const char * const funcname, const int line)
{
	TAT(TATPARMS);

	void *ptr=malloc(size);
	if (!ptr)
	{
		malloc_error(size, filename, funcname, line);
	}
	return ptr;
}

void *xcalloc_location(const size_t nmemb, const size_t size, const char * const filename, const char * const funcname, const int line)
{
	TAT(TATPARMS);

	void *ptr=calloc(nmemb, size);
	if (!ptr)
	{
		calloc_error(nmemb * size, filename, funcname, line);
	}
	return ptr;
}

void *xrealloc_location(void *ptr, const size_t size, const char * const filename, const char * const funcname, const int line)
{
	TAT(TATPARMS);

	ptr=realloc(ptr,size);
	if (!ptr)
	{
		realloc_error(size, filename, funcname, line);
	}
	return ptr;
}

int *read_map(const char *filename)
{
	TAT_COMMENT(TATPARMS, "realloc nightmare");

	/// @todo what is this "map" file that we're reading in?

	int n = 0;
	int *map = 0;
	char *str;
	FILE *file = fopen(filename, "r");
	if (!file)
	{
		file_error(filename, DARKNET_LOC);
	}

	while((str=fgetl(file)))
	{
		++n;
		/// @todo the while loop reallocs the array at every iteration, this needs to be refactored!
		map = (int*)xrealloc(map, n * sizeof(int));
		map[n-1] = atoi(str);
		free(str);
	}
	if (file) fclose(file);
	return map;
}

void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections)
{
	TAT(TATPARMS);

	for (size_t i = 0; i < sections; ++i)
	{
		size_t start = n*i/sections;
		size_t end = n*(i+1)/sections;
		size_t num = end-start;
		shuffle((char*)arr+(start*size), num, size);
	}
}

void shuffle(void *arr, size_t n, size_t size)
{
	TAT(TATPARMS);

	void* swp = (void*)xcalloc(1, size);
	for (size_t i = 0; i < n-1; ++i)
	{
		size_t j = i + random_gen()/(RAND_MAX / (n-i)+1);
		memcpy(swp,            (char*)arr+(j*size), size);
		memcpy((char*)arr+(j*size), (char*)arr+(i*size), size);
		memcpy((char*)arr+(i*size), swp,          size);
	}
	free(swp);
}

void del_arg(int argc, char **argv, int index)
{
	TAT(TATPARMS);

	int i;
	for(i = index; i < argc-1; ++i)
	{
		argv[i] = argv[i+1];
	}
	argv[i] = 0;
}

int find_arg(int argc, char* argv[], const char * const arg)
{
	TAT(TATPARMS);

	for (int i = 0; i < argc; ++i)
	{
		if (!argv[i])
		{
			continue;
		}

		if (0==strcmp(argv[i], arg))
		{
			del_arg(argc, argv, i);
			return 1;
		}
	}
	return 0;
}

int find_int_arg(int argc, char **argv, const char * const arg, int def)
{
	TAT(TATPARMS);

	for (int i = 0; i < argc-1; ++i)
	{
		if (!argv[i])
		{
			continue;
		}

		if (0==strcmp(argv[i], arg))
		{
			def = atoi(argv[i+1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

float find_float_arg(int argc, char **argv, const char * const arg, float def)
{
	TAT(TATPARMS);

	for (int i = 0; i < argc-1; ++i)
	{
		if (!argv[i])
		{
			continue;
		}

		if (0==strcmp(argv[i], arg))
		{
			def = atof(argv[i+1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

const char * find_char_arg(int argc, char **argv, const char *arg, const char *def)
{
	TAT(TATPARMS);

	for (int i = 0; i < argc-1; ++i)
	{
		if (!argv[i])
		{
			continue;
		}

		if (0==strcmp(argv[i], arg))
		{
			def = argv[i+1];
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}


const char *basecfg(const char * cfgfile)
{
	TAT(TATPARMS);

	/* This will return the base filename, no path and no extension.  For example:
	 *
	 * Input == /home/stephane/nn/LegoGears/LegoGears.cfg
	 * Output == LegoGears
	 *
	 * This is also used with images.  For example:
	 *
	 * Input == /home/stephane/nn/LegoGears/darkmark_image_cache/zoom/00000260.jpg
	 * Output == 00000260
	 */

	/// @todo 2025-04-18 replace all of this with @p std::filesystem::path::stem()?

	char * c = const_cast<char*>(cfgfile);
	char *next;
	while((next = strchr(c, '/')))
	{
		c = next+1;
	}

	if (!next)
	{
		while ((next = strchr(c, '\\')))
		{
			c = next + 1;
		}
	}

	c = copy_string(c);
	next = strchr(c, '.');

	if (next)
	{
		*next = 0;
	}

	return c;
}

int alphanum_to_int(char c)
{
	TAT(TATPARMS);

	return (c < 58) ? c - 48 : c-87;
}

char int_to_alphanum(int i)
{
	TAT(TATPARMS);

	if (i == 36) return '.';
	return (i < 10) ? i + 48 : i + 87;
}

void find_replace(const char* str, char* orig, char* rep, char* output)
{
	TAT(TATPARMS);

	char* buffer = (char*)calloc(8192, sizeof(char));
	char *p;

	sprintf(buffer, "%s", str);
	if (!(p = strstr(buffer, orig)))
	{
		// Is 'orig' even in 'str'?
		sprintf(output, "%s", buffer);
		free(buffer);
		return;
	}

	*p = '\0';

	sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
	free(buffer);
}

void replace_image_to_label(const char* input_path, char* output_path)
{
	TAT(TATPARMS);

	// keep it simple -- copy the input path, but change the extension to ".txt"

	strcpy(output_path, input_path);
	auto ptr = strrchr(output_path, '.');
	if (ptr)
	{
		ptr[0] = '\0';
	}
	strcat(output_path, ".txt");

	return;
}

float sec(clock_t clocks)
{
	TAT(TATPARMS);

	return (float)clocks/CLOCKS_PER_SEC;
}

void top_k(float *a, int n, int k, int *index)
{
	TAT(TATPARMS);

	for (int j = 0; j < k; ++j)
	{
		index[j] = -1;
	}

	for (int i = 0; i < n; ++i)
	{
		int curr = i;
		for (int j = 0; j < k; ++j)
		{
			if((index[j] < 0) || a[curr] > a[index[j]])
			{
				std::swap(curr, index[j]);
			}
		}
	}
}


void log_backtrace()
{
	TAT(TATPARMS);

	#define MAX_STACK_FRAMES 50
	void * stack[MAX_STACK_FRAMES];

#ifdef WIN32
	auto process = GetCurrentProcess();
	SymSetOptions(
		SYMOPT_CASE_INSENSITIVE 		|
		SYMOPT_FAIL_CRITICAL_ERRORS		|
		SYMOPT_INCLUDE_32BIT_MODULES	|
		SYMOPT_NO_PROMPTS				|
		SYMOPT_UNDNAME					);
	SymInitialize(process, nullptr, TRUE);

	auto count = CaptureStackBackTrace(0, MAX_STACK_FRAMES, stack, nullptr);

	*cfg_and_state.output << "backtrace (" << count << " entries):" << std::endl;

	char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
	SYMBOL_INFO * symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
	symbol->MaxNameLen = MAX_SYM_NAME;
	symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

	for (auto idx = 0; idx < count; idx ++)
	{
		SymFromAddr(process, DWORD64(stack[idx]), 0, symbol);
		*cfg_and_state.output << (idx + 1) << "/" << count << ": " << symbol->Name << "()" << std::endl;
	}
	SymCleanup(process);
	CloseHandle(process);
#else
	int count = backtrace(stack, MAX_STACK_FRAMES);
	char **symbols = backtrace_symbols(stack, count);

	*cfg_and_state.output << "backtrace (" << count << " entries):" << std::endl;

	for (int idx = 0; idx < count; idx ++)
	{
		*cfg_and_state.output << (idx + 1) << "/" << count << ": " << symbols[idx] << std::endl;
	}

	free(symbols);
#endif
}


namespace
{
	/* When things start going wrong, it isn't unusual for multiple threads to exit at the same time.  Multiple calls to
	 * darknet_fatal_error() results in overlapping messages, or we may not understand exactly which call was the first
	 * one to cause the original error.  To prevent this confusion, we use a mutex lock and will allow a single thread
	 * at a time to call into darknet_fatal_error().
	 */
	static std::timed_mutex darknet_fatal_error_mutex;
}


[[noreturn]] void darknet_fatal_error(const char * const filename, const char * const funcname, const int line, const char * const msg, ...)
{
//	TAT(TATPARMS); ... don't bother, we're already about to abort because something has gone wrong, don't make things worse

	const int saved_errno = errno;

	// make an attempt to lock, but proceed even if the lock failed (we're fatally exiting anyway!)
	const auto is_locked = darknet_fatal_error_mutex.try_lock_for(std::chrono::seconds(5));

	// only log the message and the rest of the information if this is the first call into darknet_fatal_error()
	auto & cfg_and_state = Darknet::CfgAndState::get();

	if (cfg_and_state.must_immediately_exit == false)
	{
		cfg_and_state.must_immediately_exit = true;

		char msg_buffer[1024];
		va_list args;
		va_start(args, msg);
		vsnprintf(msg_buffer, sizeof(msg_buffer), msg, args);
		va_end(args);

		decltype(cfg_and_state.thread_names) all_thread_names;
		if (true)
		{
			std::scoped_lock lock(cfg_and_state.thread_names_mutex);
			all_thread_names = cfg_and_state.thread_names;
		}

		const auto tid_to_digits = [](const std::thread::id & t) -> uint64_t
		{
			std::stringstream ss;
			ss << t;
			return std::stoull(ss.str());
		};

		*cfg_and_state.output << std::endl << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << std::endl;

		if (saved_errno != 0)
		{
			*cfg_and_state.output
				<< "* Errno " << saved_errno
#ifndef WIN32
				<< ": " << strerror(saved_errno)
#endif
				<< std::endl;

			/* We can no longer call perror() since we want to control where the output is saved.
			 *
			errno = saved_errno;
			perror(buffer);
			 */
		}

		*cfg_and_state.output
			<< "* Error location: " << filename << ", " << funcname << "(), line #" << line << std::endl
			<< "* Error message:  " << Darknet::in_colour(Darknet::EColour::kBrightRed) << msg_buffer << Darknet::in_colour(Darknet::EColour::kNormal) << std::endl
			<< "* Thread #" << tid_to_digits(std::this_thread::get_id()) << ": " << cfg_and_state.get_thread_name() << std::endl
			<< "* Version " << "\"" << DARKNET_VERSION_KEYWORD << "\" " << DARKNET_VERSION_STRING << " built on " << __DATE__ << " " << __TIME__ << std::endl
			<< "* * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << std::endl << std::endl;

		log_backtrace();

		*cfg_and_state.output << std::endl << "known threads:  " << all_thread_names.size() << std::endl;
		size_t count = 0;
		for (const auto & [tid, name] : all_thread_names)
		{
			count ++;

			*cfg_and_state.output << count << "/" << all_thread_names.size() << ": #" << tid_to_digits(tid) << ": " << name << std::endl;

			if (count > 15 and all_thread_names.size() >= 20)
			{
				*cfg_and_state.output << "..." << std::endl;
				break;
			}
		}
	}

	*cfg_and_state.output << std::endl << std::flush;

	if (is_locked)
	{
		darknet_fatal_error_mutex.unlock();
	}

	// Don't bother trying to exit() cleanly since some threads might be tied up in CUDA calls that
	// tend to hang when things go wrong.  Reset the signal handler to the default action and abort.
	std::signal(SIGABRT, SIG_DFL);
	std::abort();
}

const char * size_to_IEC_string(const size_t size)
{
	TAT(TATPARMS);

	const float bytes = static_cast<float>(size);
	const float KiB = 1024.0f;
	const float MiB = 1024.0f * KiB;
	const float GiB = 1024.0f * MiB;

	static char buffer[25]; /// @todo not thread safe

	if		(size < 0.75f * KiB)	sprintf(buffer, "%ld bytes", static_cast<long>(size));
	else if (size < 0.75f * MiB)	sprintf(buffer, "%1.1f KiB", bytes / KiB);
	else if (size < 0.75f * GiB)	sprintf(buffer, "%1.1f MiB", bytes / MiB);
	else							sprintf(buffer, "%1.1f GiB", bytes / GiB);

	return buffer;
}

void malloc_error(const size_t size, const char * const filename, const char * const funcname, const int line)
{
//	TAT(TATPARMS); ... don't bother, we're about to abort
	darknet_fatal_error(filename, funcname, line, "failed to malloc %s", size_to_IEC_string(size));
}

void calloc_error(const size_t size, const char * const filename, const char * const funcname, const int line)
{
//	TAT(TATPARMS); ... don't bother, we're about to abort
	darknet_fatal_error(filename, funcname, line, "failed to calloc %s", size_to_IEC_string(size));
}

void realloc_error(const size_t size, const char * const filename, const char * const funcname, const int line)
{
//	TAT(TATPARMS); ... don't bother, we're about to abort
	darknet_fatal_error(filename, funcname, line, "failed to realloc %s", size_to_IEC_string(size));
}

void file_error(const char * const s, const char * const filename, const char * const funcname, const int line)
{
//	TAT(TATPARMS); ... don't bother, we're about to abort

	if (s == nullptr or s[0] == '\0')
	{
		darknet_fatal_error(filename, funcname, line, "filename seems to be missing; cannot open the file \"<unknown>\"");
	}

	darknet_fatal_error(filename, funcname, line, "failed to open the file \"%s\"", s);
}

list *split_str(char *s, char delim)
{
	TAT(TATPARMS);

	size_t i;
	size_t len = strlen(s);
	list *l = make_list();
	list_insert(l, s);
	for(i = 0; i < len; ++i)
	{
		if(s[i] == delim)
		{
			s[i] = '\0';
			list_insert(l, &(s[i+1]));
		}
	}
	return l;
}

void strip(char *s)
{
	TAT(TATPARMS);

	if (s == nullptr)
	{
		return;
	}

	/* The old strip() function would remove *all* whitespace, including whitespace in the middle of the text.  This would
	 * cause problems when a filename or a subdirectory contains a space.  So this new function (2024-02) expects a KEY=VAL
	 * pair and rebuilds the string from what regex returns.
	 */

	static const std::regex rx(
		"^"			// start of string
		"\\s*"		// optional leading whitespace
		"("			// GROUP #1
		"[^=]*?"	// "KEY" is everything up to the first "=" (lazy match to ignore whitespace before the "=")
		")"
		"\\s*"		// optional whitespace
		"[:=]"		// ":" or "="
		"\\s*"		// optional whitespace
		"("			// GROUP #2
		".*?"		// "VAL" (lazy match so we don't grab the trailing whitespace)
		")"
		"\\s*"		// optional trailing whitespace
		"$"			// end of string
		);

	std::smatch match;
	std::string str = s;
	if (std::regex_match(str, match, rx))
	{
		std::string key = match.str(1);
		std::string val = match.str(2);
		str = key + "=" + val;

		strcpy(s, str.c_str());
		return;
	}

	// if we get here, then trim() is being used for something other than KEY=VAL, so we'll run the original code

	size_t len = strlen(s);
	size_t offset = 0;
	for (size_t i = 0; i < len; ++i)
	{
		char c = s[i];
		if (c == ' '	||
			c == '\t'	||
			c == '\n'	||
			c == '\r'	)
		{
			++offset;
		}
		else
		{
			s[i - offset] = c;
		}
	}
	s[len - offset] = '\0';
}


void strip_args(char *s)
{
	TAT(TATPARMS);

	size_t i;
	size_t len = strlen(s);
	size_t offset = 0;
	for (i = 0; i < len; ++i)
	{
		char c = s[i];
		if (c == '\t' || c == '\n' || c == '\r' || c == 0x0d || c == 0x0a)
		{
			++offset;
		}
		else
		{
			s[i - offset] = c;
		}
	}
	s[len - offset] = '\0';
}

void strip_char(char *s, char bad)
{
	TAT(TATPARMS);

	size_t i;
	size_t len = strlen(s);
	size_t offset = 0;
	for(i = 0; i < len; ++i)
	{
		char c = s[i];
		if(c==bad)
		{
			++offset;
		}
		else
		{
			s[i-offset] = c;
		}
	}
	s[len-offset] = '\0';
}

void free_ptrs(void **ptrs, int n)
{
	TAT(TATPARMS);

	for(int i = 0; i < n; ++i)
	{
		free(ptrs[i]);
	}

	free(ptrs);
}


char * fgetl(FILE *fp)
{
	TAT(TATPARMS);

	if (feof(fp))
	{
		return nullptr;
	}

	// get the first part of the line...
	size_t size = 512;
	char * line = (char*)xmalloc(size * sizeof(char));
	if (!fgets(line, size, fp))
	{
		free(line);
		return 0;
	}

	// ...and if the line does not end with \n we reallocate and keep reading more
	size_t curr = strlen(line);
	while ((line[curr - 1] != '\n') && !feof(fp))
	{
		if (curr == size - 1)
		{
			size *= 2;
			line = (char*)xrealloc(line, size * sizeof(char));
		}
		size_t readsize = size-curr;
		if (readsize > INT_MAX)
		{
			readsize = INT_MAX-1;
		}

		auto ptr = fgets(&line[curr], readsize, fp);
		curr = strlen(ptr);
	}

	// get rid of CR and LF at the end of the line
	if (curr >= 2)
	{
		if (line[curr-2] == '\r')
		{
			line[curr-2] = '\0';
		}
	}

	if (curr >= 1)
	{
		if (line[curr-1] == '\n')
		{
			line[curr-1] = '\0';
		}
	}

	return line;
}

int read_int(int fd)
{
	TAT(TATPARMS);

	int n = 0;
	int next = read(fd, &n, sizeof(int));
	if (next <= 0)
	{
		return -1;
	}
	return n;
}

void write_int(int fd, int n)
{
	TAT(TATPARMS);

	int next = write(fd, &n, sizeof(int));
	if (next <= 0)
	{
		darknet_fatal_error(DARKNET_LOC, "write failed");
	}
}

int read_all_fail(int fd, char *buffer, size_t bytes)
{
	TAT(TATPARMS);

	size_t n = 0;
	while(n < bytes)
	{
		int next = read(fd, buffer + n, bytes-n);
		if (next <= 0)
		{
			return 1;
		}
		n += next;
	}
	return 0;
}

int write_all_fail(int fd, char *buffer, size_t bytes)
{
	TAT(TATPARMS);

	size_t n = 0;
	while(n < bytes)
	{
		size_t next = write(fd, buffer + n, bytes-n);
		if (next <= 0)
		{
			return 1;
		}
		n += next;
	}
	return 0;
}

void read_all(int fd, char *buffer, size_t bytes)
{
	TAT(TATPARMS);

	size_t n = 0;
	while(n < bytes)
	{
		int next = read(fd, buffer + n, bytes-n);
		if (next <= 0)
		{
			darknet_fatal_error(DARKNET_LOC, "read failed");
		}
		n += next;
	}
}

void write_all(int fd, char *buffer, size_t bytes)
{
	TAT(TATPARMS);

	size_t n = 0;
	while(n < bytes)
	{
		size_t next = write(fd, buffer + n, bytes-n);
		if (next <= 0)
		{
			darknet_fatal_error(DARKNET_LOC, "write failed");
		}
		n += next;
	}
}


char *copy_string(char *s)
{
	TAT(TATPARMS);

	if (!s)
	{
		return NULL;
	}

	const size_t len = strlen(s) + 1;

	char* copy = (char*)xmalloc(len);

	memcpy(copy, s, len);

	return copy;
}

float sum_array(float *a, int n)
{
	TAT(TATPARMS);

	float sum = 0;
	for (int i = 0; i < n; ++i)
	{
		sum += a[i];
	}

	return sum;
}

float mean_array(float *a, int n)
{
	TAT(TATPARMS);

	return sum_array(a,n)/n;
}

void mean_arrays(float **a, int n, int els, float *avg)
{
	TAT(TATPARMS);

	memset(avg, 0, els*sizeof(float));
	for (int j = 0; j < n; ++j)
	{
		for (int i = 0; i < els; ++i)
		{
			avg[i] += a[j][i];
		}
	}
	for (int i = 0; i < els; ++i)
	{
		avg[i] /= n;
	}
}

void print_statistics(float *a, int n)
{
	TAT(TATPARMS);

	float m = mean_array(a, n);
	float v = variance_array(a, n);
	*cfg_and_state.output
		<< "MSE: "			<< mse_array(a, n)
		<< ", Mean: "		<< m
		<< ", Variance: "	<< v
		<< std::endl;
}

float variance_array(float *a, int n)
{
	TAT(TATPARMS);

	int i;
	float sum = 0;
	float mean = mean_array(a, n);
	for(i = 0; i < n; ++i)
	{
		sum += (a[i] - mean)*(a[i]-mean);
	}
	float variance = sum/n;
	return variance;
}

float constrain(float min, float max, float a)
{
	TAT(TATPARMS);

	return std::clamp(a, min, max);
}

float dist_array(float *a, float *b, int n, int sub)
{
	TAT(TATPARMS);

	int i;
	float sum = 0;
	for (i = 0; i < n; i += sub)
	{
		sum += pow(a[i]-b[i], 2);
	}
	return sqrt(sum);
}

float mse_array(float *a, int n)
{
	TAT(TATPARMS);

	int i;
	float sum = 0;
	for (i = 0; i < n; ++i)
	{
		sum += a[i]*a[i];
	}
	return sqrt(sum/n);
}

void normalize_array(float *a, int n)
{
	TAT(TATPARMS);

	float mu = mean_array(a,n);
	float sigma = sqrt(variance_array(a,n));
	for(int i = 0; i < n; ++i)
	{
		a[i] = (a[i] - mu)/sigma;
	}
	//mu = mean_array(a,n);
	//sigma = sqrt(variance_array(a,n));
}

void translate_array(float *a, int n, float s)
{
	TAT(TATPARMS);

	for(int i = 0; i < n; ++i)
	{
		a[i] += s;
	}
}

float mag_array(float *a, int n)
{
	TAT(TATPARMS);

	float sum = 0;
	for(int i = 0; i < n; ++i)
	{
		sum += a[i]*a[i];
	}
	return sqrt(sum);
}

void scale_array(float *a, int n, float s)
{
	TAT(TATPARMS);

	for(int i = 0; i < n; ++i)
	{
		a[i] *= s;
	}
}

int max_index(float *a, int n)
{
	TAT(TATPARMS);

	if (n <= 0)
	{
		return -1;
	}

	int max_i = 0;
	float max = a[0];
	for(int i = 1; i < n; ++i)
	{
		if (a[i] > max)
		{
			max = a[i];
			max_i = i;
		}
	}
	return max_i;
}

int top_max_index(float *a, int n, int k)
{
	TAT(TATPARMS);

	if (n <= 0)
	{
		return -1;
	}

	float *values = (float*)xcalloc(k, sizeof(float));
	int *indexes = (int*)xcalloc(k, sizeof(int));
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			if (a[i] > values[j])
			{
				values[j] = a[i];
				indexes[j] = i;
				break;
			}
		}
	}

	int count = 0;
	for (int j = 0; j < k; ++j)
	{
		if (values[j] > 0)
		{
			count++;
		}
	}

	int get_index = rand_int(0, count-1);
	int val = indexes[get_index];
	free(indexes);
	free(values);
	return val;
}


int int_index(int *a, int val, int n)
{
	TAT(TATPARMS);

	for (int i = 0; i < n; ++i)
	{
		if (a[i] == val)
		{
			return i;
		}
	}

	return -1;
}

int rand_int(int min, int max)
{
	TAT(TATPARMS);

	if (max < min)
	{
		std::swap(min, max);
	}

	int r = random_gen(min, max);

	return r;
}

// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float rand_normal()
{
	TAT(TATPARMS);

	static int haveSpare = 0;
	static double rand1, rand2;

	if (haveSpare)
	{
		haveSpare = 0;
		return sqrt(rand1) * sin(rand2);
	}

	haveSpare = 1;

	rand1 = random_gen() / ((double) RAND_MAX);
	if (rand1 < 1e-100)
	{
		rand1 = 1e-100;
	}
	rand1 = -2 * log(rand1);
	rand2 = (random_gen() / ((double)RAND_MAX)) * 2.0 * M_PI;

	return sqrt(rand1) * cos(rand2);
}

#if 1
static inline float rand_uniform_fast(float min, float max)
{
	// スレッドローカルな軽量乱数ジェネレータ（xorshift128+）
	thread_local static struct {
		// 静的カウンターを使用（アドレスの代わり）
		static uint64_t get_seed() {
			static uint64_t counter = 0;
			return ++counter;
		}

		// 初期化済みフラグと状態配列
		bool initialized = false;
		uint64_t state[2];

		// 初期化関数
		void init() {
			if (!initialized) {
				// シードの生成（アドレス値を使わない安全な方法）
				uint64_t seed1 = static_cast<uint64_t>(time(nullptr));
				uint64_t seed2 = get_seed();

				// シードを設定
				state[0] = seed1 ^ 0x5555555555555555ULL;
				state[1] = seed2 ^ 0xAAAAAAAAAAAAAAAAULL;

				// 初期状態を少し混ぜる
				for (int i = 0; i < 10; i++) {
					next();
				}

				initialized = true;
			}
		}

		// 超高速なxorshift128+実装
		inline uint64_t next() {
			uint64_t s0 = state[0];
			uint64_t s1 = state[1];
			state[0] = s1;
			s1 ^= s1 << 23;
			state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
			return state[1] + s0;
		}

		// [0,1)の範囲に正規化
		inline float nextFloat() {
			// 初回呼び出し時に初期化
			if (!initialized) {
				init();
			}

			// 浮動小数点変換方法1：シンプルな方法
			return static_cast<float>(static_cast<double>(next()) /
				static_cast<double>(UINT64_MAX));

			/* 浮動小数点変換方法2：ビット操作（コンパイラによっては問題が出るため使用を検討）
			const uint32_t x = static_cast<uint32_t>(next());
			return static_cast<float>(x) / static_cast<float>(UINT32_MAX);
			*/
		}
	} rng;

	// [min,max]の範囲にスケーリング
	return min + rng.nextFloat() * (max - min);
}

// 元の関数シグネチャとの互換性維持
float rand_uniform(float min, float max)
{
	TAT(TATPARMS);
	// min/maxスワップを維持
	if (max < min) {
		float temp = min;
		min = max;
		max = temp;
	}

	return rand_uniform_fast(min, max);
}
#else
float rand_uniform(float min, float max)
{
	TAT(TATPARMS);

	/// @todo Re-write this using std::uniform_real_distribution

	if (max < min)
	{
		std::swap(min, max);
	}

#if (RAND_MAX < 65536)
		int rnd = rand()*(RAND_MAX + 1) + rand();
		return ((float)rnd / (RAND_MAX*RAND_MAX) * (max - min)) + min;
#else
		return ((float)rand() / (float)RAND_MAX * (max - min)) + min;
#endif
	//return (random_float() * (max - min)) + min;
}
#endif // 0

float rand_scale(float s)
{
	TAT(TATPARMS);

	float scale = rand_uniform_strong(1, s);
	if (random_gen()%2)
	{
		return scale;
	}
	return 1.0f/scale;
}

namespace
{
	inline std::mt19937 & get_rnd_engine()
	{
		TAT(TATPARMS);

		// we must have 1 per thread of these (use random_device to seed the engine)
		static thread_local std::mt19937 rnd_engine(std::random_device{}());

		return rnd_engine;
	}
}

unsigned int random_gen(unsigned int min, unsigned int max)
{
	TAT(TATPARMS);

	// This is inclusive.  It is possible to get back "min", "max", and every integer value in between.
	std::uniform_int_distribution<unsigned int> distribution(min, max);

	return distribution(get_rnd_engine());
}


float random_float()
{
	TAT(TATPARMS);

	/// @todo Re-write this using std::uniform_real_distribution

	unsigned int rnd = 0;
#ifdef WIN32
	rand_s(&rnd);
	return ((float)rnd / (float)UINT_MAX);
#else   // WIN32

	rnd = rand();
#if (RAND_MAX < 65536)
	rnd = rand()*(RAND_MAX + 1) + rnd;
	return((float)rnd / (float)(RAND_MAX*RAND_MAX));
#endif  //(RAND_MAX < 65536)
	return ((float)rnd / (float)RAND_MAX);

#endif  // WIN32
}

float rand_uniform_strong(float min, float max)
{
	TAT(TATPARMS);

	/// @todo Re-write this using std::uniform_real_distribution

	if (max < min)
	{
		std::swap(min, max);
	}
	return (random_float() * (max - min)) + min;
}

float rand_precalc_random(float min, float max, float random_part)
{
	TAT(TATPARMS);

	if (max < min)
	{
		std::swap(min, max);
	}
	return (random_part * (max - min)) + min;
}

int make_directory(char *path, int mode)
{
	TAT(TATPARMS);

#ifdef WIN32
	return _mkdir(path);
#else
	return mkdir(path, mode);
#endif
}

unsigned long custom_hash(char *str)
{
	TAT(TATPARMS);

	unsigned long hash = 5381;
	int c;
	while ((c = (*str++)))
	{
		hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
	}

	return hash;
}
