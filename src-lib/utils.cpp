#define _CRT_RAND_S

#include <csignal>
#include <regex>
#include "darknet_internal.hpp"
#include "darkunistd.hpp"

#ifdef WIN32
#include "gettimeofday.h"
#include <dbghelp.h>
#pragma comment(lib, "DbgHelp.lib")
#else
#include <sys/time.h>
#include <sys/stat.h>
#include <execinfo.h>
#endif


void *xmalloc_location(const size_t size, const char * const filename, const char * const funcname, const int line)
{
	TAT(TATPARMS);

	void *ptr=malloc(size);
	if(!ptr) {
		malloc_error(size, filename, funcname, line);
	}
	return ptr;
}

void *xcalloc_location(const size_t nmemb, const size_t size, const char * const filename, const char * const funcname, const int line)
{
	TAT(TATPARMS);

	void *ptr=calloc(nmemb, size);
	if(!ptr) {
		calloc_error(nmemb * size, filename, funcname, line);
	}
	return ptr;
}

void *xrealloc_location(void *ptr, const size_t size, const char * const filename, const char * const funcname, const int line)
{
	TAT(TATPARMS);

	ptr=realloc(ptr,size);
	if(!ptr) {
		realloc_error(size, filename, funcname, line);
	}
	return ptr;
}

double what_time_is_it_now()
{
	TAT(TATPARMS);

	struct timeval time;
	if (gettimeofday(&time, NULL)) {
		return 0;
	}

	/* tv_usec is in microseconds, so 1/1000000 seconds.
	* We're converting the time into a double.
	*/

	return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}

int *read_map(char *filename)
{
	TAT(TATPARMS);

	int n = 0;
	int *map = 0;
	char *str;
	FILE *file = fopen(filename, "r");
	if(!file) file_error(filename, DARKNET_LOC);
	while((str=fgetl(file))){
		++n;
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

	size_t i;
	for(i = 0; i < sections; ++i){
		size_t start = n*i/sections;
		size_t end = n*(i+1)/sections;
		size_t num = end-start;
		shuffle((char*)arr+(start*size), num, size);
	}
}

void shuffle(void *arr, size_t n, size_t size)
{
	TAT(TATPARMS);

	size_t i;
	void* swp = (void*)xcalloc(1, size);
	for(i = 0; i < n-1; ++i){
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
	for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
	argv[i] = 0;
}

int find_arg(int argc, char* argv[], const char * const arg)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < argc; ++i) {
		if(!argv[i]) continue;
		if(0==strcmp(argv[i], arg)) {
			del_arg(argc, argv, i);
			return 1;
		}
	}
	return 0;
}

int find_int_arg(int argc, char **argv, const char * const arg, int def)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < argc-1; ++i){
		if(!argv[i]) continue;
		if(0==strcmp(argv[i], arg)){
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

	int i;
	for(i = 0; i < argc-1; ++i){
		if(!argv[i]) continue;
		if(0==strcmp(argv[i], arg)){
			def = atof(argv[i+1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < argc-1; ++i){
		if(!argv[i]) continue;
		if(0==strcmp(argv[i], arg)){
			def = argv[i+1];
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}


char *basecfg(char *cfgfile)
{
	TAT(TATPARMS);

	char *c = cfgfile;
	char *next;
	while((next = strchr(c, '/')))
	{
		c = next+1;
	}
	if(!next) while ((next = strchr(c, '\\'))) { c = next + 1; }
	c = copy_string(c);
	next = strchr(c, '.');
	if (next) *next = 0;
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

void pm(int M, int N, float *A)
{
	TAT(TATPARMS);

	int i,j;
	for(i =0 ; i < M; ++i){
		printf("%d ", i+1);
		for(j = 0; j < N; ++j){
			printf("%2.4f, ", A[i*N+j]);
		}
		printf("\n");
	}
	printf("\n");
}

void find_replace(const char* str, char* orig, char* rep, char* output)
{
	TAT(TATPARMS);

	char* buffer = (char*)calloc(8192, sizeof(char));
	char *p;

	sprintf(buffer, "%s", str);
	if (!(p = strstr(buffer, orig))) {  // Is 'orig' even in 'str'?
		sprintf(output, "%s", buffer);
		free(buffer);
		return;
	}

	*p = '\0';

	sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
	free(buffer);
}

void trim(char *str)
{
	TAT(TATPARMS);

	char* buffer = (char*)xcalloc(8192, sizeof(char));
	sprintf(buffer, "%s", str);

	char *p = buffer;
	while (*p == ' ' || *p == '\t') ++p;

	char *end = p + strlen(p) - 1;
	while (*end == ' ' || *end == '\t') {
		*end = '\0';
		--end;
	}
	sprintf(str, "%s", p);

	free(buffer);
}

void find_replace_extension(char *str, char *orig, char *rep, char *output)
{
	TAT(TATPARMS);

	char* buffer = (char*)calloc(8192, sizeof(char));

	sprintf(buffer, "%s", str);
	char *p = strstr(buffer, orig);
	int offset = (p - buffer);
	int chars_from_end = strlen(buffer) - offset;
	if (!p || chars_from_end != strlen(orig)) {  // Is 'orig' even in 'str' AND is 'orig' found at the end of 'str'?
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

	find_replace(input_path, "/images/train2017/", "/labels/train2017/", output_path);    // COCO
	find_replace(output_path, "/images/val2017/", "/labels/val2017/", output_path);        // COCO
	find_replace(output_path, "/JPEGImages/", "/labels/", output_path);    // PascalVOC
	find_replace(output_path, "\\images\\train2017\\", "\\labels\\train2017\\", output_path);    // COCO
	find_replace(output_path, "\\images\\val2017\\", "\\labels\\val2017\\", output_path);        // COCO

	find_replace(output_path, "\\images\\train2014\\", "\\labels\\train2014\\", output_path);    // COCO
	find_replace(output_path, "\\images\\val2014\\", "\\labels\\val2014\\", output_path);        // COCO
	find_replace(output_path, "/images/train2014/", "/labels/train2014/", output_path);    // COCO
	find_replace(output_path, "/images/val2014/", "/labels/val2014/", output_path);        // COCO

	find_replace(output_path, "\\JPEGImages\\", "\\labels\\", output_path);    // PascalVOC
	//find_replace(output_path, "/images/", "/labels/", output_path);    // COCO
	//find_replace(output_path, "/VOC2007/JPEGImages/", "/VOC2007/labels/", output_path);        // PascalVOC
	//find_replace(output_path, "/VOC2012/JPEGImages/", "/VOC2012/labels/", output_path);        // PascalVOC

	//find_replace(output_path, "/raw/", "/labels/", output_path);
	trim(output_path);

	// replace only ext of files
	find_replace_extension(output_path, ".jpg", ".txt", output_path);
	find_replace_extension(output_path, ".JPG", ".txt", output_path); // error
	find_replace_extension(output_path, ".jpeg", ".txt", output_path);
	find_replace_extension(output_path, ".JPEG", ".txt", output_path);
	find_replace_extension(output_path, ".png", ".txt", output_path);
	find_replace_extension(output_path, ".PNG", ".txt", output_path);
	find_replace_extension(output_path, ".bmp", ".txt", output_path);
	find_replace_extension(output_path, ".BMP", ".txt", output_path);
	find_replace_extension(output_path, ".ppm", ".txt", output_path);
	find_replace_extension(output_path, ".PPM", ".txt", output_path);
	find_replace_extension(output_path, ".tiff", ".txt", output_path);
	find_replace_extension(output_path, ".TIFF", ".txt", output_path);

	// Check file ends with txt:
	if(strlen(output_path) > 4) {
		char *output_path_ext = output_path + strlen(output_path) - 4;
		if( strcmp(".txt", output_path_ext) != 0){
			fprintf(stderr, "Failed to infer label file name (check image extension is supported): %s \n", output_path);
		}
	}else{
		fprintf(stderr, "Label file name is too short: %s \n", output_path);
	}
}

float sec(clock_t clocks)
{
	TAT(TATPARMS);

	return (float)clocks/CLOCKS_PER_SEC;
}

void top_k(float *a, int n, int k, int *index)
{
	TAT(TATPARMS);

	int i,j;
	for(j = 0; j < k; ++j) index[j] = -1;
	for(i = 0; i < n; ++i){
		int curr = i;
		for(j = 0; j < k; ++j){
			if((index[j] < 0) || a[curr] > a[index[j]]){
				int swap = curr;
				curr = index[j];
				index[j] = swap;
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

	fprintf(stderr, "backtrace (%d entries):\n", count);

	char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
	SYMBOL_INFO * symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
	symbol->MaxNameLen = MAX_SYM_NAME;
	symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

	for (auto idx = 0; idx < count; idx ++)
	{
		SymFromAddr(process, DWORD64(stack[idx]), 0, symbol);
		fprintf(stderr, "%d/%d: %s()\n", idx + 1, count, symbol->Name);
	}
	SymCleanup(process);
	CloseHandle(process);
#else
	int count = backtrace(stack, MAX_STACK_FRAMES);
	char **symbols = backtrace_symbols(stack, count);

	fprintf(stderr, "backtrace (%d entries):\n", count);

	for (int idx = 0; idx < count; idx ++)
	{
		fprintf(stderr, "%d/%d: %s\n", idx + 1, count, symbols[idx]);
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


void darknet_fatal_error(const char * const filename, const char * const funcname, const int line, const char * const msg, ...)
{
	const int saved_errno = errno;

	// make an attempt to lock, but proceed even if the lock failed (we're fatally exiting anyway!)
	const auto is_locked = darknet_fatal_error_mutex.try_lock_for(std::chrono::seconds(5));

	// only log the message and the rest of the information if this is the first call into darknet_fatal_error()
	if (Darknet::CfgAndState::get().must_immediately_exit == false)
	{
		Darknet::CfgAndState::get().must_immediately_exit = true;

		fprintf(stderr, "\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");

		if (saved_errno != 0)
		{
			char buffer[50];
			sprintf(buffer, "* Errno %d", saved_errno);
			errno = saved_errno;
			perror(buffer);
		}

		fprintf(stderr, "* Error location: %s, %s(), line #%d\n", filename, funcname, line);
		fprintf(stderr, "* Error message:  %s", Darknet::in_colour(Darknet::EColour::kBrightRed).c_str());
		va_list args;
		va_start(args, msg);
		vfprintf(stderr, msg, args);
		va_end(args);

		// the vfprintf() message is not newline-terminated so we need to take care of that before we print anything else
		fprintf(stderr,
			"%s\n"
			"* Version %s built on %s %s\n"
			"* * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n",
			Darknet::in_colour(Darknet::EColour::kNormal).c_str(),
			DARKNET_VERSION_STRING, __DATE__, __TIME__);

		log_backtrace();
	}

	std::fflush(stdout);
	std::fflush(stderr);
	std::cout << std::flush;
	std::cerr << std::flush;

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

	const float bytes = (double)size;
	const float KiB = 1024;
	const float MiB = 1024 * KiB;
	const float GiB = 1024 * MiB;

	static char buffer[25]; /// @todo not thread safe

	if (size < KiB)         sprintf(buffer, "%ld bytes", size);
	else if (size < MiB)    sprintf(buffer, "%1.1f KiB", bytes / KiB);
	else if (size < GiB)    sprintf(buffer, "%1.1f MiB", bytes / MiB);
	else                    sprintf(buffer, "%1.1f GiB", bytes / GiB);

	return buffer;
}

void malloc_error(const size_t size, const char * const filename, const char * const funcname, const int line)
{
	darknet_fatal_error(filename, funcname, line, "failed to malloc %s", size_to_IEC_string(size));
}

void calloc_error(const size_t size, const char * const filename, const char * const funcname, const int line)
{
	darknet_fatal_error(filename, funcname, line, "failed to calloc %s", size_to_IEC_string(size));
}

void realloc_error(const size_t size, const char * const filename, const char * const funcname, const int line)
{
	darknet_fatal_error(filename, funcname, line, "failed to realloc %s", size_to_IEC_string(size));
}

void file_error(const char * const s, const char * const filename, const char * const funcname, const int line)
{
	darknet_fatal_error(filename, funcname, line, "failed to open the file \"%s\"", s);
}

list *split_str(char *s, char delim)
{
	TAT(TATPARMS);

	size_t i;
	size_t len = strlen(s);
	list *l = make_list();
	list_insert(l, s);
	for(i = 0; i < len; ++i){
		if(s[i] == delim){
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
	for (i = 0; i < len; ++i) {
		char c = s[i];
		if (c == '\t' || c == '\n' || c == '\r' || c == 0x0d || c == 0x0a) ++offset;
		else s[i - offset] = c;
	}
	s[len - offset] = '\0';
}

void strip_char(char *s, char bad)
{
	TAT(TATPARMS);

	size_t i;
	size_t len = strlen(s);
	size_t offset = 0;
	for(i = 0; i < len; ++i){
		char c = s[i];
		if(c==bad) ++offset;
		else s[i-offset] = c;
	}
	s[len-offset] = '\0';
}

void free_ptrs(void **ptrs, int n)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < n; ++i) free(ptrs[i]);
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
		fgets(&line[curr], readsize, fp);
		curr = strlen(line);
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
	if(next <= 0) return -1;
	return n;
}

void write_int(int fd, int n)
{
	TAT(TATPARMS);

	int next = write(fd, &n, sizeof(int));
	if(next <= 0)
	{
		darknet_fatal_error(DARKNET_LOC, "write failed");
	}
}

int read_all_fail(int fd, char *buffer, size_t bytes)
{
	TAT(TATPARMS);

	size_t n = 0;
	while(n < bytes){
		int next = read(fd, buffer + n, bytes-n);
		if(next <= 0) return 1;
		n += next;
	}
	return 0;
}

int write_all_fail(int fd, char *buffer, size_t bytes)
{
	TAT(TATPARMS);

	size_t n = 0;
	while(n < bytes){
		size_t next = write(fd, buffer + n, bytes-n);
		if(next <= 0) return 1;
		n += next;
	}
	return 0;
}

void read_all(int fd, char *buffer, size_t bytes)
{
	TAT(TATPARMS);

	size_t n = 0;
	while(n < bytes){
		int next = read(fd, buffer + n, bytes-n);
		if(next <= 0)
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
	while(n < bytes){
		size_t next = write(fd, buffer + n, bytes-n);
		if(next <= 0)
		{
			darknet_fatal_error(DARKNET_LOC, "write failed");
		}
		n += next;
	}
}


char *copy_string(char *s)
{
	TAT(TATPARMS);

	if(!s)
	{
		return NULL;
	}

	char* copy = (char*)xmalloc(strlen(s) + 1);

/// @todo 2023-06-26 For now I'd rather ignore this warning than to try and mess with this old code and risk breaking things.
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

	strncpy(copy, s, strlen(s)+1);

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

	return copy;
}

list *parse_csv_line(char *line)
{
	TAT(TATPARMS);

	list *l = make_list();
	char *c, *p;
	int in = 0;
	for(c = line, p = line; *c != '\0'; ++c){
		if(*c == '"') in = !in;
		else if(*c == ',' && !in){
			*c = '\0';
			list_insert(l, copy_string(p));
			p = c+1;
		}
	}
	list_insert(l, copy_string(p));
	return l;
}

int count_fields(char *line)
{
	TAT(TATPARMS);

	int count = 0;
	int done = 0;
	char *c;
	for(c = line; !done; ++c){
		done = (*c == '\0');
		if(*c == ',' || done) ++count;
	}
	return count;
}

float *parse_fields(char *line, int n)
{
	TAT(TATPARMS);

	float* field = (float*)xcalloc(n, sizeof(float));
	char *c, *p, *end;
	int count = 0;
	int done = 0;
	for(c = line, p = line; !done; ++c){
		done = (*c == '\0');
		if(*c == ',' || done){
			*c = '\0';
			field[count] = strtod(p, &end);
			if(p == c) field[count] = nan("");
			if(end != c && (end != c-1 || *end != '\r')) field[count] = nan(""); //DOS file formats!
			p = c+1;
			++count;
		}
	}
	return field;
}

float sum_array(float *a, int n)
{
	TAT(TATPARMS);

	int i;
	float sum = 0;
	for(i = 0; i < n; ++i) sum += a[i];
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

	int i;
	int j;
	memset(avg, 0, els*sizeof(float));
	for(j = 0; j < n; ++j){
		for(i = 0; i < els; ++i){
			avg[i] += a[j][i];
		}
	}
	for(i = 0; i < els; ++i){
		avg[i] /= n;
	}
}

void print_statistics(float *a, int n)
{
	TAT(TATPARMS);

	float m = mean_array(a, n);
	float v = variance_array(a, n);
	printf("MSE: %.6f, Mean: %.6f, Variance: %.6f\n", mse_array(a, n), m, v);
}

float variance_array(float *a, int n)
{
	TAT(TATPARMS);

	int i;
	float sum = 0;
	float mean = mean_array(a, n);
	for(i = 0; i < n; ++i) sum += (a[i] - mean)*(a[i]-mean);
	float variance = sum/n;
	return variance;
}

int constrain_int(int a, int min, int max)
{
	TAT(TATPARMS);

	if (a < min) return min;
	if (a > max) return max;
	return a;
}

float constrain(float min, float max, float a)
{
	TAT(TATPARMS);

	if (a < min) return min;
	if (a > max) return max;
	return a;
}

float dist_array(float *a, float *b, int n, int sub)
{
	TAT(TATPARMS);

	int i;
	float sum = 0;
	for(i = 0; i < n; i += sub) sum += pow(a[i]-b[i], 2);
	return sqrt(sum);
}

float mse_array(float *a, int n)
{
	TAT(TATPARMS);

	int i;
	float sum = 0;
	for(i = 0; i < n; ++i) sum += a[i]*a[i];
	return sqrt(sum/n);
}

void normalize_array(float *a, int n)
{
	TAT(TATPARMS);

	int i;
	float mu = mean_array(a,n);
	float sigma = sqrt(variance_array(a,n));
	for(i = 0; i < n; ++i){
		a[i] = (a[i] - mu)/sigma;
	}
	//mu = mean_array(a,n);
	//sigma = sqrt(variance_array(a,n));
}

void translate_array(float *a, int n, float s)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < n; ++i){
		a[i] += s;
	}
}

float mag_array(float *a, int n)
{
	TAT(TATPARMS);

	int i;
	float sum = 0;
	for(i = 0; i < n; ++i){
		sum += a[i]*a[i];
	}
	return sqrt(sum);
}

// indicies to skip is a bit array
float mag_array_skip(float *a, int n, int * indices_to_skip)
{
	TAT(TATPARMS);

	int i;
	float sum = 0;
	for (i = 0; i < n; ++i) {
		if (indices_to_skip[i] != 1) {
			sum += a[i] * a[i];
		}
	}
	return sqrt(sum);
}

void scale_array(float *a, int n, float s)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < n; ++i){
		a[i] *= s;
	}
}

int sample_array(float *a, int n)
{
	TAT(TATPARMS);

	float sum = sum_array(a, n);
	scale_array(a, n, 1. / sum);
	float r = rand_uniform(0, 1);
	int i;
	for (i = 0; i < n; ++i) {
		r = r - a[i];
		if (r <= 0) return i;
	}
	return n - 1;
}

int sample_array_custom(float *a, int n)
{
	TAT(TATPARMS);

	float sum = sum_array(a, n);
	scale_array(a, n, 1./sum);
	float r = rand_uniform(0, 1);
	int start_index = rand_int(0, 0);
	int i;
	for(i = 0; i < n; ++i){
		r = r - a[(i + start_index) % n];
		if (r <= 0) return i;
	}
	return n-1;
}

int max_index(float *a, int n)
{
	TAT(TATPARMS);

	if(n <= 0) return -1;
	int i, max_i = 0;
	float max = a[0];
	for(i = 1; i < n; ++i){
		if(a[i] > max){
			max = a[i];
			max_i = i;
		}
	}
	return max_i;
}

int top_max_index(float *a, int n, int k)
{
	TAT(TATPARMS);

	if (n <= 0) return -1;
	float *values = (float*)xcalloc(k, sizeof(float));
	int *indexes = (int*)xcalloc(k, sizeof(int));
	int i, j;
	for (i = 0; i < n; ++i) {
		for (j = 0; j < k; ++j) {
			if (a[i] > values[j]) {
				values[j] = a[i];
				indexes[j] = i;
				break;
			}
		}
	}
	int count = 0;
	for (j = 0; j < k; ++j) if (values[j] > 0) count++;
	int get_index = rand_int(0, count-1);
	int val = indexes[get_index];
	free(indexes);
	free(values);
	return val;
}


int int_index(int *a, int val, int n)
{
	TAT(TATPARMS);

	int i;
	for (i = 0; i < n; ++i) {
		if (a[i] == val) return i;
	}
	return -1;
}

int rand_int(int min, int max)
{
	TAT(TATPARMS);

	if (max < min){
		int s = min;
		min = max;
		max = s;
	}
	int r = (random_gen()%(max - min + 1)) + min;
	return r;
}

// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float rand_normal()
{
	TAT(TATPARMS);

	static int haveSpare = 0;
	static double rand1, rand2;

	if(haveSpare)
	{
		haveSpare = 0;
		return sqrt(rand1) * sin(rand2);
	}

	haveSpare = 1;

	rand1 = random_gen() / ((double) RAND_MAX);
	if(rand1 < 1e-100) rand1 = 1e-100;
	rand1 = -2 * log(rand1);
	rand2 = (random_gen() / ((double)RAND_MAX)) * 2.0 * M_PI;

	return sqrt(rand1) * cos(rand2);
}

/*
float rand_normal()
{
int n = 12;
int i;
float sum= 0;
for(i = 0; i < n; ++i) sum += (float)random_gen()/RAND_MAX;
return sum-n/2.;
}
*/

size_t rand_size_t()
{
	TAT(TATPARMS);

	return  ((size_t)(random_gen()&0xff) << 56) |
			((size_t)(random_gen()&0xff) << 48) |
			((size_t)(random_gen()&0xff) << 40) |
			((size_t)(random_gen()&0xff) << 32) |
			((size_t)(random_gen()&0xff) << 24) |
			((size_t)(random_gen()&0xff) << 16) |
			((size_t)(random_gen()&0xff) << 8) |
			((size_t)(random_gen()&0xff) << 0);
}

float rand_uniform(float min, float max)
{
	TAT(TATPARMS);

	if(max < min){
		float swap = min;
		min = max;
		max = swap;
	}

#if (RAND_MAX < 65536)
		int rnd = rand()*(RAND_MAX + 1) + rand();
		return ((float)rnd / (RAND_MAX*RAND_MAX) * (max - min)) + min;
#else
		return ((float)rand() / RAND_MAX * (max - min)) + min;
#endif
	//return (random_float() * (max - min)) + min;
}

float rand_scale(float s)
{
	TAT(TATPARMS);

	float scale = rand_uniform_strong(1, s);
	if(random_gen()%2) return scale;
	return 1./scale;
}

float **one_hot_encode(float *a, int n, int k)
{
	TAT(TATPARMS);

	int i;
	float** t = (float**)xcalloc(n, sizeof(float*));
	for(i = 0; i < n; ++i){
		t[i] = (float*)xcalloc(k, sizeof(float));
		int index = (int)a[i];
		t[i][index] = 1;
	}
	return t;
}

static unsigned int x = 123456789, y = 362436069, z = 521288629;

// Marsaglia's xorshf96 generator: period 2^96-1
unsigned int random_gen_fast(void)
{
	TAT(TATPARMS);

	unsigned int t;
	x ^= x << 16;
	x ^= x >> 5;
	x ^= x << 1;

	t = x;
	x = y;
	y = z;
	z = t ^ x ^ y;

	return z;
}

float random_float_fast()
{
	TAT(TATPARMS);

	return ((float)random_gen_fast() / (float)UINT_MAX);
}

int rand_int_fast(int min, int max)
{
	TAT(TATPARMS);

	if (max < min) {
		int s = min;
		min = max;
		max = s;
	}
	int r = (random_gen_fast() % (max - min + 1)) + min;
	return r;
}

unsigned int random_gen()
{
	TAT(TATPARMS);

	unsigned int rnd = 0;
#ifdef WIN32
	rand_s(&rnd);
#else   // WIN32
	rnd = rand();
#if (RAND_MAX < 65536)
		rnd = rand()*(RAND_MAX + 1) + rnd;
#endif  //(RAND_MAX < 65536)
#endif  // WIN32
	return rnd;
}

float random_float()
{
	TAT(TATPARMS);

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

	if (max < min) {
		float swap = min;
		min = max;
		max = swap;
	}
	return (random_float() * (max - min)) + min;
}

float rand_precalc_random(float min, float max, float random_part)
{
	TAT(TATPARMS);

	if (max < min) {
		float swap = min;
		min = max;
		max = swap;
	}
	return (random_part * (max - min)) + min;
}

#define RS_SCALE (1.0 / (1.0 + RAND_MAX))

double double_rand(void)
{
	TAT(TATPARMS);

	double d;
	do {
		d = (((rand() * RS_SCALE) + rand()) * RS_SCALE + rand()) * RS_SCALE;
	} while (d >= 1); // Round off
	return d;
}

unsigned int uint_rand(unsigned int less_than)
{
	TAT(TATPARMS);

	return (unsigned int)((less_than)* double_rand());
}

int check_array_is_nan(float *arr, int size)
{
	TAT(TATPARMS);

	int i;
	for (i = 0; i < size; ++i) {
		if (isnan(arr[i])) return 1;
	}
	return 0;
}

int check_array_is_inf(float *arr, int size)
{
	TAT(TATPARMS);

	int i;
	for (i = 0; i < size; ++i) {
		if (isinf(arr[i])) return 1;
	}
	return 0;
}

int *random_index_order(int min, int max)
{
	TAT(TATPARMS);

	int *inds = (int *)xcalloc(max - min, sizeof(int));
	int i;
	for (i = min; i < max; ++i) {
		inds[i - min] = i;
	}
	for (i = min; i < max - 1; ++i) {
		int swap = inds[i - min];
		int index = i + rand() % (max - i);
		inds[i - min] = inds[index - min];
		inds[index - min] = swap;
	}
	return inds;
}

int max_int_index(int *a, int n)
{
	TAT(TATPARMS);

	if (n <= 0) return -1;
	int i, max_i = 0;
	int max = a[0];
	for (i = 1; i < n; ++i) {
		if (a[i] > max) {
			max = a[i];
			max_i = i;
		}
	}
	return max_i;
}


// Absolute box from relative coordinate bounding box and image size
boxabs box_to_boxabs(const box* b, const int img_w, const int img_h, const int bounds_check)
{
	TAT(TATPARMS);

	boxabs ba;
	ba.left = (b->x - b->w / 2.)*img_w;
	ba.right = (b->x + b->w / 2.)*img_w;
	ba.top = (b->y - b->h / 2.)*img_h;
	ba.bot = (b->y + b->h / 2.)*img_h;

	if (bounds_check) {
		if (ba.left < 0) ba.left = 0;
		if (ba.right > img_w - 1) ba.right = img_w - 1;
		if (ba.top < 0) ba.top = 0;
		if (ba.bot > img_h - 1) ba.bot = img_h - 1;
	}

	return ba;
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

/// @todo 2023-06-26 For now I'd rather ignore this warning than to try and mess with this old code and risk breaking things.
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
#endif

	while (c = *str++)
	{
		hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
	}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

	return hash;
}

bool is_live_stream(const char * path)
{
	TAT(TATPARMS);

	const char *url_schema = "://";
	return (NULL != strstr(path, url_schema));
}
