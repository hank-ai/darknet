#pragma once

#include "darknet.h"

#include "image.hpp"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void send_json(detection *dets, int nboxes, int classes, char **names, long long int frame_id, int port, int timeout);

void send_mjpeg(mat_cv* mat, int port, int timeout, int quality);

int send_http_post_request(char *http_post_host, int server_port, const char *videosource,
    detection *dets, int nboxes, int classes, char **names, long long int frame_id, int ext_output, int timeout);

typedef void* custom_thread_t;
typedef void* custom_attr_t;

int custom_create_thread(custom_thread_t * tid, const custom_attr_t * attr, void *(*func) (void *), void *arg);
int custom_join(custom_thread_t thread, void **value_ptr);

int custom_atomic_load_int(volatile int* obj);
void custom_atomic_store_int(volatile int* obj, int desr);
int get_num_threads();
void this_thread_sleep_for(int ms_time);
void this_thread_yield();

#ifdef __cplusplus
}
#endif
