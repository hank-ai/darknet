#pragma once

#include "layer.hpp"
#include "network.hpp"

typedef layer cost_layer;

#ifdef __cplusplus
extern "C" {
#endif
COST_TYPE get_cost_type(char *s);
char *get_cost_string(COST_TYPE a);
cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale);
void forward_cost_layer(const cost_layer l, network_state state);
void backward_cost_layer(const cost_layer l, network_state state);
void resize_cost_layer(cost_layer *l, int inputs);

#ifdef GPU
void forward_cost_layer_gpu(cost_layer l, network_state state);
void backward_cost_layer_gpu(const cost_layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
