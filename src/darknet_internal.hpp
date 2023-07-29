#pragma once

#ifndef __cplusplus
#error "The darknet project requires the use of a C++ compiler."
#endif

#if __cplusplus < 201703L
#error "The darknet project requires C++17 or newer."
#endif

#include <iostream>
#include <cmath>

#include "darknet.hpp"
#include "blas.hpp"
#include "utils.hpp"
#include "darknet_layers.hpp"
