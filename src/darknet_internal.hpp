#pragma once

#ifndef __cplusplus
#error "The darknet project requires the use of a C++ compiler."
#endif

#if __cplusplus < 201703L
//#error "The darknet project requires C++17 or newer."
#endif

#include <iostream>
#include <string>

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "darknet.h"

#include "box.hpp"
#include "blas.hpp"
#include "utils.hpp"

#include "darknet_layers.hpp"
