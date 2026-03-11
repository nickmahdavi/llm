#ifndef CHECKER_H
#define CHECKER_H

#include "struct.h"

void check_gradients(Weights *results, Tensor *in, Tensor *labels, Activations *acts, Weights *weights, Weights *grad, Config *config);

#endif