#ifndef GRAD_H
#define GRAD_H

#include "struct.h"

void backpropagate(Tensor *in, Tensor *labels, Activations *acts, Weights *weights, Weights *grad, Config *config);
void update(Weights *weights, Weights *grad, Config *config);

#endif