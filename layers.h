#ifndef LAYERS_H
#define LAYERS_H

#include "struct.h"

void forward(Tensor *in, Weights *weights, Activations *acts, Config *config);
void forward_from(int layer, Tensor *in, Weights *weights, Activations *acts, Config *config);

#endif