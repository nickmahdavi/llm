#ifndef INIT_H
#define INIT_H

#include "struct.h"

Activations *init_acts(Config *config);
Weights *init_weights(Config *config);
Weights *init_gradient_checker(int n, Config *config);
Weights *init_copy_zeros(Weights *ex, Config *config);

#endif