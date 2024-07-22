#include <cstdlib>
#include "malloc2D.h"

int** malloc2Di(int jmax, int imax)
{
    int** x = (int**)malloc(jmax * sizeof(int*) + jmax * imax * sizeof(int));

    x[0] = (int*)(x + jmax);

    for (int j = 1; j < jmax; j++) {
        x[j] = x[j - 1] + imax;
    }

    return(x);
}

float** malloc2Df(int jmax, int imax)
{
    float** x = (float**)malloc(jmax * sizeof(float*) + jmax * imax * sizeof(float));

    x[0] = (float*)(x + jmax);

    for (int j = 1; j < jmax; j++) {
        x[j] = x[j - 1] + imax;
    }

    return(x);
}
