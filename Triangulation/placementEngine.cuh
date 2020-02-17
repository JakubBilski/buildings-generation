#pragma once
#include <cuda_runtime.h>
#include "defines.h"

__device__
inline void placeAtSameXIntervals(float start, float wallLength, int noModels, float4* rectangleSizes, float3* positions)
{
	float mandatorySpace = 0;
	for (int model = 0; model < noModels; model++)
	{
		mandatorySpace += rectangleSizes[model].x + rectangleSizes[model].z;
	}
	float freeSpaceForModel = (wallLength - mandatorySpace)/noModels;
	float x = start + freeSpaceForModel / 2;
	for (int model = 0; model < noModels; model++)
	{
		positions[model].x = x + rectangleSizes[model].z;
		x += rectangleSizes[model].x + rectangleSizes[model].z + freeSpaceForModel;
	}
}