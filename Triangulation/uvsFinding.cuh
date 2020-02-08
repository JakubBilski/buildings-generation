#pragma once
#include "defines.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cub/block/block_scan.cuh"


__global__
void findUvsGPU(
	int noWalls,
	int noVerticesInWallsBfr[],
	float2 verticesValues[],
	float grains[],
	float2 out_uvs[])
{

	int wall = threadIdx.x + blockIdx.x*blockDim.x;
	if (wall < noWalls)
	{
		for (int vertex = noVerticesInWallsBfr[wall]; vertex < noVerticesInWallsBfr[wall+1]; vertex++)
		{
			out_uvs[vertex] = { verticesValues[vertex].x * grains[wall],
				verticesValues[vertex].y * grains[wall] };
		}
	}
}