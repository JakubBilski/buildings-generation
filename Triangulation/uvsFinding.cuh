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
		for (int i = noVerticesInWallsBfr[wall]; i < noVerticesInWallsBfr[wall+1]; i++)
		{
			out_uvs[i] = { verticesValues[i].x - grains[wall] * (int)(verticesValues[i].x / grains[wall]),
				verticesValues[i].y - grains[wall] * (int)(verticesValues[i].y / grains[wall]) };
		}
	}
}