#pragma once
#include "defines.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cub/block/block_scan.cuh"


__global__
void findInsideNormalsGPU(
	int noHoles, 
	int noVerticesInHoles,
	int noVerticesInHolesBfr[], 
	float2 verticesInHoles[],
	float2 out_normalsInside[])
{
	//TODO: this is probably wrong
	int index = threadIdx.x;
	while (index < noVerticesInHoles-1)
	{
		out_normalsInside[index].y = verticesInHoles[index + 1].x - verticesInHoles[index].x;
		out_normalsInside[index].x = verticesInHoles[index].y - verticesInHoles[index + 1].y;
		index += blockDim.x;
	}
	index = threadIdx.x;
	int lastIndex, firstIndex;
	while (index < noHoles)
	{
		firstIndex = noVerticesInHolesBfr[index];
		lastIndex = noVerticesInHolesBfr[index + 1] - 1;
		out_normalsInside[lastIndex].y = verticesInHoles[firstIndex].x - verticesInHoles[lastIndex].x;
		out_normalsInside[lastIndex].x = verticesInHoles[lastIndex].y - verticesInHoles[firstIndex].y;
		index += blockDim.x;
	}
}

__global__
void findOutsideNormalsGPU(
	int noWalls, 
	int noVerticesInContours,
	int noVerticesInContoursBfr[], 
	float2 verticesInContours[],
	float2 out_normalsOutside[])
{
	int index = threadIdx.x;
	while (index < noVerticesInContours-1)
	{
		out_normalsOutside[index].y = - verticesInContours[index + 1].x + verticesInContours[index].x;
		out_normalsOutside[index].x = - verticesInContours[index].y + verticesInContours[index + 1].y;
		index += blockDim.x;
	}
	index = threadIdx.x;
	int lastIndex, firstIndex;
	while (index < noWalls)
	{
		firstIndex = noVerticesInContoursBfr[index];
		lastIndex = noVerticesInContoursBfr[index + 1] - 1;
		out_normalsOutside[lastIndex].y = - verticesInContours[firstIndex].x + verticesInContours[lastIndex].x;
		out_normalsOutside[lastIndex].x = - verticesInContours[lastIndex].y + verticesInContours[firstIndex].y;
		index += blockDim.x;
	}
}