#pragma once
#include "defines.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cub/block/block_scan.cuh"

__global__
void transformVerticesToWorldSpaceGPU(
	int noWalls,
	int noVerticesInWallsBfr[], 
	float3 xVectorsOfWalls[], 
	float3 yVectorsOfWalls[], 
	float2 verticesValues[],
	float3 out_output[])
{
	int wall = threadIdx.x + blockIdx.x * blockDim.x;
	if (wall < noWalls)
	{
		float3 xVectorOfWall = xVectorsOfWalls[wall];
		float3 yVectorOfWall = yVectorsOfWalls[wall];
		for (int i = noVerticesInWallsBfr[wall]; i < noVerticesInWallsBfr[wall+1]; i++)
		{
			out_output[i] = { verticesValues[i].x * xVectorOfWall.x + verticesValues[i].y * yVectorOfWall.x,
				verticesValues[i].x * xVectorOfWall.y + verticesValues[i].y * yVectorOfWall.y,
				verticesValues[i].x * xVectorOfWall.z + verticesValues[i].y * yVectorOfWall.z };
		}
	}
}
__global__
void transformHolesVerticesToWorldSpaceGPU(
	int noWalls,
	int noHolesInWallsBfr[],
	int noVerticesInHolesBfr[],
	float3 xVectorsOfWalls[],
	float3 yVectorsOfWalls[],
	float2 verticesValues[],
	float3 out_output[])
{
	int wall = threadIdx.x + blockIdx.x * blockDim.x;
	if (wall < noWalls)
	{
		float3 xVectorOfWall = xVectorsOfWalls[wall];
		float3 yVectorOfWall = yVectorsOfWalls[wall];
		for (int i = noVerticesInHolesBfr[noHolesInWallsBfr[wall]]; i < noVerticesInHolesBfr[noHolesInWallsBfr[wall + 1]]; i++)
		{
			float debug_x = verticesValues[i].x * xVectorOfWall.x + verticesValues[i].y * yVectorOfWall.x;
			float debug_y = verticesValues[i].x * xVectorOfWall.y + verticesValues[i].y * yVectorOfWall.y;
			float debug_z = verticesValues[i].x * xVectorOfWall.z + verticesValues[i].y * yVectorOfWall.z;
			out_output[i] = { verticesValues[i].x * xVectorOfWall.x + verticesValues[i].y * yVectorOfWall.x,
				verticesValues[i].x * xVectorOfWall.y + verticesValues[i].y * yVectorOfWall.y,
				verticesValues[i].x * xVectorOfWall.z + verticesValues[i].y * yVectorOfWall.z };
		}
	}
}

__global__
void normalizeGPU(
	int noElements,
	float3 elements[])
{
	int thid = threadIdx.x + blockIdx.x * blockDim.x;
	if (thid < noElements)
	{
		float x = elements[thid].x;
		float y = elements[thid].y;
		float z = elements[thid].z;
		float denominator = sqrt(x*x + y * y + z * z);
		elements[thid] = { x / denominator, y / denominator, z / denominator };
	}
}