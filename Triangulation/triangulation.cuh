#pragma once
#include "defines.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cub/block/block_scan.cuh"

__device__
inline int IsAngleBetweenSmallerThanPi(float2 v1, float2 v2, float2 v3)
{
	return (v1.x - v2.x)*(v3.y - v2.y) - (v1.y - v2.y)*(v3.x - v2.x) < 0;
}

__device__
inline int IsAnyReflexInsideTriangle(float2 a, float2 b, float2 c, int* nextReflexes, int* typeOfVertices, int reflexRoot, float2* verticesValues)
{
	while (reflexRoot != -2)
	{
		if (typeOfVertices[reflexRoot] == 0)
		{
			float x = verticesValues[reflexRoot].x;
			float y = verticesValues[reflexRoot].y;
			float as_x = x - a.x;
			float as_y = y - a.y;
			bool s_ab = (b.x - a.x)*as_y - (b.y - a.y)*as_x > 0;
			if ((c.x - a.x)*as_y - (c.y - a.y)*as_x > 0 != s_ab)
			{
				if ((c.x - b.x)*(y - b.y) - (c.y - b.y)*(x - b.x) > 0 == s_ab)
				{
					return true;
				}
			}
		}
		reflexRoot = nextReflexes[reflexRoot];
	}
	return false;
}

__global__
void triangulatePolygonGPU(
	int noVerticesInWallsBfr[],
	int noWallsInBlocksBfr[],
	float2 verticesInWalls[],
	int memForTable,
	int triangles[])
{
	//shared memory for all pointer structures needed in earcut algorithm
	extern __shared__ int earcutTables[];
	__shared__ int noEarsInBlock;
	__shared__ int noBufferEarsInBlock;
	float2* verticesValues = (float2*)earcutTables;

	int wallsInThisBlock_Debug = noWallsInBlocksBfr[blockIdx.x + 1] - noWallsInBlocksBfr[blockIdx.x];

	//offseting, dividing into different pointer tables
	int* nextReflexes = earcutTables + sizeof(float2) / sizeof(int) * memForTable;
	int* typeOfVertices = nextReflexes + memForTable;	//reflex 0, non-reflex 1, ear 2
	int* ears = typeOfVertices + memForTable;
	int* bufferEars = ears + memForTable;
	int* nextVertices = bufferEars + memForTable;
	int* prevVertices = nextVertices + memForTable;
	int* vertexToWall = prevVertices + memForTable;
	int* rootReflexes = vertexToWall + memForTable;
	int* addedTrianglesInWalls = rootReflexes + wallsInThisBlock_Debug;

	//shared mem initialization
	int index = threadIdx.x;
	int thisBlockWallsStart = noWallsInBlocksBfr[blockIdx.x];
	int thisBlockVerticesStart = noVerticesInWallsBfr[thisBlockWallsStart];
	while (index < memForTable)
	{
		verticesValues[index] = verticesInWalls[thisBlockVerticesStart + index];
		nextVertices[index] = index + 1;
		prevVertices[index + 1] = index;
		ears[index] = -2;	//TODO: this is unnecessary, because we have noEarsInBlock
		bufferEars[index] = -2;	//TODO: this is unnecessary, because we have noEarsInBlock
		nextReflexes[index] = -2;
		typeOfVertices[index] = 1;
		index += blockDim.x;
	}
	__syncthreads();
	index = threadIdx.x;
	//this runs only for walls handled by this block
	if (index < wallsInThisBlock_Debug)
	{
		rootReflexes[index] = -2;
		addedTrianglesInWalls[index] = 0;
		//linking ends of each wall's list
		int rightEnd = noVerticesInWallsBfr[index + thisBlockWallsStart + 1] - thisBlockVerticesStart - 1;
		int leftEnd = noVerticesInWallsBfr[index + thisBlockWallsStart] - thisBlockVerticesStart;
		nextVertices[rightEnd] = leftEnd;
		prevVertices[leftEnd] = rightEnd;
		//creating vertex to wall table
		for (int i = leftEnd; i <= rightEnd; i++)
		{
			vertexToWall[i] = index;
		}
	}

	__syncthreads();
	//all the following runs only for the block's walls and vertices

	//calculating number of vertices in block
	int noVerticesInThisBlock = noVerticesInWallsBfr[noWallsInBlocksBfr[blockIdx.x + 1]] - thisBlockVerticesStart;
	__syncthreads();

	if (threadIdx.x == 0)
	{
		noEarsInBlock = 0;
	}

	__syncthreads();

	//initializing reflex vertices
	//warp-based would be probably the fastest, but it's hard to implement
	index = threadIdx.x;
	while (index < noVerticesInThisBlock)
	{
		if (!IsAngleBetweenSmallerThanPi(verticesValues[prevVertices[index]], verticesValues[index], verticesValues[nextVertices[index]]))
		{
			//adding reflex
			nextReflexes[index] = atomicExch(&(rootReflexes[vertexToWall[index]]), index);
			__syncthreads();
			typeOfVertices[index] = 0;
		}
		index += blockDim.x;
	}
	//NOTE: if reflex has prev == -2, it is pointed to by the root

	__syncthreads();


	//gathering ears
	index = threadIdx.x;
	int foundEar = -1;
	int earSaveOffset;
	if (index < noVerticesInThisBlock)
	{
		//1. reflex cannot be an ear
		//2. zero index is never an ear
		//3. triangle made by ear and its two neighbors doesn't have any vertices inside
		if (typeOfVertices[index] == 1 &&
			index != noVerticesInWallsBfr[vertexToWall[index]] &&
			!IsAnyReflexInsideTriangle(
				verticesValues[prevVertices[index]],
				verticesValues[index],
				verticesValues[nextVertices[index]],
				nextReflexes,
				typeOfVertices,
				rootReflexes[vertexToWall[index]],
				verticesValues))
		{
			typeOfVertices[index] = 2;
			foundEar = index;
		}
	}
	__syncthreads();
	typedef cub::BlockScan<int, NO_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;
	__shared__ typename BlockScan::TempStorage temp_storage;
	__syncthreads();
	int totalEars;
	BlockScan(temp_storage).ExclusiveSum(foundEar != -1 ? 1 : 0, earSaveOffset, totalEars);
	__syncthreads();
	if (foundEar != -1)
	{
		ears[earSaveOffset] = foundEar;
		atomicAdd(&noEarsInBlock, 1);	//TODO: Change to prefixSum aggregate
	}
	__syncthreads();
	//main loop
	while (noEarsInBlock > 0)
	{
		int noEarsInThread = 0;
		int newEars[3];
		noBufferEarsInBlock = 0;
		index = threadIdx.x;
		if (index < (noEarsInBlock+1)/2)
		{
			//every thread handles two ears and unfolds only one, to avoid conflicts
			int ear = ears[2 * index];
			int globalWall = vertexToWall[ear] + thisBlockWallsStart;
			int triangleIndex = atomicAdd(&(addedTrianglesInWalls[vertexToWall[ear]]), 3) + 3 * (noVerticesInWallsBfr[globalWall] - 2 * globalWall);
			if (triangleIndex < 3 * (noVerticesInWallsBfr[globalWall + 1] - 2 * (globalWall + 1)))
			{
				triangles[triangleIndex] = prevVertices[ear] + thisBlockVerticesStart - noVerticesInWallsBfr[globalWall];
				triangles[triangleIndex + 1] = ear + thisBlockVerticesStart - noVerticesInWallsBfr[globalWall];
				triangles[triangleIndex + 2] = nextVertices[ear] + thisBlockVerticesStart - noVerticesInWallsBfr[globalWall];
			}

			int next = nextVertices[ear];
			int prev = prevVertices[ear];
			prevVertices[next] = prev;
			nextVertices[prev] = next;
			//changing some reflex to non-reflex
			if (typeOfVertices[prev] == 0)
			{
				if (IsAngleBetweenSmallerThanPi(verticesValues[prevVertices[prev]], verticesValues[prev], verticesValues[nextVertices[prev]]))
				{
					typeOfVertices[prev] = 1;
				}
			}
			if (typeOfVertices[next] == 0)
			{
				if (IsAngleBetweenSmallerThanPi(verticesValues[prevVertices[next]], verticesValues[next], verticesValues[nextVertices[next]]))
				{
					typeOfVertices[next] = 1;
				}
			}
			//checking for ears in non-reflex
			if (typeOfVertices[prev] == 1)
			{
				if (prev != noVerticesInWallsBfr[vertexToWall[prev]] &&
					!IsAnyReflexInsideTriangle(verticesValues[prevVertices[prev]],
						verticesValues[prev],
						verticesValues[nextVertices[prev]],
						nextReflexes,
						typeOfVertices,
						rootReflexes[vertexToWall[prev]],
						verticesValues))
				{
					//will add prev to ears
					typeOfVertices[prev] = 2;
					newEars[noEarsInThread] = prev;
					noEarsInThread++;
				}
			}
			if (typeOfVertices[next] == 1)
			{
				if (next != noVerticesInWallsBfr[vertexToWall[next]] &&
					!IsAnyReflexInsideTriangle(verticesValues[prevVertices[next]],
						verticesValues[next],
						verticesValues[nextVertices[next]],
						nextReflexes,
						typeOfVertices,
						rootReflexes[vertexToWall[next]],
						verticesValues))
				{
					//will add next to ears
					typeOfVertices[next] = 2;
					newEars[noEarsInThread] = next;
					noEarsInThread++;
				}
			}
			if (2 * index + 1 < noEarsInBlock)
			{
				int skippedEar = ears[2 * index + 1];
				//will add skipped ear to ears if it's still an ear
				if (!IsAnyReflexInsideTriangle(verticesValues[prevVertices[skippedEar]],
					verticesValues[skippedEar],
					verticesValues[nextVertices[skippedEar]],
					nextReflexes,
					typeOfVertices,
					rootReflexes[vertexToWall[skippedEar]],
					verticesValues))
				{
					newEars[noEarsInThread] = ears[2 * index + 1];
					noEarsInThread++;
				}
			}
		}
		//adding all gathered new ears
		__syncthreads();
		int earsInsertionIndex = 0;
		typedef cub::BlockScan<int, NO_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;
		__shared__ typename BlockScan::TempStorage temp_storage;
		BlockScan(temp_storage).ExclusiveSum(noEarsInThread, earsInsertionIndex);
		for (size_t e = 0; e < noEarsInThread; e++)
		{
			bufferEars[earsInsertionIndex + e] = newEars[e];
		}
		__syncthreads();
		atomicAdd(&noBufferEarsInBlock, noEarsInThread); //TODO: Change to blockscan aggregate
		__syncthreads();
		int* swapBuffer = bufferEars;
		bufferEars = ears;
		ears = swapBuffer;
		noEarsInBlock = noBufferEarsInBlock;
		__syncthreads();
	}
}