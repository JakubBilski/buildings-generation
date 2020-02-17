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
	int triangles[])
{
	//shared memory for all pointer structures needed in earcut algorithm
	extern __shared__ int earcutTables[];
	typedef cub::BlockScan<int, NO_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;

	int wallsInThisBlock_Debug = noWallsInBlocksBfr[blockIdx.x + 1] - noWallsInBlocksBfr[blockIdx.x];
	int thisBlockWallsStart = noWallsInBlocksBfr[blockIdx.x];
	int thisBlockVerticesStart = noVerticesInWallsBfr[thisBlockWallsStart];
	int thisBlockNoVertices = noVerticesInWallsBfr[noWallsInBlocksBfr[blockIdx.x + 1]] - thisBlockVerticesStart;

	//if (threadIdx.x == 0)
	//	printf("Starting with noVertices %d\n", thisBlockNoVertices);

	//offseting, dividing into different pointer tables
	float2* verticesValues = (float2*)earcutTables;
	int* nextReflexes = earcutTables + sizeof(float2) / sizeof(int) * thisBlockNoVertices;
	int* typeOfVertices = nextReflexes + thisBlockNoVertices;	//reflex 0, non-reflex 1, ear 2
	int* ears = typeOfVertices + thisBlockNoVertices;
	int* bufferEars = ears + thisBlockNoVertices;
	int* nextVertices = bufferEars + thisBlockNoVertices;
	int* prevVertices = nextVertices + thisBlockNoVertices;
	int* vertexToWall = prevVertices + thisBlockNoVertices;
	int* rootReflexes = vertexToWall + thisBlockNoVertices;
	int* addedTrianglesInWalls = rootReflexes + wallsInThisBlock_Debug;
	int* noEarsInBlock = addedTrianglesInWalls + wallsInThisBlock_Debug;
	int* noBufferEarsInBlock = noEarsInBlock + 1;
	BlockScan::TempStorage* temp_storage1 = (BlockScan::TempStorage*)(noBufferEarsInBlock + 1);
	BlockScan::TempStorage* temp_storage2 = (BlockScan::TempStorage*)((int*)temp_storage1 + sizeof(BlockScan::TempStorage)/sizeof(int));

	//return;
	//shared mem initialization
	int index = threadIdx.x;
	while (index < thisBlockNoVertices)
	{
		verticesValues[index] = verticesInWalls[thisBlockVerticesStart + index];
		nextVertices[index] = index + 1;
		prevVertices[index] = index - 1;
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

	//return;
	__syncthreads();
	//all the following runs only for the block's walls and vertices

	//calculating number of vertices in block

	if (threadIdx.x == 0)
	{
		*noEarsInBlock = 0;
	}

	__syncthreads();

	//initializing reflex vertices
	//warp-based would be probably the fastest, but it's hard to implement
	index = threadIdx.x;
	while (index < thisBlockNoVertices)
	{
		if (!IsAngleBetweenSmallerThanPi(verticesValues[prevVertices[index]], verticesValues[index], verticesValues[nextVertices[index]]))
		{
			//adding reflex
			nextReflexes[index] = atomicExch(&(rootReflexes[vertexToWall[index]]), index);
			typeOfVertices[index] = 0;
		}
		index += blockDim.x;
	}
	//NOTE: if reflex has prev == -2, it is pointed to by the root

	__syncthreads();

	//gathering ears
	index = threadIdx.x;
	int foundEar = -1;
	if (index < thisBlockNoVertices)
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
			//printf("[%d, %d] znalazl ear\n", blockIdx.x, threadIdx.x);
		}
	}
	int count = foundEar != -1 ? 1 : 0;
	//if (threadIdx.x == 0)
	//	printf("[%d] Yo we reached scan\n", blockIdx.x);
	__syncthreads();
	int earSaveOffset;
	BlockScan(*temp_storage1).ExclusiveSum(count, earSaveOffset);
	__syncthreads();
	//if (threadIdx.x == 255)
	//	printf("[%d] Yo we past the scan with offset %d\n", blockIdx.x, earSaveOffset);
	if (foundEar != -1)
	{
		ears[earSaveOffset] = foundEar;
		atomicAdd(noEarsInBlock, 1);	//TODO: Change to aggregate
	}
	__syncthreads();
	//main loop
	while (*noEarsInBlock > 0)
	{
		//if (threadIdx.x == 0)
		//{
		//	printf("[%d, %d] myk obrut, noEarsInBlock %d\n", blockIdx.x, threadIdx.x, *noEarsInBlock);
		//}
		int noEarsInThread = 0;
		int newEars[3];
		*noBufferEarsInBlock = 0;
		index = threadIdx.x;
		if (index < (*noEarsInBlock+1)/2)
		{
			//if (blockIdx.x == 0)
			//	printf("Unfolding ear %d: %d\n", 2 * index, ears[2*index]);
			//every thread handles two ears and unfolds only one, to avoid conflicts
			int ear = ears[2 * index];
			int globalWall = vertexToWall[ear] + thisBlockWallsStart;
			int triangleIndex = atomicAdd(&(addedTrianglesInWalls[vertexToWall[ear]]), 3) + 3 * (noVerticesInWallsBfr[globalWall] - 2 * globalWall);
			if (triangleIndex < 3 * (noVerticesInWallsBfr[globalWall + 1] - 2 * (globalWall + 1)))
			{
				triangles[triangleIndex] = prevVertices[ear] + thisBlockVerticesStart - noVerticesInWallsBfr[globalWall];
				triangles[triangleIndex + 1] = ear + thisBlockVerticesStart - noVerticesInWallsBfr[globalWall];
				triangles[triangleIndex + 2] = nextVertices[ear] + thisBlockVerticesStart - noVerticesInWallsBfr[globalWall];
				//if(blockIdx.x == 0)
				//	printf("[%d, %d] saving triangle %d %d %d in %d\n", blockIdx.x, threadIdx.x, triangles[triangleIndex], triangles[triangleIndex + 1], triangles[triangleIndex + 2], triangleIndex/3);
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
					//if (blockIdx.x == 0 && (prev == 26))
					//	printf("\tAdding to ears %d\n", prev);
					typeOfVertices[prev] = 2;
					newEars[noEarsInThread] = prev;
					noEarsInThread++;
				}
				//else if(blockIdx.x == 0 && (prev == 26))
				//{
				//	printf("\tRejected ear %d %d %d\n", prevVertices[prev], prev, nextVertices[prev]);
				//}
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
			if (2 * index + 1 < *noEarsInBlock)
			{
				//if (blockIdx.x == 0)
				//	printf("Skipped ear %d: %d\n", 2 * index + 1, ears[2 * index + 1]);
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
					//if (blockIdx.x == 0)
					//	printf("Adding skipped ear %d: %d\n", 2 * index + 1, ears[2 * index + 1]);
					newEars[noEarsInThread] = ears[2 * index + 1];
					noEarsInThread++;
				}
				else
				{
					typeOfVertices[ears[2 * index + 1]] = 1;
				}
			}
		}
		//adding all gathered new ears
		__syncthreads();
		int earsInsertionIndex = 0;
		BlockScan(*temp_storage2).ExclusiveSum(noEarsInThread, earsInsertionIndex);
		for (size_t e = 0; e < noEarsInThread; e++)
		{
			bufferEars[earsInsertionIndex + e] = newEars[e];
		}
		__syncthreads();
		atomicAdd(noBufferEarsInBlock, noEarsInThread); //TODO: Change to block aggregate?
		__syncthreads();
		int* swapBuffer = bufferEars;
		bufferEars = ears;
		ears = swapBuffer;
		if (threadIdx.x == 0)
		{
			*noEarsInBlock = *noBufferEarsInBlock;
		}
		__syncthreads();
	}
	//if(threadIdx.x == 0)
	//	printf("[%d, %d] spierdalam\n", blockIdx.x, threadIdx.x);
}