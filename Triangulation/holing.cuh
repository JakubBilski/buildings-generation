#pragma once
#include "defines.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>


__device__
inline float Dist(float3 a, float3 b)
{
	return sqrtf((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

__global__
void mergeHolesAndContours(
	int noWallsInBlocksBfr[],
	int noVerticesInContoursBfr[],
	int noHolesInWallsBfr[],
	int noVerticesInHolesBfr[],
	float3 verticesInContours[],
	float3 verticesInHoles[],
	int memForTable,
	int out_holesAndContours[])
{
	int thisBlockFirstWall = noWallsInBlocksBfr[blockIdx.x];
	int thisBlockNoWalls = noWallsInBlocksBfr[blockIdx.x + 1] - thisBlockFirstWall;
	int thisBlockFirstContourVertex = noVerticesInContoursBfr[thisBlockFirstWall];
	int thisBlockNoContourVertices = noVerticesInContoursBfr[noWallsInBlocksBfr[blockIdx.x + 1]] - thisBlockFirstContourVertex;
	int thisBlockFirstHole = noHolesInWallsBfr[thisBlockFirstWall];
	int thisBlockNoHoles = noHolesInWallsBfr[thisBlockFirstWall + thisBlockNoWalls] - thisBlockFirstHole;
	int thisBlockFirstHoleVertex = noVerticesInHolesBfr[thisBlockFirstHole];
	int thisBlockNoHoleVertices = noVerticesInHolesBfr[thisBlockFirstHole + thisBlockNoHoles] - thisBlockFirstHoleVertex;

	extern __shared__ int tables[];
	float3* contourValues = (float3*)tables;
	int* contourSaveIndexes = tables + (sizeof(float3) / sizeof(int)) * thisBlockNoContourVertices;
	int* holesSaveIndexes = contourSaveIndexes + thisBlockNoContourVertices;
	int* holesVerticesClosestContourVertex = holesSaveIndexes + thisBlockNoHoleVertices;
	float* holesVerticesClosestContourVertexDistance = (float*)holesVerticesClosestContourVertex + thisBlockNoHoleVertices;
	int* holeVerticesInWallsBfr = (int*)holesVerticesClosestContourVertexDistance + (sizeof(float) / sizeof(int)) * thisBlockNoHoleVertices;
	int* contourVerticesInWallsBfr = holeVerticesInWallsBfr + thisBlockNoWalls;

	int index = threadIdx.x;
	if (index < thisBlockNoContourVertices)
	{
		contourValues[index] = verticesInContours[thisBlockFirstContourVertex + index];
		contourSaveIndexes[index] = 1;
		index += blockDim.x;
	}
	index = threadIdx.x;
	if (index < thisBlockNoWalls)
	{
		holeVerticesInWallsBfr[index] = noVerticesInHolesBfr[noHolesInWallsBfr[index + 1 + thisBlockFirstWall]] - noVerticesInHolesBfr[noHolesInWallsBfr[index + thisBlockFirstWall]];
		contourVerticesInWallsBfr[index] = noVerticesInContoursBfr[index + 1 + thisBlockFirstWall] - noVerticesInContoursBfr[index + thisBlockFirstWall];
	}
	for (int wall = 0; wall < thisBlockNoWalls; wall++)
	{
		int thisWallFirstHole = noHolesInWallsBfr[wall + thisBlockFirstWall];
		int thisWallNoHoles = noHolesInWallsBfr[wall + 1 + thisBlockFirstWall] - noHolesInWallsBfr[wall + thisBlockFirstWall];
		for (int hole = 0; hole < thisWallNoHoles; hole++)
		{

		}
	}


}