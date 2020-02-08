#pragma once
#include <cuda_runtime.h>
#include "defines.h"
#include "typesAndStructs.h"
#include <string>
#include "cub/block/block_scan.cuh"

__global__
void generateAmazingGPU(int noBuildings, int noModelsInBuildingsBfr[], int noWallsInBuildingsBfr[], int noPlotCornersInBuildingsBfr[], float2 plotCorners[],
	modelInfo out_models[], float3 out_wallPositions[], float3 out_wallRotations[], float3 out_wallDimensions[], WallType out_wallTypes[])
{
	const float cottageHeight = 4;
	const float cottageWidth = 0.4f;
	int building = threadIdx.x;
	if (building < noBuildings)
	{
		noModelsInBuildingsBfr[building] = 0;

		int plotCornersBegin = noPlotCornersInBuildingsBfr[building];
		int plotCornersEnd = noPlotCornersInBuildingsBfr[building + 1];
		int noWalls = plotCornersEnd - plotCornersBegin;
		int noWallsBfr;
		if (building == 0)
		{
			noWalls += noWallsInBuildingsBfr[0];
		}
		typedef cub::BlockScan<int, NO_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;
		__shared__ typename BlockScan::TempStorage temp_storage;
		BlockScan(temp_storage).ExclusiveSum(noWalls, noWallsBfr);
		if (building == 0)
		{
			noWallsBfr += noWallsInBuildingsBfr[0];
			noWalls -= noWallsInBuildingsBfr[0];
		}
		noWallsInBuildingsBfr[building + 1] = noWalls + noWallsBfr;
		__syncthreads();
		for (int i = 0; i < noWalls; i++)
		{
			float2 corner = plotCorners[i + plotCornersBegin];
			float2 nextCorner;
			if (i == noWalls - 1)
			{
				nextCorner = plotCorners[plotCornersBegin];
			}
			else
			{
				nextCorner = plotCorners[plotCornersBegin + i + 1];
			}
			if (i % 2 == 0)
			{
				out_wallTypes[i + noWallsBfr] = WallType::AMAZING_W;
			}
			else
			{
				out_wallTypes[i + noWallsBfr] = WallType::BLUE_W;
			}
			float diffx = nextCorner.x - corner.x;
			float diffy = nextCorner.y - corner.y;
			float length = sqrt(diffx * diffx + diffy * diffy);
			out_wallPositions[i + noWallsBfr] = { corner.x, 0, corner.y };
			out_wallRotations[i + noWallsBfr] = { 0,-atan2(diffy, diffx),0};
			out_wallDimensions[i + noWallsBfr] = { length,  cottageHeight, cottageWidth };
		}
		__syncthreads();
	}
}

__global__
void markWallsWithBuildings(int noBuildings, int noWallsInBuildingsBfr[], int out_wallBuildingIndexes[])
{
	int building = threadIdx.x + blockIdx.x * blockDim.x;
	if (building < noBuildings)
	{
		for (int wall = noWallsInBuildingsBfr[building]; wall < noWallsInBuildingsBfr[building+1]; wall++)
		{
			out_wallBuildingIndexes[wall] = building;
		}
	}
}


void generateBuildings(int noBuildings, int noTypes, BuildingLogicType types[], int noBuildingsInTypesBfr[],
	int noPlotCornersInBuildingsBfr[], float2 plotCorners[],
	int* out_noModelsInBuildingsBfr[], int* out_noWallsInBuildingsBfr[], modelInfo* out_models[], wallsInfo* out_walls)
{
	*out_noModelsInBuildingsBfr = (int*)malloc(sizeof(int)*(noBuildings + 1));
	*out_noWallsInBuildingsBfr = (int*)malloc(sizeof(int)*(noBuildings + 1));
	int* d_noPlotCornersInBuildingsBfr;
	float2* d_plotCorners;

	cudaMalloc(&d_noPlotCornersInBuildingsBfr, sizeof(int) * (noBuildings + 1));
	cudaMalloc(&d_plotCorners, sizeof(float2) * noPlotCornersInBuildingsBfr[noBuildings]);
	cudaMemcpy(d_noPlotCornersInBuildingsBfr, noPlotCornersInBuildingsBfr, sizeof(int) * (noBuildings + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_plotCorners, plotCorners, sizeof(float2) * noPlotCornersInBuildingsBfr[noBuildings], cudaMemcpyHostToDevice);

	int* d_noModelsInBuildingsBfr;
	int* d_noWallsInBuildingsBfr;

	modelInfo* d_models;
	float3* d_wallPositions;
	float3* d_wallRotations;
	float3* d_wallDimensions;
	WallType* d_wallTypes;
	int* d_wallBuildings;

	cudaMalloc(&d_noModelsInBuildingsBfr, sizeof(int) * (noBuildings + 1));
	cudaMemset(d_noModelsInBuildingsBfr, 0, sizeof(int) * (noBuildings + 1));
	cudaMalloc(&d_noWallsInBuildingsBfr, sizeof(int) * (noBuildings + 1));
	cudaMemset(d_noWallsInBuildingsBfr, 0, sizeof(int) * (noBuildings + 1));
	int maxNoModels = noBuildings * 5;
	int maxNoWalls = noBuildings * 6;	//TODO: change to sum of maxes depending on type
	cudaMalloc(&d_models, sizeof(modelInfo) * maxNoModels);
	cudaMalloc(&d_wallPositions, sizeof(float3) * maxNoWalls);
	cudaMalloc(&d_wallRotations, sizeof(float3) * maxNoWalls);
	cudaMalloc(&d_wallDimensions, sizeof(float3) * maxNoWalls);
	cudaMalloc(&d_wallTypes, sizeof(WallType) * maxNoWalls);
	cudaMalloc(&d_wallBuildings, sizeof(int) * maxNoWalls);

	for (int typeIndex = 0; typeIndex < noTypes; typeIndex++)
	{
		int firstBuildingIndex = noBuildingsInTypesBfr[typeIndex];
		int noBuildingsThisType = noBuildingsInTypesBfr[typeIndex + 1] - firstBuildingIndex;
		while (noBuildingsThisType > NO_THREADS)
		{
			if (types[typeIndex] == AMAZING_B)
			{
				generateAmazingGPU << <1, NO_THREADS >> > (NO_THREADS, d_noModelsInBuildingsBfr + firstBuildingIndex, d_noWallsInBuildingsBfr + firstBuildingIndex,
					d_noPlotCornersInBuildingsBfr + firstBuildingIndex, d_plotCorners,
					d_models, d_wallPositions, d_wallRotations, d_wallDimensions, d_wallTypes);
				firstBuildingIndex += NO_THREADS;
				noBuildingsThisType -= NO_THREADS;
			}
		}
		if (types[typeIndex] == AMAZING_B)
		{
			generateAmazingGPU << <1, NO_THREADS >> > (noBuildingsThisType, d_noModelsInBuildingsBfr + firstBuildingIndex, d_noWallsInBuildingsBfr + firstBuildingIndex,
				d_noPlotCornersInBuildingsBfr + firstBuildingIndex, d_plotCorners,
				d_models, d_wallPositions, d_wallRotations, d_wallDimensions, d_wallTypes);
		}
	}
	int noBlocks = (noBuildings - 1) / NO_THREADS + 1;
	markWallsWithBuildings << <noBlocks, NO_THREADS >> > (noBuildings, d_noWallsInBuildingsBfr, d_wallBuildings);
	cudaMemcpy(*out_noModelsInBuildingsBfr, d_noModelsInBuildingsBfr, sizeof(int) * (noBuildings + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_noWallsInBuildingsBfr, d_noWallsInBuildingsBfr, sizeof(int) * (noBuildings + 1), cudaMemcpyDeviceToHost);
	*out_models = (modelInfo*)malloc(sizeof(modelInfo) * (*out_noModelsInBuildingsBfr)[noBuildings]);
	cudaMemcpy(*out_models, d_models, sizeof(modelInfo) * (*out_noModelsInBuildingsBfr)[noBuildings], cudaMemcpyDeviceToHost);
	int noWalls = (*out_noWallsInBuildingsBfr)[noBuildings];
	out_walls->positions = (float3*)malloc(sizeof(float3) * noWalls);
	out_walls->rotations = (float3*)malloc(sizeof(float3) * noWalls);
	out_walls->dimensions = (float3*)malloc(sizeof(float3) * noWalls);
	out_walls->types = (WallType*)malloc(sizeof(WallType) * noWalls);
	out_walls->buildingIndexes = (int*)malloc(sizeof(int) * noWalls);
	cudaMemcpy(out_walls->positions, d_wallPositions, sizeof(float3) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_walls->rotations, d_wallRotations, sizeof(float3) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_walls->dimensions, d_wallDimensions, sizeof(float3) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_walls->types, d_wallTypes, sizeof(WallType) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_walls->buildingIndexes, d_wallBuildings, sizeof(int) * noWalls, cudaMemcpyDeviceToHost);
	cudaFree(d_noModelsInBuildingsBfr);
	cudaFree(d_noWallsInBuildingsBfr);
	cudaFree(d_models);
	cudaFree(d_wallPositions);
	cudaFree(d_wallRotations);
	cudaFree(d_wallDimensions);
	cudaFree(d_wallTypes);
	cudaFree(d_wallBuildings);
	cudaFree(d_noPlotCornersInBuildingsBfr);
	cudaFree(d_plotCorners);
}

