#pragma once
#include <cuda_runtime.h>
#include "defines.h"
#include "typesAndStructs.h"
#include <string>
#include "cub/block/block_scan.cuh"

__global__
void generateAmazingGPU(int noBuildings, int noModelsInBuildingsBfr[], int noWallsInBuildingsBfr[], int noPlotCornersInBuildingsBfr[], float3 plotCorners[],
	modelInfo out_models[], float3 out_wallPositions[], float3 out_wallVectorXs[], float3 out_wallVectorYs[], float3 out_wallVectorWidths[], WallType out_wallTypes[])
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
			float3 corner = plotCorners[i + plotCornersBegin];
			float3 nextCorner;
			if (i == noWalls - 1)
			{
				nextCorner = plotCorners[plotCornersBegin];
			}
			else
			{
				nextCorner = plotCorners[plotCornersBegin + i + 1];
			}
			out_wallPositions[i + noWallsBfr] = { corner.x, corner.y, corner.z };
			if (i % 2 == 0)
			{
				out_wallTypes[i + noWallsBfr] = WallType::AMAZING_W;
			}
			else
			{
				out_wallTypes[i + noWallsBfr] = WallType::BLUE_W;
			}
			out_wallVectorXs[i + noWallsBfr] = { nextCorner.x - corner.x,  0, nextCorner.z - corner.z };
			out_wallVectorYs[i + noWallsBfr] = { 0,cottageHeight,0 };
			float dx = nextCorner.z - corner.z;
			float dz = corner.x - nextCorner.x;
			float denominator = sqrt(dx*dx + dz * dz);
			out_wallVectorWidths[i + noWallsBfr] = { dx*cottageWidth / denominator,0, dz*cottageWidth / denominator };
		}
		__syncthreads();
	}
}

__global__
void generateCottagesGPU(int noBuildings, int noModelsInBuildingsBfr[], int noWallsInBuildingsBfr[], int noPlotCornersInBuildingsBfr[], float3 plotCorners[],
	modelInfo out_models[], float3 out_wallPositions[], float3 out_wallVectorXs[], float3 out_wallVectorYs[], float3 out_wallVectorWidths[], WallType out_wallTypes[])
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
		bool isEnhanced = (threadIdx.x + 1) % 4 == 0;
		bool isRed = threadIdx.x % 3 == 0;
		if (isEnhanced)
			noWalls += 1;
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
		if (!isEnhanced)
		{
			for (int i = 0; i < noWalls; i++)
			{
				float3 corner = plotCorners[i + plotCornersBegin];
				float3 nextCorner;
				if (i == noWalls - 1)
				{
					nextCorner = plotCorners[plotCornersBegin];
				}
				else
				{
					nextCorner = plotCorners[plotCornersBegin + i + 1];
				}
				out_wallPositions[i + noWallsBfr] = { corner.x, corner.y, corner.z };
				out_wallTypes[i + noWallsBfr] = isRed ? WallType::RED_W : WallType::BLUE_W;
				out_wallVectorXs[i + noWallsBfr] = { nextCorner.x - corner.x,  0, nextCorner.z - corner.z };
				out_wallVectorYs[i + noWallsBfr] = { 0,cottageHeight,0 };
				float dx = nextCorner.z - corner.z;
				float dz = corner.x - nextCorner.x;
				float denominator = sqrt(dx*dx + dz * dz);
				out_wallVectorWidths[i + noWallsBfr] = { dx*cottageWidth / denominator,0, dz*cottageWidth / denominator };
			}
		}
		else
		{
			float3 first = plotCorners[noWalls + plotCornersBegin - 3];
			float3 second = plotCorners[noWalls + plotCornersBegin - 2];
			float3 third = plotCorners[plotCornersBegin];
			float3 fourth = plotCorners[plotCornersBegin + 1];
			for (int i = 0; i < noWalls; i++)
			{
				float3 corner;
				float3 nextCorner;
				if (i == noWalls - 3)
				{
					corner = first;
					nextCorner = { 0.7 * second.x + 0.3 * first.x, 0, 0.7 * second.z + 0.3 * first.z };
				}
				else if (i == noWalls - 2)
				{
					corner = {0.7 * second.x + 0.3 * first.x, 0, 0.7 * second.z + 0.3 * first.z};
					nextCorner = { 0.5 * second.x + 0.5 * third.x, 0, 0.5 * second.z + 0.5 * third.z };
				}
				else if (i == noWalls - 1)
				{
					corner = { 0.5 * second.x + 0.5 * third.x, 0, 0.5 * second.z + 0.5 * third.z };
					nextCorner = { 0.7 * third.x + 0.3 * fourth.x, 0, 0.7 * third.z + 0.3 * fourth.z };
				}
				else if (i == 0)
				{
					corner = { 0.7 * third.x + 0.3 * fourth.x, 0, 0.7 * third.z + 0.3 * fourth.z };
					nextCorner = fourth;
				}
				else
				{
					corner = plotCorners[i + plotCornersBegin];
					nextCorner = plotCorners[plotCornersBegin + i + 1];
				}
				out_wallPositions[i + noWallsBfr] = { corner.x, corner.y, corner.z };
				out_wallTypes[i + noWallsBfr] = isRed ?WallType::RED_W : WallType::BLUE_W;
				out_wallVectorXs[i + noWallsBfr] = { nextCorner.x - corner.x,  0, nextCorner.z - corner.z };
				out_wallVectorYs[i + noWallsBfr] = { 0,cottageHeight,0 };
				float dx = nextCorner.z - corner.z;
				float dz = corner.x - nextCorner.x;
				float denominator = sqrt(dx*dx + dz * dz);
				out_wallVectorWidths[i + noWallsBfr] = { dx*cottageWidth / denominator,0, dz*cottageWidth / denominator };
			}
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


void generateBuildings(int noBuildings, buildingsInfo& info, int noTypes, BuildingType types[], int noBuildingsInTypesBfr[],
	int noPlotCornersInBuildingsBfr[], float3 plotCorners[],
	int* out_noModelsInBuildingsBfr[], int* out_noWallsInBuildingsBfr[], modelInfo* out_models[], wallsInfo* out_walls)
{
	*out_noModelsInBuildingsBfr = (int*)malloc(sizeof(int)*(noBuildings + 1));
	*out_noWallsInBuildingsBfr = (int*)malloc(sizeof(int)*(noBuildings + 1));
	int* d_noPlotCornersInBuildingsBfr;
	float3* d_plotCorners;

	cudaMalloc(&d_noPlotCornersInBuildingsBfr, sizeof(int) * (noBuildings + 1));
	cudaMalloc(&d_plotCorners, sizeof(float3) * noPlotCornersInBuildingsBfr[noBuildings]);
	cudaMemcpy(d_noPlotCornersInBuildingsBfr, noPlotCornersInBuildingsBfr, sizeof(int) * (noBuildings + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_plotCorners, plotCorners, sizeof(float3) * noPlotCornersInBuildingsBfr[noBuildings], cudaMemcpyHostToDevice);

	int* d_noModelsInBuildingsBfr;
	int* d_noWallsInBuildingsBfr;

	modelInfo* d_models;
	float3* d_wallPositions;
	float3* d_wallVectorXs;
	float3* d_wallVectorYs;
	float3* d_wallVectorWidths;
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
	cudaMalloc(&d_wallVectorXs, sizeof(float3) * maxNoWalls);
	cudaMalloc(&d_wallVectorYs, sizeof(float3) * maxNoWalls);
	cudaMalloc(&d_wallVectorWidths, sizeof(float3) * maxNoWalls);
	cudaMalloc(&d_wallTypes, sizeof(WallType) * maxNoWalls);
	cudaMalloc(&d_wallBuildings, sizeof(int) * maxNoWalls);

	for (int typeIndex = 0; typeIndex < noTypes; typeIndex++)
	{
		int firstBuildingIndex = noBuildingsInTypesBfr[typeIndex];
		int noBuildingsThisType = noBuildingsInTypesBfr[typeIndex + 1] - firstBuildingIndex;
		while (noBuildingsThisType > NO_THREADS)
		{
			if (types[typeIndex] == COTTAGE_B)
			{
				generateCottagesGPU << <1, NO_THREADS >> > (NO_THREADS, d_noModelsInBuildingsBfr + firstBuildingIndex, d_noWallsInBuildingsBfr + firstBuildingIndex,
					d_noPlotCornersInBuildingsBfr + firstBuildingIndex, d_plotCorners,
					d_models, d_wallPositions, d_wallVectorXs, d_wallVectorYs, d_wallVectorWidths, d_wallTypes);
				firstBuildingIndex += NO_THREADS;
				noBuildingsThisType -= NO_THREADS;
			}
			else if (types[typeIndex] == AMAZING_B)
			{
				generateAmazingGPU << <1, NO_THREADS >> > (NO_THREADS, d_noModelsInBuildingsBfr + firstBuildingIndex, d_noWallsInBuildingsBfr + firstBuildingIndex,
					d_noPlotCornersInBuildingsBfr + firstBuildingIndex, d_plotCorners,
					d_models, d_wallPositions, d_wallVectorXs, d_wallVectorYs, d_wallVectorWidths, d_wallTypes);
				firstBuildingIndex += NO_THREADS;
				noBuildingsThisType -= NO_THREADS;
			}
		}
		if (types[typeIndex] == COTTAGE_B)
		{
			generateCottagesGPU << <1, NO_THREADS >> > (noBuildingsThisType, d_noModelsInBuildingsBfr + firstBuildingIndex, d_noWallsInBuildingsBfr + firstBuildingIndex,
				d_noPlotCornersInBuildingsBfr + firstBuildingIndex, d_plotCorners,
				d_models, d_wallPositions, d_wallVectorXs, d_wallVectorYs, d_wallVectorWidths, d_wallTypes);
		}
		else if (types[typeIndex] == AMAZING_B)
		{
			generateAmazingGPU << <1, NO_THREADS >> > (noBuildingsThisType, d_noModelsInBuildingsBfr + firstBuildingIndex, d_noWallsInBuildingsBfr + firstBuildingIndex,
				d_noPlotCornersInBuildingsBfr + firstBuildingIndex, d_plotCorners,
				d_models, d_wallPositions, d_wallVectorXs, d_wallVectorYs, d_wallVectorWidths, d_wallTypes);
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
	out_walls->vectorXs = (float3*)malloc(sizeof(float3) * noWalls);
	out_walls->vectorYs = (float3*)malloc(sizeof(float3) * noWalls);
	out_walls->vectorWidths = (float3*)malloc(sizeof(float3) * noWalls);
	out_walls->types = (WallType*)malloc(sizeof(WallType) * noWalls);
	out_walls->buildingIndexes = (int*)malloc(sizeof(int) * noWalls);
	cudaMemcpy(out_walls->positions, d_wallPositions, sizeof(float3) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_walls->vectorXs, d_wallVectorXs, sizeof(float3) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_walls->vectorYs, d_wallVectorYs, sizeof(float3) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_walls->vectorWidths, d_wallVectorWidths, sizeof(float3) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_walls->types, d_wallTypes, sizeof(WallType) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_walls->buildingIndexes, d_wallBuildings, sizeof(int) * noWalls, cudaMemcpyDeviceToHost);
	cudaFree(d_noModelsInBuildingsBfr);
	cudaFree(d_noWallsInBuildingsBfr);
	cudaFree(d_models);
	cudaFree(d_wallPositions);
	cudaFree(d_wallVectorXs);
	cudaFree(d_wallVectorYs);
	cudaFree(d_wallVectorWidths);
	cudaFree(d_wallTypes);
	cudaFree(d_wallBuildings);
	cudaFree(d_noPlotCornersInBuildingsBfr);
	cudaFree(d_plotCorners);
}

