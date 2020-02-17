#pragma once
#include <cuda_runtime.h>
#include "defines.h"
#include "typesAndStructs.h"
#include "cultures.cuh"
#include "modules.cuh"

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


void generateBuildings(int noBuildings,
	int noTypes,
	int types[],
	int noModels,
	int noBuildingsInTypesBfr[],
	int noPlotCornersInBuildingsBfr[],
	float2 plotCorners[], 
	int noArgumentsInBuildingsBfr[],
	int argumentsInBuildings[],
	int cultures[],
	int noAssetsInBuildingsBfr[],
	int assetsInBuildings[],
	int noCollidersInModelsBfr[],
	float3 colliderVertices[],
	int* out_noModelsInBuildingsBfr[], 
	int* out_noWallsInBuildingsBfr[],
	int* out_noAssetsInWallsBfr[],
	int* out_noArgumentsInWallsBfr[],
	float3* out_modelPositions[],
	float3* out_modelRotations[],
	int* out_modelIds[],
	wallsInfo* out_walls,
	int* out_wallAssets[],
	int* out_wallArguments[],
	int* out_wallCultures[])
{
	*out_noModelsInBuildingsBfr = (int*)malloc(sizeof(int)*(noBuildings + 1));
	*out_noWallsInBuildingsBfr = (int*)malloc(sizeof(int)*(noBuildings + 1));
	int* d_noPlotCornersInBuildingsBfr;
	float2* d_plotCorners;

	gpuErrchk(cudaMalloc(&d_noPlotCornersInBuildingsBfr, sizeof(int) * (noBuildings + 1)));
	gpuErrchk(cudaMalloc(&d_plotCorners, sizeof(float2) * noPlotCornersInBuildingsBfr[noBuildings]));
	cudaMemcpy(d_noPlotCornersInBuildingsBfr, noPlotCornersInBuildingsBfr, sizeof(int) * (noBuildings + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_plotCorners, plotCorners, sizeof(float2) * noPlotCornersInBuildingsBfr[noBuildings], cudaMemcpyHostToDevice);

	int* d_noModelsInBuildingsBfr;
	int* d_noWallsInBuildingsBfr;
	int* d_noAssetsInWallsBfr;
	int* d_noArgumentsInWallsBfr;
	int* d_noAssetsInBuildingsBfr;
	int* d_noArgumentsInBuildingsBfr;
	int* d_noCollidersInModelsBfr;

	float3* d_wallPositions;
	float3* d_wallRotations;
	float3* d_wallDimensions;
	float3* d_modelPositions;
	float3* d_modelRotations;
	int* d_modelIds;
	float3* d_collidersVertices;
	int* d_wallTypes;
	int* d_wallBuildings;
	int* d_wallAssets;
	int* d_argumentsInBuildings;
	int* d_cultures;
	int* d_assetsInBuildings;
	int* d_wallArguments;
	int* d_wallCultures;

	int maxNoModels = noBuildings * 5;
	int maxNoWalls = noBuildings * 6;	//TODO: change to sum of maxes depending on type
	int maxNoAssets = maxNoWalls * 6;
	int noArguments = noArgumentsInBuildingsBfr[noBuildings];
	int maxNoWallArguments = maxNoWalls * 10;
	int noAssetsInBuildings = noAssetsInBuildingsBfr[noBuildings];
	int noColliders = noCollidersInModelsBfr[noModels+1];
	gpuErrchk(cudaMalloc(&d_noModelsInBuildingsBfr, sizeof(int) * (noBuildings + 1)));
	cudaMemset(d_noModelsInBuildingsBfr, 0, sizeof(int) * (noBuildings + 1));
	gpuErrchk(cudaMalloc(&d_noWallsInBuildingsBfr, sizeof(int) * (noBuildings + 1)));
	cudaMemset(d_noWallsInBuildingsBfr, 0, sizeof(int) * (noBuildings + 1));
	gpuErrchk(cudaMalloc(&d_noAssetsInBuildingsBfr, sizeof(int) * (noBuildings + 1)));
	cudaMemcpy(d_noAssetsInBuildingsBfr, noAssetsInBuildingsBfr, sizeof(int) * (noBuildings + 1), cudaMemcpyHostToDevice);
	gpuErrchk(cudaMalloc(&d_noAssetsInWallsBfr, sizeof(int) * (maxNoWalls + 1)));
	cudaMemset(d_noAssetsInWallsBfr, 0, sizeof(int) * (maxNoWalls + 1));
	gpuErrchk(cudaMalloc(&d_noArgumentsInWallsBfr, sizeof(int) * (maxNoWalls + 1)));
	cudaMemset(d_noArgumentsInWallsBfr, 0, sizeof(int) * (maxNoWalls + 1));
	gpuErrchk(cudaMalloc(&d_noArgumentsInBuildingsBfr, sizeof(int) * (noBuildings + 1)));
	cudaMemset(d_noArgumentsInBuildingsBfr, 0, sizeof(int) * (noBuildings + 1));
	gpuErrchk(cudaMalloc(&d_cultures, sizeof(int) * noBuildings));
	cudaMemcpy(d_cultures, cultures, sizeof(int) * noBuildings, cudaMemcpyHostToDevice);
	gpuErrchk(cudaMalloc(&d_wallCultures, sizeof(int) * maxNoWalls));
	gpuErrchk(cudaMalloc(&d_assetsInBuildings, sizeof(int) * noAssetsInBuildings));
	cudaMemcpy(d_assetsInBuildings, assetsInBuildings, sizeof(int) * noAssetsInBuildings, cudaMemcpyHostToDevice);
	gpuErrchk(cudaMalloc(&d_wallPositions, sizeof(float3) * maxNoWalls));
	gpuErrchk(cudaMalloc(&d_wallRotations, sizeof(float3) * maxNoWalls));
	gpuErrchk(cudaMalloc(&d_wallDimensions, sizeof(float3) * maxNoWalls));
	gpuErrchk(cudaMalloc(&d_modelPositions, sizeof(float3) * maxNoModels));
	gpuErrchk(cudaMalloc(&d_modelRotations, sizeof(float3) * maxNoModels));
	gpuErrchk(cudaMalloc(&d_modelIds, sizeof(int) * maxNoModels));
	gpuErrchk(cudaMalloc(&d_wallTypes, sizeof(int) * maxNoWalls));
	gpuErrchk(cudaMalloc(&d_wallBuildings, sizeof(int) * maxNoWalls));
	gpuErrchk(cudaMalloc(&d_wallAssets, sizeof(int) * maxNoAssets));
	cudaMemset(d_wallAssets, 0, sizeof(int) * maxNoAssets);
	gpuErrchk(cudaMalloc(&d_wallArguments, sizeof(int) * maxNoWallArguments));
	cudaMemset(d_wallArguments, 0, sizeof(int) * maxNoWallArguments);
	gpuErrchk(cudaMalloc(&d_argumentsInBuildings, sizeof(int) * noArguments));
	cudaMemcpy(d_argumentsInBuildings, argumentsInBuildings, sizeof(int) * noArguments, cudaMemcpyHostToDevice);
	gpuErrchk(cudaMalloc(&d_noCollidersInModelsBfr, sizeof(int) * (noModels + 2)));
	cudaMemcpy(d_noCollidersInModelsBfr, noCollidersInModelsBfr, sizeof(int) * (noModels + 2), cudaMemcpyHostToDevice);
	gpuErrchk(cudaMalloc(&d_collidersVertices, sizeof(float3) * (noColliders * 8)));
	cudaMemcpy(d_collidersVertices, d_collidersVertices, sizeof(float3) * (noColliders * 8), cudaMemcpyHostToDevice);

	for (int typeIndex = 0; typeIndex < noTypes; typeIndex++)
	{
		int firstBuildingIndex = noBuildingsInTypesBfr[typeIndex];
		int noBuildingsThisType = noBuildingsInTypesBfr[typeIndex + 1] - firstBuildingIndex;
		while (noBuildingsThisType > NO_THREADS)
		{
			if (types[typeIndex] == 0)
			{
				generateTenementGPU << <1, NO_THREADS >> > (NO_THREADS,
					d_noModelsInBuildingsBfr + firstBuildingIndex,
					d_noWallsInBuildingsBfr + firstBuildingIndex,
					d_noPlotCornersInBuildingsBfr + firstBuildingIndex,
					d_noAssetsInBuildingsBfr + firstBuildingIndex,
					d_noAssetsInWallsBfr,
					d_plotCorners,
					d_noArgumentsInBuildingsBfr,
					d_argumentsInBuildings,
					d_cultures,
					d_assetsInBuildings,
					d_noCollidersInModelsBfr,
					d_collidersVertices,
					d_modelPositions,
					d_modelRotations,
					d_modelIds,
					d_wallPositions, 
					d_wallRotations,
					d_wallDimensions,
					d_wallTypes,
					d_wallAssets,
					d_noArgumentsInWallsBfr,
					d_wallArguments,
					d_wallCultures);
				firstBuildingIndex += NO_THREADS;
				noBuildingsThisType -= NO_THREADS;
				cudaDeviceSynchronize();
			}
		}
		if (types[typeIndex] == 0)
		{
			generateTenementGPU << <1, NO_THREADS >> > (noBuildingsThisType,
				d_noModelsInBuildingsBfr + firstBuildingIndex,
				d_noWallsInBuildingsBfr + firstBuildingIndex,
				d_noPlotCornersInBuildingsBfr + firstBuildingIndex,
				d_noAssetsInBuildingsBfr + firstBuildingIndex,
				d_noAssetsInWallsBfr,
				d_plotCorners,
				d_noArgumentsInBuildingsBfr,
				d_argumentsInBuildings,
				d_cultures,
				d_assetsInBuildings,
				d_noCollidersInModelsBfr,
				d_collidersVertices,
				d_modelPositions,
				d_modelRotations,
				d_modelIds,
				d_wallPositions,
				d_wallRotations,
				d_wallDimensions,
				d_wallTypes,
				d_wallAssets,
				d_noArgumentsInWallsBfr,
				d_wallArguments,
				d_wallCultures);
			firstBuildingIndex += NO_THREADS;
			noBuildingsThisType -= NO_THREADS;
			cudaDeviceSynchronize();
		}
	}

	int noBlocks = (noBuildings - 1) / NO_THREADS + 1;
	markWallsWithBuildings << <noBlocks, NO_THREADS >> > (noBuildings, d_noWallsInBuildingsBfr, d_wallBuildings);
	cudaMemcpy(*out_noModelsInBuildingsBfr, d_noModelsInBuildingsBfr, sizeof(int) * (noBuildings + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_noWallsInBuildingsBfr, d_noWallsInBuildingsBfr, sizeof(int) * (noBuildings + 1), cudaMemcpyDeviceToHost);

	int noWalls = (*out_noWallsInBuildingsBfr)[noBuildings];
	out_walls->positions = (float3*)malloc(sizeof(float3) * noWalls);
	out_walls->rotations = (float3*)malloc(sizeof(float3) * noWalls);
	out_walls->dimensions = (float3*)malloc(sizeof(float3) * noWalls);
	out_walls->rotations = (float3*)malloc(sizeof(float3) * noWalls);
	out_walls->dimensions = (float3*)malloc(sizeof(float3) * noWalls);
	out_walls->types = (int*)malloc(sizeof(int) * noWalls);
	out_walls->buildingIndexes = (int*)malloc(sizeof(int) * noWalls);
	*out_noAssetsInWallsBfr = (int*)malloc(sizeof(int)*(noWalls+1));
	*out_noArgumentsInWallsBfr = (int*)malloc(sizeof(int)*(noWalls+1));
	*out_wallCultures = (int*)malloc(sizeof(int)*(noWalls));
	cudaMemcpy(out_walls->positions, d_wallPositions, sizeof(float3) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_walls->rotations, d_wallRotations, sizeof(float3) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_walls->dimensions, d_wallDimensions, sizeof(float3) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_walls->types, d_wallTypes, sizeof(int) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(out_walls->buildingIndexes, d_wallBuildings, sizeof(int) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_noAssetsInWallsBfr, d_noAssetsInWallsBfr, sizeof(int) * (noWalls+1), cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_noArgumentsInWallsBfr, d_noArgumentsInWallsBfr, sizeof(int) * (noWalls+1), cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_wallCultures, d_wallCultures, sizeof(int) * (noWalls), cudaMemcpyDeviceToHost);

	int noObjects = (*out_noModelsInBuildingsBfr)[noBuildings];
	*out_modelPositions = (float3*)malloc(sizeof(float3) * noObjects);
	*out_modelRotations = (float3*)malloc(sizeof(float3) * noObjects);
	*out_modelIds = (int*)malloc(sizeof(int) * noObjects);
	cudaMemcpy(*out_modelPositions, d_modelPositions, sizeof(float3) * noObjects, cudaMemcpyHostToDevice);
	cudaMemcpy(*out_modelRotations, d_modelRotations, sizeof(float3) * noObjects, cudaMemcpyHostToDevice);
	cudaMemcpy(*out_modelIds, d_modelIds, sizeof(int) * noObjects, cudaMemcpyHostToDevice);

	int noAssets = (*out_noAssetsInWallsBfr)[noWalls];
	*out_wallAssets = (int*)malloc(sizeof(int)*noAssets);
	cudaMemcpy(*out_wallAssets, d_wallAssets, sizeof(int) * noAssets, cudaMemcpyDeviceToHost);

	int noGeneratedArguments = (*out_noArgumentsInWallsBfr)[noWalls];
	*out_wallArguments = (int*)malloc(sizeof(int)*noGeneratedArguments);
	cudaMemcpy(*out_wallArguments, d_wallArguments, sizeof(int) * noGeneratedArguments, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cudaFree(d_wallCultures);
	cudaFree(d_noCollidersInModelsBfr);
	cudaFree(d_collidersVertices);
	cudaFree(d_wallAssets);
	cudaFree(d_wallArguments);
	cudaFree(d_noAssetsInWallsBfr);
	cudaFree(d_noModelsInBuildingsBfr);
	cudaFree(d_noWallsInBuildingsBfr);
	cudaFree(d_modelPositions);
	cudaFree(d_modelRotations);
	cudaFree(d_modelIds);
	cudaFree(d_wallPositions);
	cudaFree(d_wallRotations);
	cudaFree(d_wallDimensions);
	cudaFree(d_wallTypes);
	cudaFree(d_wallBuildings);
	cudaFree(d_noPlotCornersInBuildingsBfr);
	cudaFree(d_plotCorners);
}

