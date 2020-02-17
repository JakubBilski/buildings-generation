#pragma once
#include <cuda_runtime.h>
#include "defines.h"
#include "typesAndStructs.h"
#include "walls.cuh"



void generateWalls(int noWalls, 
	wallsInfo info,
	int noTypes,
	int noModels,
	int noPunchers,
	int noWallsInTypesBfr[],
	int noArgumentsInWallsBfr[],
	int noAssetsInWallsBfr[],
	int noCollidersInModelsBfr[],
	int modelsOfPunchers[],
	int noVerticesInPunchersBfr[],
	int cultures[],
	int arguments[],
	int assetsInWalls[],
	float3 colliderVertices[],
	float2 puncherVertices[],
	int* out_noModelsInWallsBfr[], 
	int* out_noVerticesInContoursBfr[],
	int* out_noVerticesInHolesBfr[],
	int* out_noHolesInWallsBfr[],
	int* out_frontMaterials[],
	int* out_backMaterials[],
	int* out_innerMaterials[],
	int* out_outerMaterials[],
	float* out_frontMaterialGrains[],
	float* out_backMaterialGrains[],
	float* out_innerMaterialGrains[],
	float* out_outerMaterialGrains[],
	float2* out_verticesInContours[],
	float2* out_verticesInHoles[],
	float3* out_modelPositions[],
	float3* out_modelRotations[],
	int* out_modelIds[]
)
{
	*out_frontMaterials = (int*)malloc(sizeof(int) * noWalls);
	*out_backMaterials = (int*)malloc(sizeof(int) * noWalls);
	*out_innerMaterials = (int*)malloc(sizeof(int) * noWalls);
	*out_outerMaterials = (int*)malloc(sizeof(int) * noWalls);
	*out_frontMaterialGrains = (float*)malloc(sizeof(float)*noWalls);
	*out_backMaterialGrains = (float*)malloc(sizeof(float)*noWalls);
	*out_innerMaterialGrains = (float*)malloc(sizeof(float)*noWalls);
	*out_outerMaterialGrains = (float*)malloc(sizeof(float)*noWalls);

	*out_noModelsInWallsBfr = (int*)malloc(sizeof(int)*(noWalls + 1));
	*out_noVerticesInContoursBfr = (int*)malloc(sizeof(int)*(noWalls + 1));
	*out_noHolesInWallsBfr = (int*)malloc(sizeof(int)*(noWalls + 1));

	int maxHoles = noWalls * 40;
	int maxContourVertices = 4 * noWalls;
	int maxHolesVertices = maxHoles * 6;
	int maxModels = 40 * noWalls;

	int* d_noModelsInWallsBfr;
	int* d_noVerticesInContoursBfr;
	int* d_noVerticesInHolesBfr;
	int* d_noHolesInWallsBfr;
	int* d_frontMaterials;
	int* d_backMaterials;
	int* d_innerMaterials;
	int* d_outerMaterials;
	float* d_frontMaterialGrains;
	float* d_backMaterialGrains;
	float* d_innerMaterialGrains;
	float* d_outerMaterialGrains;
	float2* d_verticesInContours;
	float2* d_verticesInHoles;
	float3* d_modelPositions;
	float3* d_modelRotations;
	int* d_modelIds;

	gpuErrchk(cudaMalloc(&d_noModelsInWallsBfr, sizeof(int) * (noWalls + 1)));
	cudaMemset(d_noModelsInWallsBfr, 0, sizeof(int) * (noWalls + 1));
	gpuErrchk(cudaMalloc(&d_noVerticesInContoursBfr, sizeof(int) * (noWalls + 1)));
	cudaMemset(d_noVerticesInContoursBfr, 0, sizeof(int) * (noWalls + 1));
	gpuErrchk(cudaMalloc(&d_noHolesInWallsBfr, sizeof(int) * (noWalls + 1)));
	cudaMemset(d_noHolesInWallsBfr, 0, sizeof(int) * (noWalls + 1));
	gpuErrchk(cudaMalloc(&d_noVerticesInHolesBfr, sizeof(int) * (maxHoles + 1)));
	cudaMemset(d_noVerticesInHolesBfr, 0, sizeof(int) * (maxHoles + 1));
	gpuErrchk(cudaMalloc(&d_frontMaterials, sizeof(int) * (noWalls)));
	gpuErrchk(cudaMalloc(&d_backMaterials, sizeof(int) * (noWalls)));
	gpuErrchk(cudaMalloc(&d_innerMaterials, sizeof(int) * (noWalls)));
	gpuErrchk(cudaMalloc(&d_outerMaterials, sizeof(int) * (noWalls)));
	gpuErrchk(cudaMalloc(&d_frontMaterialGrains, sizeof(float) * (noWalls)));
	gpuErrchk(cudaMalloc(&d_backMaterialGrains, sizeof(float) * (noWalls)));
	gpuErrchk(cudaMalloc(&d_innerMaterialGrains, sizeof(float) * (noWalls)));
	gpuErrchk(cudaMalloc(&d_outerMaterialGrains, sizeof(float) * (noWalls)));
	gpuErrchk(cudaMalloc(&d_verticesInContours, sizeof(float2) * (maxContourVertices)));
	gpuErrchk(cudaMalloc(&d_verticesInHoles, sizeof(float2) * (maxHolesVertices)));
	gpuErrchk(cudaMalloc(&d_modelPositions, sizeof(float3) * (maxModels)));
	gpuErrchk(cudaMalloc(&d_modelRotations, sizeof(float3) * (maxModels)));
	gpuErrchk(cudaMalloc(&d_modelIds, sizeof(int) * (maxModels)));

	int* d_noArgumentsInWallsBfr;
	int* d_noAssetsInWallsBfr;
	int* d_noCollidersInModelsBfr;
	float3* d_dimensions;
	int* d_modelsOfPunchers;
	int* d_noVerticesInPunchersBfr;
	int* d_cultures;
	int* d_arguments;
	int* d_assetsInWalls;
	float3* d_colliderVertices;
	float2* d_puncherVertices;

	int noArguments = noArgumentsInWallsBfr[noWalls];
	int noAssetsInWalls = noAssetsInWallsBfr[noWalls];
	int noColliders = noCollidersInModelsBfr[noModels + 1];
	int noPuncherVertices = noVerticesInPunchersBfr[noPunchers+1];

	gpuErrchk(cudaMalloc(&d_noArgumentsInWallsBfr, sizeof(int) * (noWalls + 1)));
	gpuErrchk(cudaMalloc(&d_noAssetsInWallsBfr, sizeof(int) * (noWalls + 1)));
	gpuErrchk(cudaMalloc(&d_noCollidersInModelsBfr, sizeof(int) * (noModels + 2)));
	gpuErrchk(cudaMalloc(&d_dimensions, sizeof(float3) * (noWalls)));
	gpuErrchk(cudaMalloc(&d_modelsOfPunchers, sizeof(int) * (noPunchers+1)));
	gpuErrchk(cudaMalloc(&d_noVerticesInPunchersBfr, sizeof(int) * (noPunchers + 2)));
	gpuErrchk(cudaMalloc(&d_cultures, sizeof(int) * (noWalls)));
	gpuErrchk(cudaMalloc(&d_arguments, sizeof(int) * (noArguments)));
	gpuErrchk(cudaMalloc(&d_assetsInWalls, sizeof(int) * (noAssetsInWalls)));
	gpuErrchk(cudaMalloc(&d_colliderVertices, sizeof(float3) * (noColliders * 8)));
	gpuErrchk(cudaMalloc(&d_puncherVertices, sizeof(float2) * (noPuncherVertices)));

	cudaMemcpy(d_noArgumentsInWallsBfr, noArgumentsInWallsBfr, sizeof(int) * (noWalls + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_noAssetsInWallsBfr, noAssetsInWallsBfr, sizeof(int) * (noWalls + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_noCollidersInModelsBfr, noCollidersInModelsBfr, sizeof(int) * (noModels + 2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dimensions, info.dimensions, sizeof(float3) * noWalls, cudaMemcpyHostToDevice);
	cudaMemcpy(d_modelsOfPunchers, modelsOfPunchers, sizeof(int) * (noPunchers+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_noVerticesInPunchersBfr, noVerticesInPunchersBfr, sizeof(int) * (noPunchers + 2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cultures, cultures, sizeof(int) * noWalls, cudaMemcpyHostToDevice);
	cudaMemcpy(d_arguments, arguments, sizeof(int) * noArguments, cudaMemcpyHostToDevice);
	cudaMemcpy(d_assetsInWalls, assetsInWalls, sizeof(int) * noAssetsInWalls, cudaMemcpyHostToDevice);
	cudaMemcpy(d_colliderVertices, colliderVertices, sizeof(float3) * (noColliders * 8), cudaMemcpyHostToDevice);
	cudaMemcpy(d_puncherVertices, puncherVertices, sizeof(float2) * noPuncherVertices, cudaMemcpyHostToDevice);
	for (int typeIndex = 0; typeIndex < noTypes; typeIndex++)
	{
		int firstWallIndex = noWallsInTypesBfr[typeIndex];
		int noWallsThisType = noWallsInTypesBfr[typeIndex + 1] - firstWallIndex;
		while (noWallsThisType > NO_THREADS)
		{
			if (info.types[firstWallIndex] == 0)
			{
				generateTenementWallsGPU << <1, NO_THREADS >> > (NO_THREADS, 
					d_noModelsInWallsBfr + firstWallIndex,
					d_noVerticesInContoursBfr + firstWallIndex,
					d_noVerticesInHolesBfr,
					d_noHolesInWallsBfr + firstWallIndex,
					d_noArgumentsInWallsBfr + firstWallIndex,
					d_noAssetsInWallsBfr + firstWallIndex,
					d_noCollidersInModelsBfr,
					d_modelsOfPunchers,
					d_noVerticesInPunchersBfr,
					d_cultures,
					d_dimensions + firstWallIndex,
					d_arguments,
					d_assetsInWalls,
					d_colliderVertices,
					d_puncherVertices,
					d_frontMaterials + firstWallIndex,
					d_backMaterials + firstWallIndex, 
					d_innerMaterials + firstWallIndex,
					d_outerMaterials + firstWallIndex,
					d_frontMaterialGrains + firstWallIndex, 
					d_backMaterialGrains + firstWallIndex,
					d_innerMaterialGrains + firstWallIndex,
					d_outerMaterialGrains + firstWallIndex,
					d_verticesInContours,
					d_verticesInHoles,
					d_modelPositions,
					d_modelRotations,
					d_modelIds);
				cudaDeviceSynchronize();
			}
			noWallsThisType -= NO_THREADS;
			firstWallIndex += NO_THREADS;
		}
		if (info.types[firstWallIndex] == 0)
		{
			generateTenementWallsGPU << <1, NO_THREADS >> > (noWallsThisType,
				d_noModelsInWallsBfr + firstWallIndex,
				d_noVerticesInContoursBfr + firstWallIndex,
				d_noVerticesInHolesBfr,
				d_noHolesInWallsBfr + firstWallIndex,
				d_noArgumentsInWallsBfr + firstWallIndex,
				d_noAssetsInWallsBfr + firstWallIndex,
				d_noCollidersInModelsBfr,
				d_modelsOfPunchers,
				d_noVerticesInPunchersBfr,
				d_cultures,
				d_dimensions + firstWallIndex,
				d_arguments,
				d_assetsInWalls,
				d_colliderVertices,
				d_puncherVertices,
				d_frontMaterials + firstWallIndex,
				d_backMaterials + firstWallIndex,
				d_innerMaterials + firstWallIndex,
				d_outerMaterials + firstWallIndex,
				d_frontMaterialGrains + firstWallIndex,
				d_backMaterialGrains + firstWallIndex,
				d_innerMaterialGrains + firstWallIndex,
				d_outerMaterialGrains + firstWallIndex,
				d_verticesInContours,
				d_verticesInHoles,
				d_modelPositions,
				d_modelRotations,
				d_modelIds);
		}
	}
	cudaDeviceSynchronize();
	cudaMemcpy(*out_noModelsInWallsBfr, d_noModelsInWallsBfr, sizeof(int) * (noWalls + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_noHolesInWallsBfr, d_noHolesInWallsBfr, sizeof(int) * (noWalls + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_noVerticesInContoursBfr, d_noVerticesInContoursBfr, sizeof(int) * (noWalls + 1), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int noHoles = (*out_noHolesInWallsBfr)[noWalls];
	*out_noVerticesInHolesBfr = (int*)malloc(sizeof(int)*(noHoles + 1));
	cudaMemcpy(*out_noVerticesInHolesBfr, d_noVerticesInHolesBfr, sizeof(int) * (noHoles+1), cudaMemcpyDeviceToHost);

	int noContourVertices = (*out_noVerticesInContoursBfr)[noWalls];
	int noHolesVertices = (*out_noVerticesInHolesBfr)[noHoles];
	*out_verticesInContours = (float2*)malloc(sizeof(float2)*(noContourVertices));
	*out_verticesInHoles = (float2*)malloc(sizeof(float2)*(noHolesVertices));
	cudaMemcpy(*out_verticesInContours, d_verticesInContours, sizeof(float2) * noContourVertices, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_verticesInHoles, d_verticesInHoles, sizeof(float2) * noHolesVertices, cudaMemcpyDeviceToHost);

	int noGeneratedModels = (*out_noModelsInWallsBfr)[noWalls];
	*out_modelPositions = (float3*)malloc(sizeof(float3)*(noGeneratedModels));
	*out_modelRotations = (float3*)malloc(sizeof(float3)*(noGeneratedModels));
	*out_modelIds = (int*)malloc(sizeof(int)*(noGeneratedModels));

	cudaMemcpy(*out_modelPositions, d_modelPositions, sizeof(float3) * noGeneratedModels, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_modelRotations, d_modelRotations, sizeof(float3) * noGeneratedModels, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_modelIds, d_modelIds, sizeof(int) * noGeneratedModels, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_frontMaterials, d_frontMaterials, sizeof(int) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_backMaterials, d_backMaterials, sizeof(int) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_innerMaterials, d_innerMaterials, sizeof(int) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_outerMaterials, d_outerMaterials, sizeof(int) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_frontMaterialGrains, d_frontMaterialGrains, sizeof(float) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_backMaterialGrains, d_backMaterialGrains, sizeof(float) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_innerMaterialGrains, d_innerMaterialGrains, sizeof(float) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_outerMaterialGrains, d_outerMaterialGrains, sizeof(float) * noWalls, cudaMemcpyDeviceToHost);

	cudaFree(d_modelsOfPunchers);
	cudaFree(d_noVerticesInPunchersBfr);
	cudaFree(d_cultures);
	cudaFree(d_assetsInWalls);
	cudaFree(d_colliderVertices);
	cudaFree(d_puncherVertices);
	cudaFree(d_modelPositions);
	cudaFree(d_modelRotations);
	cudaFree(d_modelIds);
	cudaFree(d_noAssetsInWallsBfr);
	cudaFree(d_noArgumentsInWallsBfr);
	cudaFree(d_noCollidersInModelsBfr);
	cudaFree(d_noModelsInWallsBfr);
	cudaFree(d_noVerticesInContoursBfr);
	cudaFree(d_noVerticesInHolesBfr);
	cudaFree(d_noHolesInWallsBfr);
	cudaFree(d_frontMaterials);
	cudaFree(d_backMaterials);
	cudaFree(d_innerMaterials);
	cudaFree(d_outerMaterials);
	cudaFree(d_frontMaterialGrains);
	cudaFree(d_backMaterialGrains);
	cudaFree(d_innerMaterialGrains);
	cudaFree(d_outerMaterialGrains);
	cudaFree(d_verticesInContours);
	cudaFree(d_verticesInHoles);
	cudaFree(d_dimensions);
}

