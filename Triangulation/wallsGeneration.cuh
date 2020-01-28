#pragma once
#include <cuda_runtime.h>
#include "defines.h"
#include "typesAndStructs.h"
#include <string>
#include "cub/block/block_scan.cuh"



__global__
void generateBlueWallsGPU(int noWalls, int noModelsInWallsBfr[], int noVerticesInContoursBfr[], int noVerticesInHolesBfr[], int noHolesInWallsBfr[],
	int out_frontMaterials[], int out_backMaterials[], int out_innerMaterials[], int out_outerMaterials[],
	float out_frontMaterialGrains[], float out_backMaterialGrains[], float out_innerMaterialGrains[], float out_outerMaterialGrains[],
	float2 out_verticesInContours[], float2 out_verticesInHoles[], float3 vectorXs[], float3 vectorYs[]
	)
{
	int wall = threadIdx.x;
	if (wall < noWalls)
	{
		//TODO: scan no models, add models etc. 
		noModelsInWallsBfr[wall] = 0;

		int noContourVertices = 4;
		int noHoles = 0;
		int noHolesVertices = 0;
		float wallLength = sqrt(vectorXs[wall].x *  vectorXs[wall].x + vectorXs[wall].z *  vectorXs[wall].z);
		float wallHeight = vectorYs[wall].y;
		const int windowLength = 2.0f;
		const int windowHeight = 3.0f;

		if (wallLength > windowLength*1.1f && wallHeight > windowHeight*1.1f);
		{
			noHoles = 1;
			noHolesVertices = 4;
		}
		if (wall == 0)
		{
			noContourVertices += noVerticesInContoursBfr[0];
			noHoles += noHolesInWallsBfr[0];
			noHolesVertices += noVerticesInHolesBfr[noHolesInWallsBfr[0]];
		}
		__syncthreads();
		int noContourVerticesBfr;
		int noHolesBfr;
		int noHolesVerticesBfr;
		typedef cub::BlockScan<int, NO_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;

		__shared__ typename BlockScan::TempStorage temp_storage1;
		BlockScan(temp_storage1).ExclusiveSum(noContourVertices, noContourVerticesBfr);
		noVerticesInContoursBfr[wall + 1] = noContourVerticesBfr + noContourVertices;
		__syncthreads();

		__shared__ typename BlockScan::TempStorage temp_storage2;
		BlockScan(temp_storage2).ExclusiveSum(noHoles, noHolesBfr);
		noHolesInWallsBfr[wall + 1] = noHolesBfr + noHoles;
		__syncthreads();

		__shared__ typename BlockScan::TempStorage temp_storage3;
		BlockScan(temp_storage3).ExclusiveSum(noHolesVertices, noHolesVerticesBfr);
		if(noHoles > 0)
			noVerticesInHolesBfr[noHolesBfr + noHoles] = noHolesVerticesBfr + noHolesVertices;
		__syncthreads();
		if (wall == 0)
		{
			noContourVerticesBfr += noVerticesInContoursBfr[0];
			noHolesBfr += noHolesInWallsBfr[0];
			noHolesVerticesBfr += noVerticesInHolesBfr[noHolesInWallsBfr[0]];
		}

		out_frontMaterialGrains[wall] = 0.5f;
		out_backMaterialGrains[wall] = 0.5f;
		out_innerMaterialGrains[wall] = 0.5f;
		out_outerMaterialGrains[wall] = 0.5f;
		out_frontMaterials[wall] = 0;
		out_backMaterials[wall] = 2;
		out_innerMaterials[wall] = 2;
		out_outerMaterials[wall] = 2;
		out_verticesInContours[noContourVerticesBfr] = { 0,0 };
		out_verticesInContours[noContourVerticesBfr + 3] = { 0,1 };
		out_verticesInContours[noContourVerticesBfr + 2] = { 1,1 };
		out_verticesInContours[noContourVerticesBfr + 1] = { 1,0 };

		if (noHoles > 0)
		{
			out_verticesInHoles[noHolesVerticesBfr] = { 0.3f, 0.2f };
			out_verticesInHoles[noHolesVerticesBfr + 1] = { 0.3f, 0.8f };
			out_verticesInHoles[noHolesVerticesBfr + 2] = { 0.7f, 0.8f };
			out_verticesInHoles[noHolesVerticesBfr + 3] = { 0.7f, 0.2f };
		}
		__syncthreads();
	}
}
__global__
void generateRedWallsGPU(int noWalls, int noModelsInWallsBfr[], int noVerticesInContoursBfr[], int noVerticesInHolesBfr[], int noHolesInWallsBfr[],
	int out_frontMaterials[], int out_backMaterials[], int out_innerMaterials[], int out_outerMaterials[],
	float out_frontMaterialGrains[], float out_backMaterialGrains[], float out_innerMaterialGrains[], float out_outerMaterialGrains[],
	float2 out_verticesInContours[], float2 out_verticesInHoles[], float3 vectorXs[], float3 vectorYs[]
	)
{
	int wall = threadIdx.x;
	if (wall < noWalls)
	{
		//TODO: scan no models, add models etc. 
		noModelsInWallsBfr[wall] = 0;

		int noContourVertices = 4;
		int noHoles = 0;
		int noHolesVertices = 0;
		float wallLength = sqrt(vectorXs[wall].x *  vectorXs[wall].x + vectorXs[wall].z *  vectorXs[wall].z);
		float wallHeight = vectorYs[wall].y;
		const int windowLength = 2.0f;
		const int windowHeight = 3.0f;

		if (wallLength > windowLength*1.1f && wallHeight > windowHeight*1.1f);
		{
			noHoles = 1;
			noHolesVertices = 4;
		}
		if (wall == 0)
		{
			noContourVertices += noVerticesInContoursBfr[0];
			noHoles += noHolesInWallsBfr[0];
			noHolesVertices += noVerticesInHolesBfr[noHolesInWallsBfr[0]];
		}
		__syncthreads();
		int noContourVerticesBfr;
		int noHolesBfr;
		int noHolesVerticesBfr;
		typedef cub::BlockScan<int, NO_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;

		__shared__ typename BlockScan::TempStorage temp_storage1;
		BlockScan(temp_storage1).ExclusiveSum(noContourVertices, noContourVerticesBfr);
		noVerticesInContoursBfr[wall + 1] = noContourVerticesBfr + noContourVertices;
		__syncthreads();

		__shared__ typename BlockScan::TempStorage temp_storage2;
		BlockScan(temp_storage2).ExclusiveSum(noHoles, noHolesBfr);
		noHolesInWallsBfr[wall + 1] = noHolesBfr + noHoles;
		__syncthreads();

		__shared__ typename BlockScan::TempStorage temp_storage3;
		BlockScan(temp_storage3).ExclusiveSum(noHolesVertices, noHolesVerticesBfr);
		if(noHoles > 0)
			noVerticesInHolesBfr[noHolesBfr + noHoles] = noHolesVerticesBfr + noHolesVertices;
		__syncthreads();
		if (wall == 0)
		{
			noContourVerticesBfr += noVerticesInContoursBfr[0];
			noHolesBfr += noHolesInWallsBfr[0];
			noHolesVerticesBfr += noVerticesInHolesBfr[noHolesInWallsBfr[0]];
		}

		out_frontMaterialGrains[wall] = 0.5f;
		out_backMaterialGrains[wall] = 0.5f;
		out_innerMaterialGrains[wall] = 0.5f;
		out_outerMaterialGrains[wall] = 0.5f;
		out_frontMaterials[wall] = 1;
		out_backMaterials[wall] = 2;
		out_innerMaterials[wall] = 2;
		out_outerMaterials[wall] = 2;
		out_verticesInContours[noContourVerticesBfr] = { 0,0 };
		out_verticesInContours[noContourVerticesBfr + 3] = { 0,1 };
		out_verticesInContours[noContourVerticesBfr + 2] = { 1,1 };
		out_verticesInContours[noContourVerticesBfr + 1] = { 1,0 };

		if (noHoles > 0)
		{
			out_verticesInHoles[noHolesVerticesBfr] = { 0.3f, 0.2f };
			out_verticesInHoles[noHolesVerticesBfr + 1] = { 0.3f, 0.8f };
			out_verticesInHoles[noHolesVerticesBfr + 2] = { 0.7f, 0.8f };
			out_verticesInHoles[noHolesVerticesBfr + 3] = { 0.7f, 0.2f };
		}
		__syncthreads();
	}
}


void generateWalls(int noWalls, wallsInfo info, int noTypes, int noWallsInTypesBfr[],
	int* out_noModelsInWallsBfr[], int* out_noVerticesInContoursBfr[],int* out_noVerticesInHolesBfr[], int* out_noHolesInWallsBfr[],
	int* out_frontMaterials[], int* out_backMaterials[], int* out_innerMaterials[], int* out_outerMaterials[],
	float* out_frontMaterialGrains[],float* out_backMaterialGrains[],float* out_innerMaterialGrains[],float* out_outerMaterialGrains[],
	float2* out_verticesInContours[], float2* out_verticesInHoles[])
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
	int maxHoles = noWalls * 3;
	int maxContourVertices = 6 * noWalls;
	int maxHolesVertices = maxHoles * 6;

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

	cudaMalloc(&d_noModelsInWallsBfr, sizeof(int) * (noWalls + 1));
	cudaMemset(d_noModelsInWallsBfr, 0, sizeof(int) * (noWalls + 1));
	cudaMalloc(&d_noVerticesInContoursBfr, sizeof(int) * (noWalls + 1));
	cudaMemset(d_noVerticesInContoursBfr, 0, sizeof(int) * (noWalls + 1));
	cudaMalloc(&d_noHolesInWallsBfr, sizeof(int) * (noWalls + 1));
	cudaMemset(d_noHolesInWallsBfr, 0, sizeof(int) * (noWalls + 1));
	cudaMalloc(&d_noVerticesInHolesBfr, sizeof(int) * (maxHoles + 1));
	cudaMemset(d_noVerticesInHolesBfr, 0, sizeof(int) * (maxHoles + 1));
	cudaMalloc(&d_frontMaterials, sizeof(int) * (noWalls));
	cudaMalloc(&d_backMaterials, sizeof(int) * (noWalls));
	cudaMalloc(&d_innerMaterials, sizeof(int) * (noWalls));
	cudaMalloc(&d_outerMaterials, sizeof(int) * (noWalls));
	cudaMalloc(&d_frontMaterialGrains, sizeof(float) * (noWalls));
	cudaMalloc(&d_backMaterialGrains, sizeof(float) * (noWalls));
	cudaMalloc(&d_innerMaterialGrains, sizeof(float) * (noWalls));
	cudaMalloc(&d_outerMaterialGrains, sizeof(float) * (noWalls));
	cudaMalloc(&d_verticesInContours, sizeof(float2) * (maxContourVertices));
	cudaMalloc(&d_verticesInHoles, sizeof(float2) * (maxHolesVertices));

	float3* d_vectorXs;
	float3* d_vectorYs;

	cudaMalloc(&d_vectorXs, sizeof(float3) * (noWalls));
	cudaMalloc(&d_vectorYs, sizeof(float3) * (noWalls));

	cudaMemcpy(d_vectorXs, info.vectorXs, sizeof(float3) * noWalls, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vectorYs, info.vectorYs, sizeof(float3) * noWalls, cudaMemcpyHostToDevice);

	for (int typeIndex = 0; typeIndex < noTypes; typeIndex++)
	{
		int firstWallIndex = noWallsInTypesBfr[typeIndex];
		int noWallsThisType = noWallsInTypesBfr[typeIndex + 1] - firstWallIndex;
		while (noWallsThisType > NO_THREADS)
		{
			if (info.types[firstWallIndex] == WallType::BLUE_W)
			{
				generateBlueWallsGPU << <1, NO_THREADS >> > (NO_THREADS, d_noModelsInWallsBfr + firstWallIndex, d_noVerticesInContoursBfr + firstWallIndex,
					d_noVerticesInHolesBfr, d_noHolesInWallsBfr + firstWallIndex,
					d_frontMaterials + firstWallIndex, d_backMaterials + firstWallIndex, d_innerMaterials + firstWallIndex, d_outerMaterials + firstWallIndex,
					d_frontMaterialGrains + firstWallIndex, d_backMaterialGrains + firstWallIndex, d_innerMaterialGrains + firstWallIndex, d_outerMaterialGrains + firstWallIndex,
					d_verticesInContours, d_verticesInHoles, d_vectorXs + firstWallIndex, d_vectorYs + firstWallIndex);
				cudaDeviceSynchronize();
			}
			else if (info.types[firstWallIndex] == WallType::RED_W)
			{
				generateRedWallsGPU << <1, NO_THREADS >> > (NO_THREADS, d_noModelsInWallsBfr + firstWallIndex, d_noVerticesInContoursBfr + firstWallIndex,
					d_noVerticesInHolesBfr, d_noHolesInWallsBfr + firstWallIndex,
					d_frontMaterials + firstWallIndex, d_backMaterials + firstWallIndex, d_innerMaterials + firstWallIndex, d_outerMaterials + firstWallIndex,
					d_frontMaterialGrains + firstWallIndex, d_backMaterialGrains + firstWallIndex, d_innerMaterialGrains + firstWallIndex, d_outerMaterialGrains + firstWallIndex,
					d_verticesInContours, d_verticesInHoles, d_vectorXs + firstWallIndex, d_vectorYs + firstWallIndex);
				cudaDeviceSynchronize();
			}

			noWallsThisType -= NO_THREADS;
			firstWallIndex += NO_THREADS;
		}
		if (info.types[firstWallIndex] == WallType::BLUE_W)
		{
			generateBlueWallsGPU << <1, NO_THREADS >> > (noWallsThisType, d_noModelsInWallsBfr + firstWallIndex, d_noVerticesInContoursBfr + firstWallIndex,
				d_noVerticesInHolesBfr, d_noHolesInWallsBfr + firstWallIndex,
				d_frontMaterials + firstWallIndex, d_backMaterials + firstWallIndex, d_innerMaterials + firstWallIndex, d_outerMaterials + firstWallIndex,
				d_frontMaterialGrains + firstWallIndex, d_backMaterialGrains + firstWallIndex, d_innerMaterialGrains + firstWallIndex, d_outerMaterialGrains + firstWallIndex,
				d_verticesInContours, d_verticesInHoles, d_vectorXs + firstWallIndex, d_vectorYs + firstWallIndex);
		}
		else if (info.types[firstWallIndex] == WallType::RED_W)
		{
			generateRedWallsGPU << <1, NO_THREADS >> > (noWallsThisType, d_noModelsInWallsBfr + firstWallIndex, d_noVerticesInContoursBfr + firstWallIndex,
				d_noVerticesInHolesBfr, d_noHolesInWallsBfr + firstWallIndex,
				d_frontMaterials + firstWallIndex, d_backMaterials + firstWallIndex, d_innerMaterials + firstWallIndex, d_outerMaterials + firstWallIndex,
				d_frontMaterialGrains + firstWallIndex, d_backMaterialGrains + firstWallIndex, d_innerMaterialGrains + firstWallIndex, d_outerMaterialGrains + firstWallIndex,
				d_verticesInContours, d_verticesInHoles, d_vectorXs + firstWallIndex, d_vectorYs + firstWallIndex);
		}
	}
	cudaMemcpy(*out_noModelsInWallsBfr, d_noModelsInWallsBfr, sizeof(int) * (noWalls + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_noHolesInWallsBfr, d_noHolesInWallsBfr, sizeof(int) * (noWalls + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_noVerticesInContoursBfr, d_noVerticesInContoursBfr, sizeof(int) * (noWalls + 1), cudaMemcpyDeviceToHost);

	int noHoles = (*out_noHolesInWallsBfr)[noWalls];
	*out_noVerticesInHolesBfr = (int*)malloc(sizeof(int)*(noHoles + 1));
	cudaMemcpy(*out_noVerticesInHolesBfr, d_noVerticesInHolesBfr, sizeof(int) * (noHoles+1), cudaMemcpyDeviceToHost);

	int noContourVertices = (*out_noVerticesInContoursBfr)[noWalls];
	int noHolesVertices = (*out_noVerticesInHolesBfr)[noHoles];
	*out_verticesInContours = (float2*)malloc(sizeof(float2)*(noContourVertices));
	*out_verticesInHoles = (float2*)malloc(sizeof(float2)*(noHolesVertices));
	cudaMemcpy(*out_verticesInContours, d_verticesInContours, sizeof(float2) * noContourVertices, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_verticesInHoles, d_verticesInHoles, sizeof(float2) * noHolesVertices, cudaMemcpyDeviceToHost);

	cudaMemcpy(*out_frontMaterials, d_frontMaterials, sizeof(int) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_backMaterials, d_backMaterials, sizeof(int) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_innerMaterials, d_innerMaterials, sizeof(int) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_outerMaterials, d_outerMaterials, sizeof(int) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_frontMaterialGrains, d_frontMaterialGrains, sizeof(float) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_backMaterialGrains, d_backMaterialGrains, sizeof(float) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_innerMaterialGrains, d_innerMaterialGrains, sizeof(float) * noWalls, cudaMemcpyDeviceToHost);
	cudaMemcpy(*out_outerMaterialGrains, d_outerMaterialGrains, sizeof(float) * noWalls, cudaMemcpyDeviceToHost);

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
	cudaFree(d_vectorXs);
	cudaFree(d_vectorYs);
}

