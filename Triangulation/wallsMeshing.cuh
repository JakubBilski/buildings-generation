#pragma once
#include <cuda_runtime.h>
#include "defines.h"
#include "holing.cuh"
#include "triangulation.cuh"
#include "uvsFinding.cuh"
#include "normalsFinding.cuh"

struct generateWallsArgs
{
	int noWalls;
	float* frontMaterialGrains;
	float* backMaterialGrains;
	float* innerMaterialGrains;
	float* outerMaterialGrains;

	float3* wallPositions;
	float3* wallRotations;
	float3* wallDimensions;
	int* noVerticesInContoursBfr;
	int* noHolesInWallsBfr;
	int* noVerticesInHolesBfr;
	float2* verticesInContours;
	float2* verticesInHoles;
};

struct generateWallsResult
{
	int* noVerticesInWallsBfr;
	int* triangles;
	float2* allVerticesValues;
	float2* holesVerticesNormalsValues;
	float2* contourNormalsValues;
	float2* frontUvs;
	float2* backUvs;
	float2* innerUvs;
	float2* outerUvs;
};

void step_mergeHolesAndContours(int noWalls, int noVerticesInContoursBfr[], int noHolesInWallsBfr[], int noVerticesInHolesBfr[], float2 verticesInContours[], float2 verticesInHoles[],
	float2* out_holesAndContours[], int* out_noVerticesInWallsBfr[], int* out_noWallsInBlocksBfr[], int* out_noBlocks)
{
	*out_holesAndContours = (float2*)malloc(sizeof(float2)*(noVerticesInContoursBfr[noWalls] + noVerticesInHolesBfr[noHolesInWallsBfr[noWalls]] + 2 * noHolesInWallsBfr[noWalls]));
	*out_noVerticesInWallsBfr = (int*)malloc(sizeof(int)*(noWalls + 1));
	mergeHolesAndContoursCPU(
		noWalls,
		noVerticesInContoursBfr,
		noHolesInWallsBfr,
		noVerticesInHolesBfr,
		verticesInContours,
		verticesInHoles,
		*out_holesAndContours,
		*out_noVerticesInWallsBfr,
		out_noWallsInBlocksBfr,
		out_noBlocks
	);
}

void step_triangulateFronts(int noWalls, int noBlocks, int noAllVertices, int* noWallsInBlocksBfr, int* d_noVerticesInWallsBfr, float2* d_verticesInWalls,
	int* out_d_triangles[], int* out_noTriangles)
{
	int* d_noWallsInBlocksBfr;
#ifdef DEBUG
	printf("No walls handled by blocks:\n");
	for (int block = 0; block < noBlocks; block++)
	{
		printf("\tBlock %d: %d walls\n", block, noWallsInBlocksBfr[block + 1] - noWallsInBlocksBfr[block]);
	}
#endif
	cudaMalloc(&d_noWallsInBlocksBfr, sizeof(int)*(noBlocks + 1));
	cudaMemcpy(d_noWallsInBlocksBfr, noWallsInBlocksBfr, sizeof(int)*(noBlocks + 1), cudaMemcpyHostToDevice);
	int maxWallsForBlock = NO_VERTICES_FOR_BLOCK_TRIANGULATION / 4;	//very weak heuristics, better would be some max
	int sizeOfSharedMemoryPerBlock = sizeof(float2) * NO_VERTICES_FOR_BLOCK_TRIANGULATION + sizeof(int) * NO_VERTICES_FOR_BLOCK_TRIANGULATION * 7 + sizeof(int) * maxWallsForBlock * 2;
	*out_noTriangles = 3 * (noAllVertices - 2 * noWalls);
	cudaMalloc(out_d_triangles, sizeof(int) * (*out_noTriangles));
	triangulatePolygonGPU << <noBlocks, NO_THREADS, sizeOfSharedMemoryPerBlock >> > (d_noVerticesInWallsBfr, d_noWallsInBlocksBfr, d_verticesInWalls, NO_VERTICES_FOR_BLOCK_TRIANGULATION, *out_d_triangles);
	cudaDeviceSynchronize();
	cudaFree(d_noWallsInBlocksBfr);
}

void step_findUvs(int noWalls, int noVertices, int d_noVerticesInWallsBfr[], float2 d_verticesValues[], float d_grains[], float2* out_uvs[])
{
	*out_uvs = (float2*)malloc(sizeof(float2)*noVertices);
	float2* d_uvs;
	cudaMalloc(&d_uvs, sizeof(float2)*noVertices);
	int noBlocks = (noWalls - 1) / NO_THREADS + 1;
	findUvsGPU << <noBlocks, NO_THREADS >> > (noWalls, d_noVerticesInWallsBfr, d_verticesValues, d_grains, d_uvs);
	cudaMemcpy(*out_uvs, d_uvs, sizeof(float2)*noVertices, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_uvs);
}

void step_findNormals(int noHoles, int noVerticesInHoles, int d_noVerticesInHolesBfr[], float2 d_verticesInHoles[],
	float2* out_normalsInside[])
{
	*out_normalsInside = (float2*)malloc(sizeof(float2) * noVerticesInHoles);
	float2* d_normalsInside;
	cudaMalloc(&d_normalsInside, sizeof(float2) * noVerticesInHoles);
	findInsideNormalsGPU << <1, NO_THREADS >> > (noHoles, noVerticesInHoles, d_noVerticesInHolesBfr, d_verticesInHoles, d_normalsInside);
	cudaMemcpy(*out_normalsInside, d_normalsInside, sizeof(float2) * noVerticesInHoles, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_normalsInside);
}

generateWallsResult meshWalls(generateWallsArgs args)
{
	float2* verticesValues;
	int* noVerticesInWallsBfr;
	int* noWallsInBlocksBfr;
	int noBlocks;

	step_mergeHolesAndContours(args.noWalls, args.noVerticesInContoursBfr, args.noHolesInWallsBfr, args.noVerticesInHolesBfr, args.verticesInContours, args.verticesInHoles,
		&verticesValues, &noVerticesInWallsBfr, &noWallsInBlocksBfr, &noBlocks);

	int* d_triangles;
	int noTriangles;
	int* d_noVerticesInWallsBfr;
	cudaMalloc(&d_noVerticesInWallsBfr, sizeof(int)*(args.noWalls + 1));
	cudaMemcpy(d_noVerticesInWallsBfr, noVerticesInWallsBfr, sizeof(int)*(args.noWalls + 1), cudaMemcpyHostToDevice);
	int noAllVertices = noVerticesInWallsBfr[args.noWalls];

	float2* d_verticesInWalls;
	cudaMalloc(&d_verticesInWalls, sizeof(float2)*(noAllVertices));
	cudaMemcpy(d_verticesInWalls, verticesValues, sizeof(float2)*(noAllVertices), cudaMemcpyHostToDevice);

	float2* frontUvs;
	float2* backUvs;
	float* d_frontMaterialGrains;
	float* d_backMaterialGrains;

	cudaMalloc(&d_frontMaterialGrains, sizeof(float)*args.noWalls);
	cudaMalloc(&d_backMaterialGrains, sizeof(float)*args.noWalls);
	cudaMemcpy(d_frontMaterialGrains, args.frontMaterialGrains, sizeof(float)*args.noWalls, cudaMemcpyHostToDevice);
	cudaMemcpy(d_backMaterialGrains, args.backMaterialGrains, sizeof(float)*args.noWalls, cudaMemcpyHostToDevice);
#ifdef DEBUG
	printf("Front grains:\n");
	for (int i = 0; i < args.noWalls; i++)
	{
		printf("%f ", args.frontMaterialGrains[i]);
	}
	printf("\n");
#endif

	step_findUvs(args.noWalls, noAllVertices, d_noVerticesInWallsBfr, d_verticesInWalls, d_frontMaterialGrains, &frontUvs);
	step_findUvs(args.noWalls, noAllVertices, d_noVerticesInWallsBfr, d_verticesInWalls, d_backMaterialGrains, &backUvs);

	cudaFree(d_frontMaterialGrains);
	cudaFree(d_backMaterialGrains);
#ifdef DEBUG
	printf("Front uvs:\n");
	for (int i = 0; i < noAllVertices; i++)
	{
		printf("%f %f\n", frontUvs[i].x, frontUvs[i].y);
	}
#endif

#ifdef DEBUG
	printf("Information before triangulation:\n");
	for (int wall = 0; wall < args.noWalls; wall++)
	{
		printf("Wall %d\n", wall);
		printf("\tNo holes: %d\n", args.noHolesInWallsBfr[wall + 1] - args.noHolesInWallsBfr[wall]);
		printf("\tContour length: %d\n", args.noVerticesInContoursBfr[wall + 1] - args.noVerticesInContoursBfr[wall]);
		printf("\tSummarized length: %d\n", noVerticesInWallsBfr[wall + 1] - noVerticesInWallsBfr[wall]);
	}
#endif

	step_triangulateFronts(args.noWalls, noBlocks, noAllVertices, noWallsInBlocksBfr, d_noVerticesInWallsBfr, d_verticesInWalls, &d_triangles, &noTriangles);

	int* triangles = (int*)malloc(sizeof(int) * noTriangles);
	cudaMemcpy(triangles, d_triangles, sizeof(int) * noTriangles, cudaMemcpyDeviceToHost);
	cudaFree(d_triangles);


#ifdef DEBUG
	printf("Triangles\n");
	for (int i = 0; i < noTriangles; i+=3)
	{
		printf("triangle %d: %d %d %d\n", i/3, triangles[i], triangles[i + 1], triangles[i + 2]);
	}
#endif

	cudaFree(d_noVerticesInWallsBfr);
	cudaFree(d_verticesInWalls);

	int noHoles = args.noHolesInWallsBfr[args.noWalls];
	int noVerticesInHoles = args.noVerticesInHolesBfr[noHoles];
	int noVerticesInContours = args.noVerticesInContoursBfr[args.noWalls];

	float2* innerUvs = (float2*)malloc(sizeof(float2)*noVerticesInHoles);
	float2* outerUvs = (float2*)malloc(sizeof(float2)*noVerticesInContours);
	for (int i = 0; i < noVerticesInHoles; i++)
	{
		innerUvs[i] = { 0,0 };
	}
	for (int i = 0; i < noVerticesInContours; i++)
	{
		outerUvs[i] = { 0,0 };
	}
	cudaDeviceSynchronize();

	int* d_noVerticesInContoursBfr;
	float2* normalsOutside;
	float2* d_verticesInContours;
	cudaMalloc(&d_verticesInContours, sizeof(float2)*noVerticesInContours);
	cudaMemcpy(d_verticesInContours, args.verticesInContours, sizeof(float2)*noVerticesInContours, cudaMemcpyHostToDevice);
	cudaMalloc(&d_noVerticesInContoursBfr, sizeof(float2)*args.noWalls);
	cudaMemcpy(d_noVerticesInContoursBfr, args.noVerticesInContoursBfr, sizeof(float2)*args.noWalls, cudaMemcpyHostToDevice);
	//TODO: change findNormalsInside to findNormals, as inner and outer normals are the only ones we have to find
	step_findNormals(args.noWalls, noVerticesInContours, d_noVerticesInContoursBfr, d_verticesInContours, &normalsOutside);
	cudaFree(d_verticesInContours);
	cudaFree(d_noVerticesInContoursBfr);

	cudaDeviceSynchronize();

	int* d_noVerticesInHolesBfr;
	float2* normalsInside;
	float2* d_verticesInHoles;
	cudaMalloc(&d_noVerticesInHolesBfr, sizeof(float2)*noHoles);
	cudaMemcpy(d_noVerticesInHolesBfr, args.noVerticesInHolesBfr, sizeof(float2)*noHoles, cudaMemcpyHostToDevice);
	cudaMalloc(&d_verticesInHoles, sizeof(float2)*noVerticesInHoles);
	cudaMemcpy(d_verticesInHoles, args.verticesInHoles, sizeof(float2)*noVerticesInHoles, cudaMemcpyHostToDevice);
	step_findNormals(noHoles, noVerticesInHoles, d_noVerticesInHolesBfr, d_verticesInHoles, &normalsInside);
	cudaFree(d_verticesInHoles);
	cudaFree(d_noVerticesInHolesBfr);

	generateWallsResult result;
	result.noVerticesInWallsBfr = noVerticesInWallsBfr;
	result.triangles = triangles;
	result.allVerticesValues = verticesValues;
	result.holesVerticesNormalsValues = normalsInside;
	result.contourNormalsValues = normalsOutside;
	result.frontUvs = frontUvs;
	result.backUvs = backUvs;
	result.innerUvs = innerUvs;
	result.outerUvs = outerUvs;
	return result;
}