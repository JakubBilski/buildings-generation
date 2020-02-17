#pragma once
#include <cuda_runtime.h>
#include "defines.h"
#include "typesAndStructs.h"
#include "cultures.cuh"
#include "walls.cuh"
#include "moduleTools.cuh"

__device__
inline int getTenementNoAssets()
{
	return 6;
}

//arguments:
//0 number of condignations
//1 height of a condignation in 0.1f
//style:
//0 front material
//1 back material
//2 inside material
//3 outside material
//4 door
//5 window
__global__
void generateTenementGPU(int noBuildings, 
	int noModelsInBuildingsBfr[],
	int noWallsInBuildingsBfr[],
	int noPlotCornersInBuildingsBfr[],
	int noAssetsInBuildingsBfr[],
	int noAssetsInWallsBfr[],
	float2 plotCorners[],
	int noArgumentsInBuildingsBfr[],
	int argumentValues[],
	int cultures[],
	int assetsInStyles[],
	int noCollidersInModelsBfr[],
	float3 colliderVertices[],
	float3 out_modelPositions[],
	float3 out_modelRotations[],
	int out_modelIds[],
	float3 out_wallPositions[],
	float3 out_wallRotations[],
	float3 out_wallDimensions[], 
	int out_wallTypes[],
	int out_wallAssets[],
	int noArgumentsInWallsBfr[],
	int out_wallArguments[],
	int out_wallCultures[])
{
	int building = threadIdx.x;
	if (building < noBuildings)
	{
		//cuda-specific preparations
		int plotCornersBegin = noPlotCornersInBuildingsBfr[building];
		int noPlots = noPlotCornersInBuildingsBfr[building + 1] - plotCornersBegin;

		int noWalls = noPlots;
		int noModels = 0;

		int noWallsBfr = bookBlockSpace(building, noWalls, noWallsInBuildingsBfr);
		int noModelsBfr = bookBlockSpace(building, noModels, noModelsInBuildingsBfr);

		plotCorners += plotCornersBegin;
		out_wallPositions += noWallsBfr;
		out_wallRotations += noWallsBfr;
		out_wallDimensions += noWallsBfr;
		out_wallTypes += noWallsBfr;
		out_wallCultures += noWallsBfr;
		out_modelPositions += noModelsBfr;
		out_modelRotations += noModelsBfr;
		out_modelIds += noModelsBfr;
		assetsInStyles += noAssetsInBuildingsBfr[building];

		generateTenementStyle(cultures[building], assetsInStyles);
		int noCondignations = argumentValues[0];
		float condignationHeight = argumentValues[1] * 0.1f;

		const float wallWidth = 0.4f;

		for (int i = 0; i < noWalls; i++)
		{
			//using out_wallTypes as temporary container
			out_wallTypes[i] = getTenementWallNoAssets();
		}
		bookAssetsSpace(building, noWallsBfr, noWalls, out_wallTypes, noAssetsInWallsBfr);
		for (int i = 0; i < noWalls; i++)
		{
			//using out_wallTypes as temporary container
			out_wallTypes[i] = 2;
		}
		bookAssetsSpace(building, noWallsBfr, noWalls, out_wallTypes, noArgumentsInWallsBfr);
		noAssetsInWallsBfr += noWallsInBuildingsBfr[building];
		noArgumentsInWallsBfr += noWallsInBuildingsBfr[building];
		//the actual generation
		for (int wall = 0; wall < noWalls; wall++)
		{
			float2 corner = plotCorners[wall];
			float2 nextCorner;
			if (wall == noWalls - 1)
			{
				nextCorner = plotCorners[0];
			}
			else
			{
				nextCorner = plotCorners[wall + 1];
			}
			float diffx = nextCorner.x - corner.x;
			float diffy = nextCorner.y - corner.y;
			float length = sqrt(diffx * diffx + diffy * diffy);
			out_wallPositions[wall] = { corner.x, 0, corner.y };
			out_wallRotations[wall] = { 0,-atan2(diffy, diffx),0 };
			out_wallDimensions[wall] = { length,  condignationHeight, wallWidth };
			//using out_wallTypes according to its puropse
			out_wallTypes[wall] = 0;
			out_wallAssets[noAssetsInWallsBfr[wall]] = assetsInStyles[0];
			out_wallAssets[noAssetsInWallsBfr[wall] + 1] = assetsInStyles[1];
			out_wallAssets[noAssetsInWallsBfr[wall] + 2] = assetsInStyles[2];
			out_wallAssets[noAssetsInWallsBfr[wall] + 3] = assetsInStyles[3];
			out_wallAssets[noAssetsInWallsBfr[wall] + 4] = assetsInStyles[4];
			out_wallAssets[noAssetsInWallsBfr[wall] + 5] = assetsInStyles[5];
			out_wallArguments[noArgumentsInWallsBfr[wall]] = noCondignations;
			out_wallArguments[noArgumentsInWallsBfr[wall] + 1] = 69;
			out_wallCultures[wall] = cultures[building];
		}
	}
}
