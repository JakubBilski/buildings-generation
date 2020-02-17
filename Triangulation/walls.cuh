#pragma once
#include "defines.h"
#include "typesAndStructs.h"
#include "colliderEngine.cuh"
#include "placementEngine.cuh"
#include "cub/block/block_scan.cuh"

__device__
inline int getTenementWallNoAssets()
{
	return 6;
}

//arguments:
//0 noCondignations
//style:
//0 front material
//1 back material
//2 inside material
//3 outside material
//4 door
//5 window
__global__
void generateTenementWallsGPU(int noWalls, 
	int noModelsInWallsBfr[],
	int noVerticesInContoursBfr[],
	int noVerticesInHolesBfr[],
	int noHolesInWallsBfr[],
	int noArgumentsInWallsBfr[],
	int noAssetsInWallsBfr[],
	int noCollidersInModelsBfr[],
	int modelsOfPunchers[],
	int noVerticesInPunchersBfr[],
	int cultures[],
	float3 dimensions[],
	int arguments[],
	int assetsInWalls[],
	float3 colliderVertices[],
	float2 puncherVertices[],
	int out_frontMaterials[],
	int out_backMaterials[],
	int out_innerMaterials[],
	int out_outerMaterials[],
	float out_frontMaterialGrains[],
	float out_backMaterialGrains[],
	float out_innerMaterialGrains[],
	float out_outerMaterialGrains[],
	float2 out_verticesInContours[],
	float2 out_verticesInHoles[],
	float3 out_modelPositions[],
	float3 out_modelRotations[],
	int out_modelIds[]
)
{
	int wall = threadIdx.x;
	if (wall < noWalls)
	{
		arguments += noArgumentsInWallsBfr[wall];
		assetsInWalls += noAssetsInWallsBfr[wall];
		generateTenementWallStyle(cultures[wall], assetsInWalls);

		float wallLength = dimensions[wall].x;
		float wallHeight = dimensions[wall].y;
		const float2 marginAroundWindows { 0.5f, 0.1f };
		const float2 marginAroundDoor { 1.0f, 0.1f };
		int windowPuncher = assetsInWalls[5];
		int windowPuncherBegin = noVerticesInPunchersBfr[windowPuncher];
		int noVerticesInWindowPuncher = noVerticesInPunchersBfr[windowPuncher + 1] - noVerticesInPunchersBfr[windowPuncher];
		int windowModel = modelsOfPunchers[windowPuncher];
		float3* windowCollider = colliderVertices + noCollidersInModelsBfr[windowModel] * 8;
		int noWindowColliders = noCollidersInModelsBfr[windowModel + 1] - noCollidersInModelsBfr[windowModel];
		float4 windowRectangleSizes = getPivotRectangleSizes(windowCollider, noWindowColliders);
		windowRectangleSizes.x += marginAroundWindows.x;
		windowRectangleSizes.z += marginAroundWindows.x;
		windowRectangleSizes.y += marginAroundWindows.y;
		windowRectangleSizes.w += marginAroundWindows.y;
		float windowWidth = windowRectangleSizes.x + windowRectangleSizes.z;
		float windowHeight = windowRectangleSizes.y + windowRectangleSizes.w;

		int doorPuncher = assetsInWalls[4];
		int doorPuncherBegin = noVerticesInPunchersBfr[doorPuncher];
		int noVerticesInDoorPuncher = noVerticesInPunchersBfr[doorPuncher + 1] - noVerticesInPunchersBfr[doorPuncher];
		int doorModel = modelsOfPunchers[doorPuncher];
		float3* doorCollider = colliderVertices + noCollidersInModelsBfr[doorModel] * 8;
		int noDoorColliders = noCollidersInModelsBfr[doorModel + 1] - noCollidersInModelsBfr[doorModel];
		float4 doorRectangleSizes = getPivotRectangleSizes(doorCollider, noDoorColliders);
		doorRectangleSizes.x += marginAroundDoor.x;
		doorRectangleSizes.z += marginAroundDoor.x;
		doorRectangleSizes.y += marginAroundDoor.y;
		doorRectangleSizes.w += marginAroundDoor.y;
		float doorWidth = doorRectangleSizes.x + doorRectangleSizes.z;
		float doorHeight = doorRectangleSizes.y + doorRectangleSizes.w;

		int noWindowsInCondignation;
		int noWindowsInGroundFloor;
		int noDoors;
		if (doorHeight > wallHeight || doorWidth > wallLength)
		{
			noDoors = 0;
		}
		else
		{
			noDoors = 1;
		}
		if (windowHeight > wallHeight)
		{
			noWindowsInCondignation = 0;
			noWindowsInGroundFloor = 0;
		}
		else
		{
			noWindowsInCondignation = (int)(wallLength / windowWidth);
			noWindowsInGroundFloor = (int)((wallLength - doorWidth * noDoors) / windowWidth);
		}
		int noCondignations = arguments[0];
		int noWindows = noWindowsInCondignation * (noCondignations - 1) + noWindowsInGroundFloor;
		int noHoles = noWindows + noDoors;
		int noHolesVertices = noVerticesInWindowPuncher * noWindows + noVerticesInDoorPuncher * noDoors;

		int noModels = noWindows + noDoors;

		int noContourVertices = 4;
		if (wall == 0)
		{
			noModels += noModelsInWallsBfr[0];
			noContourVertices += noVerticesInContoursBfr[0];
			noHoles += noHolesInWallsBfr[0];
			noHolesVertices += noVerticesInHolesBfr[noHolesInWallsBfr[0]];
		}
		int noModelsBfr;
		int noContourVerticesBfr;
		int noHolesBfr;
		int noHolesVerticesBfr;
		typedef cub::BlockScan<int, NO_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;

		__shared__ typename BlockScan::TempStorage temp_storage1;
		BlockScan(temp_storage1).ExclusiveSum(noContourVertices, noContourVerticesBfr);
		noVerticesInContoursBfr[wall + 1] = noContourVerticesBfr + noContourVertices;

		__shared__ typename BlockScan::TempStorage temp_storage2;
		BlockScan(temp_storage2).ExclusiveSum(noHoles, noHolesBfr);
		noHolesInWallsBfr[wall + 1] = noHolesBfr + noHoles;

		__shared__ typename BlockScan::TempStorage temp_storage3;
		BlockScan(temp_storage3).ExclusiveSum(noHolesVertices, noHolesVerticesBfr);
		if (noHoles > 0)
			noVerticesInHolesBfr[noHolesBfr + noHoles] = noHolesVerticesBfr + noHolesVertices;

		__shared__ typename BlockScan::TempStorage temp_storage4;
		BlockScan(temp_storage4).ExclusiveSum(noModels, noModelsBfr);
		noModelsInWallsBfr[wall + 1] = noModelsBfr + noModels;

		if (wall == 0)
		{
			noModelsBfr += noModelsInWallsBfr[0];
			noContourVerticesBfr += noVerticesInContoursBfr[0];
			noHolesBfr += noHolesInWallsBfr[0];
			noHolesVerticesBfr += noVerticesInHolesBfr[noHolesInWallsBfr[0]];
		}

		noVerticesInHolesBfr += noHolesBfr;

		out_frontMaterialGrains[wall] = 0.5f;
		out_backMaterialGrains[wall] = 0.5f;
		out_innerMaterialGrains[wall] = 0.5f;
		out_outerMaterialGrains[wall] = 0.5f;
		//czy grainsy sa wgl potrzebne? chyba nie, kazdy material powinien miec na stale, albo wgl bez
		out_frontMaterials[wall] = assetsInWalls[0];
		out_backMaterials[wall] = assetsInWalls[1];
		out_innerMaterials[wall] = assetsInWalls[2];
		out_outerMaterials[wall] = assetsInWalls[3];

		out_verticesInContours += noContourVerticesBfr;
		out_verticesInContours[0] = { 0,0 };
		out_verticesInContours[1] = { wallLength,0 };
		out_verticesInContours[2] = { wallLength,wallHeight*noCondignations };
		out_verticesInContours[3] = { 0, wallHeight*noCondignations };

		out_modelPositions += noModelsBfr;
		out_modelRotations += noModelsBfr;
		out_modelIds += noModelsBfr;
		for (int door = 0; door < noDoors; door++)
		{
			out_modelIds[door] = doorModel;
			out_modelPositions[door].y = doorRectangleSizes.w;
			out_modelPositions[door].z = 0;
			out_modelRotations[door] = { 0,0 };
		}
		for (int window = 0; window < noWindowsInGroundFloor; window++)
		{
			int offset = window + noDoors;
			out_modelIds[offset] = windowModel;
			out_modelPositions[offset].y = (wallHeight - windowHeight)*0.5f + windowRectangleSizes.w;
			out_modelPositions[offset].z = 0;
			out_modelRotations[offset] = { 0,0 };
		}
		for (int floor = 1; floor < noCondignations; floor++)
		{
			for (int window = 0; window < noWindowsInCondignation; window++)
			{
				int offset = noDoors + noWindowsInGroundFloor + (floor - 1)*noWindowsInCondignation + window;
				out_modelIds[offset] = windowModel;
				out_modelPositions[offset].y = floor * wallHeight + (wallHeight - windowHeight)*0.5f + windowRectangleSizes.w;
				out_modelPositions[offset].z = 0;
				out_modelRotations[offset] = { 0,0 };
			}
		}
		//TODO: to do this properly, I'm afraid I need shared memory
		//or some define maxModels or sth

		float4 rectangleSizesInNotGroundFloor[40];
		for (int i = 0; i < noWindowsInCondignation; i++)
		{
			rectangleSizesInNotGroundFloor[i] = windowRectangleSizes;
		}
		float4 rectangleSizesInGroundFloor[40];
		for (int door = 0; door < noDoors; door++)
		{
			rectangleSizesInGroundFloor[door] = doorRectangleSizes;
		}
		for (int i = 0; i < noWindowsInGroundFloor; i++)
		{
			rectangleSizesInGroundFloor[i + noDoors] = windowRectangleSizes;
		}
		placeAtSameXIntervals(0, wallLength, noWindowsInGroundFloor + noDoors, rectangleSizesInGroundFloor, out_modelPositions);
		for (int floor = 1; floor < noCondignations; floor++)
		{
			placeAtSameXIntervals(0, wallLength, noWindowsInCondignation, rectangleSizesInNotGroundFloor,
				out_modelPositions + (floor - 1) * noWindowsInCondignation + noWindowsInGroundFloor + noDoors);
		}
		out_verticesInHoles += noHolesVerticesBfr;
		for (int door = 0; door < noDoors; door++)
		{
			noVerticesInHolesBfr[door + 1] = noVerticesInHolesBfr[door] + noVerticesInDoorPuncher;
			for (int v = 0; v < noVerticesInDoorPuncher; v++)
			{
				int iter = door * noVerticesInDoorPuncher + v;
				out_verticesInHoles[iter] = puncherVertices[doorPuncherBegin + v];
				out_verticesInHoles[iter].x += out_modelPositions[door].x;
				out_verticesInHoles[iter].y += out_modelPositions[door].y;
			}
		}
		int noPuncherVerticesInDoors = noDoors * noVerticesInDoorPuncher;
		for (int window = 0; window < noWindows; window++)
		{
			noVerticesInHolesBfr[noDoors + window + 1] = noVerticesInHolesBfr[noDoors + window] + noVerticesInWindowPuncher;
			for (int v = 0; v < noVerticesInWindowPuncher; v++)
			{
				int iter = noPuncherVerticesInDoors + window * noVerticesInWindowPuncher + v;
				out_verticesInHoles[iter] = puncherVertices[windowPuncherBegin + v];
				//the neccessity of calling window positions like this is embarassing
				//probably better would be using modelid
				out_verticesInHoles[iter].x += out_modelPositions[window + noDoors].x;
				out_verticesInHoles[iter].y += out_modelPositions[window + noDoors].y;
			}
		}
	}
}