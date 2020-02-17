#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <map>

#include "meshing.cuh"
#include "modulesGeneration.cuh"
#include "wallsGeneration.cuh"
#include "jsonOperations.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <cuda_runtime.h>



int main()
{
	//input
	const int noBuildings = 100;
	//TODO: foreach stage
	printf("Start\n");
	float3* buildingPositions = (float3*)malloc(sizeof(float3)*noBuildings);
	for (int i = 0; i < noBuildings; i++)
	{
		buildingPositions[i] = { 0, 0, (float)(i * 20) };
	}
	const int noTypes = 1;
	int types[noTypes]{ 0 };
	int noBuildingsInTypesBfr[noTypes + 1]{ 0,noBuildings };
	int noPlotCornersInBuildingsBfr[noBuildings + 1];
	noPlotCornersInBuildingsBfr[0] = 0;
	const int noPlotCorners = 4;
	for (int i = 1; i < noBuildings + 1; i++)
	{
		noPlotCornersInBuildingsBfr[i] = noPlotCornersInBuildingsBfr[i - 1] + noPlotCorners;
	}
	float2 plotCorners[noBuildings * noPlotCorners];
	for (int i = 0; i < noBuildings; i++)
	{
		plotCorners[noPlotCorners * i] = { 0,0 };
		plotCorners[noPlotCorners * i + 1] = { 10,0 };
		plotCorners[noPlotCorners * i + 2] = { 10,10 };
		plotCorners[noPlotCorners * i + 3] = { 0,10 };
	}
	//for (int i = 0; i < noBuildings; i++)
	//{
	//	plotCorners[noPlotCorners * i] = { 0,0 };
	//	plotCorners[noPlotCorners * i + 1] = { 5,0 };
	//	plotCorners[noPlotCorners * i + 2] = { 10,5 };
	//	plotCorners[noPlotCorners * i + 3] = { 5,10 };
	//	plotCorners[noPlotCorners * i + 4] = { 0,5 };
	//}
	int noArgumentsInBuildingsBfr[noBuildings + 1];
	int argumentsInBuildings[noBuildings * 2];
	int cultures[noBuildings]{ 0 };
	int noAssetsInBuildingsBfr[noBuildings + 1];
	int assetsInBuildings[noBuildings * 6];
	noArgumentsInBuildingsBfr[0] = 0;
	for (int i = 1; i < noBuildings + 1; i++)
	{
		noArgumentsInBuildingsBfr[i] = noArgumentsInBuildingsBfr[i - 1] + 2;
		argumentsInBuildings[2 * i - 2] = 2;
		argumentsInBuildings[2 * i - 1] = 40;
	}
	noAssetsInBuildingsBfr[0] = 0;
	for (int i = 0; i < noBuildings; i++)
	{
		noAssetsInBuildingsBfr[i + 1] = noAssetsInBuildingsBfr[i] + 6;
		for (int j = 0; j < 6; j++)
		{
			assetsInBuildings[6 * i + j] = 0;
		}
	}
	const int noModels = 2;
	int noCollidersInModelsBfr[noModels + 2]{ 0, 0, 1, 2 };
	float3 colliders[noModels * 8];
	float colliderWidth = 1.5f;
	float colliderHeight = 2.0f;
	float colliderDepth = 0.4f;
	//window
	colliders[0] = { 0,0,0 };
	colliders[1] = { colliderWidth,0,0 };
	colliders[2] = { colliderWidth,colliderHeight,0 };
	colliders[3] = { 0,colliderHeight,0 };
	colliders[4] = { 0,0,colliderDepth };
	colliders[5] = { colliderWidth,0,colliderDepth };
	colliders[6] = { colliderWidth,colliderHeight,colliderDepth };
	colliders[7] = { 0,colliderHeight,colliderDepth };
	//door
	colliderWidth = 2.0f;
	colliderHeight = 3.0f;
	colliders[8+0] = { 0,0,0 };
	colliders[8 + 1] = { colliderWidth,0,0 };
	colliders[8 + 2] = { colliderWidth,colliderHeight,0 };
	colliders[8 + 3] = { 0,colliderHeight,0 };
	colliders[8 + 4] = { 0,0,colliderDepth };
	colliders[8 + 5] = { colliderWidth,0,colliderDepth };
	colliders[8 + 6] = { colliderWidth,colliderHeight,colliderDepth };
	colliders[8 + 7] = { 0,colliderHeight,colliderDepth };

	const int noPunchers = 2;
	int modelsOfPunchers[noPunchers + 1]{ 0,1,2 };
	int noVerticesInPunchersBfr[noPunchers + 2]{ 0, 0, 4, 8 };
	float2 puncherVertices[8];
	colliderWidth = 1.5f;
	colliderHeight = 2.0f;
	//window
	puncherVertices[0] = { 0,0 };
	puncherVertices[1] = { 0,colliderHeight };
	puncherVertices[2] = { colliderWidth,colliderHeight };
	puncherVertices[3] = { colliderWidth,0 };
	//door
	colliderWidth = 2.0f;
	colliderHeight = 3.0f;
	puncherVertices[4] = { 0,0 };
	puncherVertices[5] = { 0,colliderHeight };
	puncherVertices[6] = { colliderWidth,colliderHeight};
	puncherVertices[7] = { colliderWidth,0 };
	//end of input

	printf("End of input\n");
	int* noModelsInBuildingsBfr;
	int* noWallsInBuildingsBfr;
	int* noAssetsInWallsBfr;
	int* noArgumentsInWallsBfr;
	float3* modelPositions;
	float3* modelRotations;
	int* modelIds;
	wallsInfo walls;
	int* wallAssets;
	int* wallArguments;
	int* wallCultures;

	printf("Generating buildings\n");
	generateBuildings(noBuildings,
		noTypes,
		types,
		noModels,
		noBuildingsInTypesBfr,
		noPlotCornersInBuildingsBfr,
		plotCorners,
		noArgumentsInBuildingsBfr,
		argumentsInBuildings,
		cultures,
		noAssetsInBuildingsBfr,
		assetsInBuildings,
		noCollidersInModelsBfr,
		colliders,
		&noModelsInBuildingsBfr, 
		&noWallsInBuildingsBfr,
		&noAssetsInWallsBfr,
		&noArgumentsInWallsBfr,
		&modelPositions,
		&modelRotations,
		&modelIds,
		&walls,
		&wallAssets,
		&wallArguments,
		&wallCultures
	);
	int noWalls = noWallsInBuildingsBfr[noBuildings];

	int* sortedWallsIndexes = (int*)malloc(sizeof(int)*noWalls);
	for (int i = 0; i < noWalls; i++)
	{
		sortedWallsIndexes[i] = i;
	}
	auto sortRuleLambda = [walls](int a, int b) {return (int)walls.types[a] > (int)walls.types[b]; };
	std::sort(sortedWallsIndexes, sortedWallsIndexes + noWalls, sortRuleLambda);
	int* noAssetsInWallsBfrBuffer = (int*)malloc(sizeof(int)*(noWalls + 1));
	int noAssets = noAssetsInWallsBfr[noWalls];
	int* wallAssetsBuffer = (int*)malloc(sizeof(int)*noAssets);
	wallsInfo bufferInfo;
	bufferInfo.positions = (float3*)malloc(sizeof(float3)*noWalls);
	bufferInfo.rotations = (float3*)malloc(sizeof(float3)*noWalls);
	bufferInfo.dimensions = (float3*)malloc(sizeof(float3)*noWalls);
	bufferInfo.types = (int*)malloc(sizeof(int)*noWalls);
	bufferInfo.buildingIndexes = (int*)malloc(sizeof(int)*noWalls);
	//TODO: przez to, ze przechowuje info o assetach scian w splaszczonej tablicy,
	//nie da sie jej tak latwo zamienic indeksow. Chyba najprosciej bedzie
	//zrobic descan, zamienic, a potem zrobic scan. Dobrze byloby zrobic to na gpu
	//(ale sortowanie na cpu, podajemy do gpu indeksy)
	//no ale liczba scian jest wieksza niz block, trzeba robic zewnetrzny scan
	noAssetsInWallsBfrBuffer[0] = 0;
	for (int i = 0; i < noWalls; i++)
	{
		bufferInfo.positions[i] = walls.positions[sortedWallsIndexes[i]];
		bufferInfo.rotations[i] = walls.rotations[sortedWallsIndexes[i]];
		bufferInfo.dimensions[i] = walls.dimensions[sortedWallsIndexes[i]];
		bufferInfo.types[i] = walls.types[sortedWallsIndexes[i]];
		bufferInfo.buildingIndexes[i] = walls.buildingIndexes[sortedWallsIndexes[i]];
		//descan and sort
		noAssetsInWallsBfrBuffer[i + 1] = noAssetsInWallsBfr[sortedWallsIndexes[i] + 1] - noAssetsInWallsBfr[sortedWallsIndexes[i]];
		//scan
		noAssetsInWallsBfrBuffer[i + 1] += noAssetsInWallsBfrBuffer[i];
		//sorting compressed table
		for (int asset = 0; asset < noAssetsInWallsBfr[sortedWallsIndexes[i] + 1] - noAssetsInWallsBfr[sortedWallsIndexes[i]]; asset++)
		{
			wallAssetsBuffer[noAssetsInWallsBfrBuffer[i] + asset] = wallAssets[noAssetsInWallsBfr[sortedWallsIndexes[i]] + asset];
		}
	}
	free(wallAssets);
	free(noAssetsInWallsBfr);
	free(walls.positions);
	free(walls.rotations);
	free(walls.dimensions);
	free(walls.types);
	free(walls.buildingIndexes);
	wallAssets = wallAssetsBuffer;
	noAssetsInWallsBfr = noAssetsInWallsBfrBuffer;
	walls = bufferInfo;

#ifdef DEBUG
	printf("Assets\n");
	for (int wall = 0; wall < noWalls; wall++)
	{
		printf("Wall %d\n", wall);
		for (int asset = noAssetsInWallsBfr[wall]; asset < noAssetsInWallsBfr[wall + 1]; asset++)
		{
			printf("\t%d ", wallAssets[asset]);
		}
		printf("\n");
	}
#endif // DEBUG

	//TODO: gpu this
	int noWallTypes = 1;
	for (int i = 0; i < noWalls - 1; i++)
	{
		if (walls.types[i] != walls.types[i + 1])
		{
			noWallTypes++;
		}
	}
	int* noWallsInTypesBfr = (int*)malloc(sizeof(int)*(noWallTypes+1));
	noWallTypes = 1;
	noWallsInTypesBfr[0] = 0;
	for (int i = 0; i < noWalls - 1; i++)
	{
		if (walls.types[i] != walls.types[i + 1])
		{
			noWallsInTypesBfr[noWallTypes] = i + 1;
			noWallTypes++;
		}
	}
	noWallsInTypesBfr[noWallTypes] = noWalls;

	int* noModelsInWallsBfr;
	int* noVerticesInContoursBfr; 
	int* noVerticesInHolesBfr; 
	int* noHolesInWallsBfr;
	int* frontMaterials;
	int* backMaterials;
	int* innerMaterials; 
	int* outerMaterials;
	float* frontMaterialGrains; 
	float* backMaterialGrains; 
	float* innerMaterialGrains;
	float* outerMaterialGrains;
	float2* verticesInContours;
	float2* verticesInHoles;
	printf("Generating walls\n");
	generateWalls(noWalls,
		walls,
		noWallTypes,
		noModels,
		noPunchers,
		noWallsInTypesBfr,
		noArgumentsInWallsBfr,
		noAssetsInWallsBfr,
		noCollidersInModelsBfr,
		modelsOfPunchers,
		noVerticesInPunchersBfr,
		wallCultures,
		wallArguments,
		wallAssets,
		colliders,
		puncherVertices,
		&noModelsInWallsBfr,
		&noVerticesInContoursBfr,
		&noVerticesInHolesBfr,
		&noHolesInWallsBfr,
		&frontMaterials,
		&backMaterials,
		&innerMaterials,
		&outerMaterials,
		&frontMaterialGrains,
		&backMaterialGrains,
		&innerMaterialGrains,
		&outerMaterialGrains,
		&verticesInContours,
		&verticesInHoles,
		&modelPositions,
		&modelRotations,
		&modelIds
	);

#ifdef DEBUG
	printf("Models in walls\n");
	for (int wall = 0; wall < noWalls; wall++)
	{
		printf("Wall %d\n", wall);
		for (int model = noModelsInWallsBfr[wall]; model < noModelsInWallsBfr[wall + 1]; model++)
		{
			printf("\t%d: %f %f %f\n", modelIds[model], modelPositions[model].x, modelPositions[model].y, modelPositions[model].z);
		}
	}
#endif // DEBUG
	generateWallsArgs generateWallsArgs{ noWalls,
		frontMaterialGrains, backMaterialGrains, innerMaterialGrains, outerMaterialGrains,
		walls.positions, walls.rotations, walls.dimensions,
		noVerticesInContoursBfr, noHolesInWallsBfr, noVerticesInHolesBfr,
		verticesInContours, verticesInHoles };
	printf("Meshing walls\n");
	generateWallsResult generateWallsResult = meshWalls(generateWallsArgs);

	assembleFinalJsonArgs assembleJsonArgs{
		noBuildings, noWalls, walls.buildingIndexes, noHolesInWallsBfr, generateWallsResult.noVerticesInWallsBfr, noVerticesInHolesBfr, noVerticesInContoursBfr,
		generateWallsResult.triangles, walls.positions, walls.rotations, walls.dimensions, generateWallsResult.allVerticesValues,
		generateWallsResult.holesVerticesNormalsValues,  verticesInHoles,
		verticesInContours, generateWallsResult.contourNormalsValues,
		generateWallsResult.frontUvs, generateWallsResult.backUvs, generateWallsResult.innerUvs, generateWallsResult.outerUvs,
		buildingPositions,
		frontMaterials, backMaterials, innerMaterials, outerMaterials
	};
	printf("Assembling json\n");

	rapidjson::Document outputDoc;
	assembleFinalJson(assembleJsonArgs, outputDoc);
	writeDocumentToFile(outputDoc, "TriangulationOutput.json");
	printf("Done!\n");
	return 0;
}