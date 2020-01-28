#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <map>

#include "wallsMeshing.cuh"
#include "buildingsGeneration.cuh"
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
	buildingsInfo info;
	info.positions = (float3*)malloc(sizeof(float3)*noBuildings);
	for (int i = 0; i < noBuildings; i++)
	{
		info.positions[i] = { (float)((i % 20) * 20), 0, (float)((i / 20) * 20) };
	}
	const int noTypes = 1;
	BuildingType types[noTypes]{ BuildingType::COTTAGE_B };
	int noBuildingsInTypesBfr[noTypes+1]{ 0,noBuildings };
	int noPlotCornersInBuildingsBfr[noBuildings+1];
	noPlotCornersInBuildingsBfr[0] = 0;
	const int noPlotCorners = 5;
	for (int i = 1; i < noBuildings+1; i++)
	{
		noPlotCornersInBuildingsBfr[i] = noPlotCornersInBuildingsBfr[i - 1] + noPlotCorners;
	}
	float3 plotCorners[noBuildings * noPlotCorners];
	//for (int i = 0; i < noBuildings; i++)
	//{
	//	plotCorners[noPlotCorners * i] = { 0,0,0 };
	//	plotCorners[noPlotCorners * i + 1] = { 10,0,0 };
	//	plotCorners[noPlotCorners * i + 2] = { 10,0,10 };
	//	plotCorners[noPlotCorners * i + 3] = { 0,0,10 };
	//}
	for (int i = 0; i < noBuildings; i++)
	{
		plotCorners[noPlotCorners * i] = { 0,0,0 };
		plotCorners[noPlotCorners * i + 1] = { 5,0,0 };
		plotCorners[noPlotCorners * i + 2] = { 10,0,5 };
		plotCorners[noPlotCorners * i + 3] = { 5,0,10 };
		plotCorners[noPlotCorners * i + 4] = { 0,0,5 };
	}
	//end of input



	int* noModelsInBuildingsBfr;
	int* noWallsInBuildingsBfr;
	modelInfo* models;
	wallsInfo walls;
	generateBuildings(noBuildings, info, noTypes, types, noBuildingsInTypesBfr, noPlotCornersInBuildingsBfr, plotCorners,
		&noModelsInBuildingsBfr, &noWallsInBuildingsBfr, &models, &walls);

	int noWalls = noWallsInBuildingsBfr[noBuildings];

	int* sortedWallsIndexes = (int*)malloc(sizeof(int)*noWalls);
	for (int i = 0; i < noWalls; i++)
	{
		sortedWallsIndexes[i] = i;
	}
	auto sortRuleLambda = [walls](int a, int b) {return (int)walls.types[a] > (int)walls.types[b]; };
	std::sort(sortedWallsIndexes, sortedWallsIndexes + noWalls, sortRuleLambda);
	wallsInfo bufferInfo;
	bufferInfo.positions = (float3*)malloc(sizeof(float3)*noWalls);
	bufferInfo.vectorXs = (float3*)malloc(sizeof(float3)*noWalls);
	bufferInfo.vectorYs = (float3*)malloc(sizeof(float3)*noWalls);
	bufferInfo.vectorWidths = (float3*)malloc(sizeof(float3)*noWalls);
	bufferInfo.types = (WallType*)malloc(sizeof(WallType)*noWalls);
	bufferInfo.buildingIndexes = (int*)malloc(sizeof(int)*noWalls);
	for (int i = 0; i < noWalls; i++)
	{
		bufferInfo.positions[i] = walls.positions[sortedWallsIndexes[i]];
		bufferInfo.vectorXs[i] = walls.vectorXs[sortedWallsIndexes[i]];
		bufferInfo.vectorYs[i] = walls.vectorYs[sortedWallsIndexes[i]];
		bufferInfo.vectorWidths[i] = walls.vectorWidths[sortedWallsIndexes[i]];
		bufferInfo.types[i] = walls.types[sortedWallsIndexes[i]];
		bufferInfo.buildingIndexes[i] = walls.buildingIndexes[sortedWallsIndexes[i]];
	}
	free(walls.positions);
	free(walls.vectorXs);
	free(walls.vectorYs);
	free(walls.vectorWidths);
	free(walls.types);
	free(walls.buildingIndexes);
	walls = bufferInfo;


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
	generateWalls(noWalls, walls, noWallTypes, noWallsInTypesBfr,
		&noModelsInWallsBfr, &noVerticesInContoursBfr, &noVerticesInHolesBfr, &noHolesInWallsBfr,
		&frontMaterials, &backMaterials, &innerMaterials, &outerMaterials,
		&frontMaterialGrains, &backMaterialGrains, &innerMaterialGrains, &outerMaterialGrains,
		&verticesInContours, &verticesInHoles
	);

	//TODO: replace following with gpu operations

	float3* normalsOfWalls = (float3*)malloc(sizeof(float3)*noWalls);
	for (int i = 0; i < noWalls; i++)
	{
		float x = - walls.vectorWidths[i].x;
		float y = - walls.vectorWidths[i].y;
		float z = - walls.vectorWidths[i].z;
		float delimiter = sqrt(x*x + y * y + z * z);
		normalsOfWalls[i] = { x / delimiter, y / delimiter, z / delimiter };
	}
	std::map<int, std::string> materialsInfo;
	materialsInfo.insert(std::pair<int, std::string>(0, "plaster_blue_damaged"));
	materialsInfo.insert(std::pair<int, std::string>(1, "plaster_red_damaged"));
	materialsInfo.insert(std::pair<int, std::string>(2, "stone"));
	generateWallsArgs generateWallsArgs{ noWalls,
		frontMaterialGrains, backMaterialGrains, innerMaterialGrains, outerMaterialGrains,
		walls.vectorWidths, walls.vectorXs, walls.vectorYs, walls.positions,
		noVerticesInContoursBfr, noHolesInWallsBfr, noVerticesInHolesBfr,
		verticesInContours, verticesInHoles };
	generateWallsResult generateWallsResult = meshWalls(generateWallsArgs);

	assembleFinalJsonArgs assembleJsonArgs{
		noBuildings, noWalls, walls.buildingIndexes, noHolesInWallsBfr, generateWallsResult.noVerticesInWallsBfr, noVerticesInHolesBfr, noVerticesInContoursBfr,
		generateWallsResult.triangles, normalsOfWalls, walls.vectorWidths, walls.positions, generateWallsResult.worldSpaceVerticesValues,
		generateWallsResult.worldSpaceHolesVerticesNormalsValues,  generateWallsResult.worldSpaceHolesVerticesValues,
		generateWallsResult.worldSpaceContoursVerticesValues, generateWallsResult.worldSpaceContourNormalsValues,
		generateWallsResult.frontUvs, generateWallsResult.backUvs, generateWallsResult.innerUvs, generateWallsResult.outerUvs,
		info.positions,
		frontMaterials, backMaterials, innerMaterials, outerMaterials, &materialsInfo
	};
	rapidjson::Document outputDoc;
	assembleFinalJson(assembleJsonArgs, outputDoc);
	writeDocumentToFile(outputDoc, "TriangulationOutput.json");
	printf("Done!\n");
	return 0;
}