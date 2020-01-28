#pragma once
#include <cuda_runtime.h>
#include <string>

enum BuildingType
{
	COTTAGE_B
};

enum WallType
{
	BLUE_W,
	RED_W
};

struct modelInfo
{
	float3 position;
	float3 rotation;
	std::string name;
};

struct buildingsInfo
{
	float3* positions;
};

struct wallsInfo
{
	WallType* types;
	float3* positions;
	float3* vectorXs;
	float3* vectorYs;
	float3* vectorWidths;
	int* buildingIndexes;
};