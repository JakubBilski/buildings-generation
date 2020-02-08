#pragma once
#include <cuda_runtime.h>
#include <string>

enum BuildingLogicType
{
	COTTAGE_B,
	AMAZING_B
};

enum WallType
{
	BLUE_W,
	AMAZING_W
};

struct modelInfo
{
	float3 position;
	float3 rotation;
	int id;
};

struct buildingsInfo
{
	float3* positions;
};

struct wallsInfo
{
	WallType* types;
	float3* positions;
	float3* rotations;
	float3* dimensions;
	int* buildingIndexes;
};