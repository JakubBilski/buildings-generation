#pragma once
#include <cuda_runtime.h>
#include <string>

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
	int* types;
	float3* positions;
	float3* rotations;
	float3* dimensions;
	int* buildingIndexes;
};