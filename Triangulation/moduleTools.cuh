#pragma once
#include <cuda_runtime.h>
#include "cub/block/block_scan.cuh"
#include "defines.h"

__device__
inline int bookBlockSpace(int building, int space, int table[])
{
	int out_noWallsBfr;
	if (building == 0)
	{
		space += table[0];
	}
	typedef cub::BlockScan<int, NO_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;
	__shared__ typename BlockScan::TempStorage temp_storage;
	BlockScan(temp_storage).ExclusiveSum(space, out_noWallsBfr);
	if (building == 0)
	{
		out_noWallsBfr += table[0];
		space -= table[0];
	}
	table[building + 1] = space + out_noWallsBfr;
	return out_noWallsBfr;
}

__device__
inline void bookAssetsSpace(int building, int noWallsBfr, int noWalls, int noWallAssets[], int noAssetsInWallsBfr[])
{
	int buildingSpace = 0;
	for (int i = 0; i < noWalls; i++)
	{
		buildingSpace += noWallAssets[i];
	}
	int assetSaveIndex;
	if (building == 0)
	{
		buildingSpace += noAssetsInWallsBfr[noWallsBfr];
	}
	typedef cub::BlockScan<int, NO_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;
	__shared__ typename BlockScan::TempStorage temp_storage;
	BlockScan(temp_storage).ExclusiveSum(buildingSpace, assetSaveIndex);
	for (int i = 0; i < noWalls + 1; i++)
	{
		noAssetsInWallsBfr[noWallsBfr + i] = assetSaveIndex;
		assetSaveIndex += noWallAssets[i];
	}
}