#pragma once
#include <cuda_runtime.h>
#include "defines.h"

__device__
inline void generateTenementStyle(int culture, int assets[])
{
	if (culture == 0)
	{
		if (assets[0] == 0)
		{
			assets[0] = 1;
		}
		if (assets[1] == 0)
		{
			assets[1] = 2;
		}
		if (assets[2] == 0)
		{
			assets[2] = 3;
		}
		if (assets[3] == 0)
		{
			assets[3] = 3;
		}
		if (assets[4] == 0)
		{
			assets[4] = 2;
		}
		if (assets[5] == 0)
		{
			assets[5] = 1;
		}
	}
}

__device__
inline void generateTenementWallStyle(int culture, int assets[])
{
	if (culture == 0)
	{
		if (assets[0] == 0)
		{
			assets[0] = 1;
		}
		if (assets[1] == 0)
		{
			assets[1] = 2;
		}
		if (assets[2] == 0)
		{
			assets[2] = 3;
		}
		if (assets[3] == 0)
		{
			assets[3] = 3;
		}
		if (assets[4] == 0)
		{
			assets[4] = 2;
		}
		if (assets[5] == 0)
		{
			assets[5] = 1;
		}
	}
}

__device__
inline void generateTenementKitchenWallStyle(int culture, int assets[])
{
	if (culture == 0)
	{
		if (assets[0] == 0)
		{
			assets[0] = 1;
		}
		if (assets[1] == 0)
		{
			assets[1] = 2;
		}
		if (assets[2] == 0)
		{
			assets[2] = 3;
		}
		if (assets[3] == 0)
		{
			assets[3] = 3;
		}
		if (assets[4] == 0)
		{
			assets[4] = 2;
		}
		if (assets[5] == 0)
		{
			assets[5] = 1;
		}
	}
}