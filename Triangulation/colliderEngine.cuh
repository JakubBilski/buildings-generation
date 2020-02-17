#pragma once
#include <cuda_runtime.h>
#include "defines.h"

__device__
inline float2 getRectangularSize(float2* colliderVertices, int noVertices)
{
	float minX = colliderVertices[0].x;
	float maxX = minX;
	float minY = colliderVertices[0].y;
	float maxY = minY;
	int x, y;
	for (int v = 1; v < noVertices; v++)
	{
		x = colliderVertices[v].x;
		y = colliderVertices[v].y;
		if (x < minX)
		{
			minX = x;
		}
		else if(x > maxX)
		{
			maxX = x;
		}
		if (y < minY)
		{
			minY = y;
		}
		else if(y > maxY)
		{
			maxY = y;
		}
	}
	float2 result;
	result.x = maxX - minX;
	result.y = maxY - minY;
	return result;
}

__device__
inline float4 getPivotRectangleSizes(float2* colliderVertices, int noVertices)
{
	float2 pivot = colliderVertices[0];
	float right = 0;
	float up = 0;
	float left = 0;
	float down = 0;
	int x, y;
	for (int v = 1; v < noVertices; v++)
	{
		x = colliderVertices[v].x - pivot.x;
		y = colliderVertices[v].y - pivot.y;
		if (x < left)
		{
			left = x;
		}
		else if (x > right)
		{
			right = x;
		}
		if (y < down)
		{
			down = y;
		}
		else if (y > up)
		{
			up = y;
		}
	}
	return { right, up, -left, -down };
}

__device__
inline float4 getPivotRectangleSizes(float3* colliderVertices, int noColliders)
{
	float2 pivot = { colliderVertices[0].x, colliderVertices[0].y };
	float right = 0;
	float up = 0;
	float left = 0;
	float down = 0;
	float x, y;
	for (int v = 1; v < 8*noColliders; v++)
	{
		x = colliderVertices[v].x - pivot.x;
		y = colliderVertices[v].y - pivot.y;
		if (x < left)
		{
			left = x;
		}
		else if (x > right)
		{
			right = x;
		}
		if (y < down)
		{
			down = y;
		}
		else if (y > up)
		{
			up = y;
		}
	}
	return { right, up, -left, -down };
}