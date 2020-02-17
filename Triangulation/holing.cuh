#pragma once
#include "defines.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <list>

bool isPointInsideTriangle(float2 s, float2 a, float2 b, float2 c)
{
	int as_x = s.x - a.x;
	int as_y = s.y - a.y;

	bool s_ab = (b.x - a.x)*as_y - (b.y - a.y)*as_x > 0;

	if ((c.x - a.x)*as_y - (c.y - a.y)*as_x > 0 == s_ab) return false;

	if ((c.x - b.x)*(s.y - b.y) - (c.y - b.y)*(s.x - b.x) > 0 != s_ab) return false;

	return true;
}
bool isPointBetweenRays(float2 s, float2 a, float2 b, float2 c)
{
	int as_x = s.x - a.x;
	int as_y = s.y - a.y;
	int cs_x = s.x - c.x;
	int cs_y = s.y - c.y;

	return (((c.x - a.x)*cs_y - (c.y - a.y)*cs_x > 0) != ((b.x - a.x)*as_y - (b.y - a.y)*as_x > 0));
}

void mergeHolesAndContoursCPU(
	int noWalls,
	int noVerticesInContoursBfr[],
	int noHolesInWallsBfr[],
	int noVerticesInHolesBfr[],
	float2 verticesInContours[],
	float2 verticesInHoles[],
	float2 out_holesAndContours[],
	int out_verticesInWallsBfr[],
	int* out_noWallsInBlocksBfr[],
	int* out_noBlocks)
{
	std::list<int> noWallsInBlocksBfrList;
	int noVerticesAssignedToBlocks = 0;
	out_verticesInWallsBfr[0] = 0; 
	noWallsInBlocksBfrList.push_back(0);
	for (int wall = 0; wall < noWalls; wall++)
	{
		int noContourVertices = noVerticesInContoursBfr[wall + 1] - noVerticesInContoursBfr[wall];
		int noHoles = noHolesInWallsBfr[wall + 1] - noHolesInWallsBfr[wall];

		std::list<float2> outList;
		outList.clear();
		for (int i = 0; i < noContourVertices; i++)
		{
			outList.push_back(verticesInContours[i + noVerticesInContoursBfr[wall]]);
		}

		int* holes = (int*)malloc(sizeof(int) * noHoles);
		int* maxXVertices = (int*)malloc(sizeof(int) * noHoles);
		int* maxXValues = (int*)malloc(sizeof(int) * noHoles);
		for (int hole = 0; hole < noHoles; hole++)
		{
			holes[hole] = hole;
			maxXVertices[hole] = 0;
			int firstVertexInHole = noVerticesInHolesBfr[hole + noHolesInWallsBfr[wall]];
			int noVerticesInHole = noVerticesInHolesBfr[hole + noHolesInWallsBfr[wall] + 1] - firstVertexInHole;
			for (int vertex = 1; vertex < noVerticesInHole; vertex++)
			{
				if (maxXValues[hole] < verticesInHoles[vertex + firstVertexInHole].x)
				{
					maxXVertices[hole] = vertex;
					maxXValues[hole] = verticesInHoles[vertex + firstVertexInHole].x;
				}
			}
		}
		auto sortRuleLambda = [maxXValues](int a, int b) {return maxXValues[a] > maxXValues[b]; };
		std::sort(holes, holes + noHoles, sortRuleLambda);
		for (int holesIndex = 0; holesIndex < noHoles; holesIndex++)
		{
			int hole = holes[holesIndex];
			float2 maxXVertex = verticesInHoles[maxXVertices[hole] + noVerticesInHolesBfr[hole + noHolesInWallsBfr[wall]]];
			std::list<float2>::iterator visiblePointCandidate;
			float2 closestIntersectionPoint;
			closestIntersectionPoint.x = 10000000;
			auto p1 = outList.begin();
			auto p2 = outList.begin();
			p2++;
			while (p2 != outList.end())
			{
				if ((p1->y >= maxXVertex.y && p2->y <= maxXVertex.y) || (p1->y <= maxXVertex.y && p2->y >= maxXVertex.y))
				{
					float2 intersectionPoint;
					intersectionPoint.y = maxXVertex.y;
					intersectionPoint.x = ((p2->y - maxXVertex.y)*(p1->x - p2->x)) / (p1->y - p2->y) + p2->x;
					if (intersectionPoint.x > maxXVertex.x && intersectionPoint.x < closestIntersectionPoint.x)
					{
						closestIntersectionPoint.x = intersectionPoint.x;
						closestIntersectionPoint.y = intersectionPoint.y;
						visiblePointCandidate = (p1->x > p2->x) ? p1 : p2;
					}
				}
				p1++;
				p2++;
			}
			p2 = outList.begin();
			{
				if ((p1->y > maxXVertex.y && p2->y < maxXVertex.y) || (p1->y < maxXVertex.y && p2->y > maxXVertex.y))
				{
					float2 intersectionPoint;
					intersectionPoint.y = maxXVertex.y;
					intersectionPoint.x = ((p2->y - maxXVertex.y)*(p1->x - p2->x)) / (p1->y - p2->y) + p2->x;
					if (intersectionPoint.x > maxXVertex.x && intersectionPoint.x < closestIntersectionPoint.x)
					{
						closestIntersectionPoint.x = intersectionPoint.x;
						closestIntersectionPoint.y = intersectionPoint.y;
						visiblePointCandidate = (p1->x > p2->x) ? p1 : p2;
					}
				}
			}
			float cosSmallestAngle = 0;
			std::list<float2>::iterator smallestAnglePointInsideTriangle = visiblePointCandidate;
			auto v = outList.begin();
			while(v != outList.end())
			{
				if (v->x == closestIntersectionPoint.x && v->y == closestIntersectionPoint.y)
				{
					smallestAnglePointInsideTriangle = v;
					break;
				}
				if (!isPointInsideTriangle(*v, maxXVertex, closestIntersectionPoint, *visiblePointCandidate))
				{
					float cosv = (v->x - maxXVertex.x) / (sqrt((v->x - maxXVertex.x)*(v->x - maxXVertex.x) + (v->y - maxXVertex.y)*(v->y - maxXVertex.y)));
					if (cosv > cosSmallestAngle)
					{
						smallestAnglePointInsideTriangle = v;
						cosSmallestAngle = cosv;
					}
					else if (cosv == cosSmallestAngle)
					{
						auto prev = smallestAnglePointInsideTriangle;
						prev++;
						auto next = v;
						next++;
						if (isPointBetweenRays(*visiblePointCandidate, *prev, *v, *next))
						{
							smallestAnglePointInsideTriangle = v;
							cosSmallestAngle = cosv;
						}
					}
				}
				v++;
			}
			float2* holeBegin = &verticesInHoles[noVerticesInHolesBfr[hole + noHolesInWallsBfr[wall]]];
			float2* holeEnd = &verticesInHoles[noVerticesInHolesBfr[hole + noHolesInWallsBfr[wall] + 1]];
			float2* holeJointVertex = &(verticesInHoles[maxXVertices[hole] + noVerticesInHolesBfr[hole + noHolesInWallsBfr[wall]]]);
			//duplicating vertex in contour
			outList.insert(smallestAnglePointInsideTriangle, *smallestAnglePointInsideTriangle);
			//adding hole vertices from joint to end of the hole
			outList.insert(smallestAnglePointInsideTriangle, holeJointVertex, holeEnd);
			//adding hole vertices from the beginning of the hole to joint, including the joint once again to duplicate it
			outList.insert(smallestAnglePointInsideTriangle, holeBegin, holeJointVertex + 1);
		}
		int outIndex = noVerticesInContoursBfr[wall] + noVerticesInHolesBfr[noHolesInWallsBfr[wall]] + noHolesInWallsBfr[wall] * 2;
		int noVerticesToSave = noContourVertices + noVerticesInHolesBfr[noHolesInWallsBfr[wall+1]] - noVerticesInHolesBfr[noHolesInWallsBfr[wall]] + noHoles * 2;
		if (outIndex + noVerticesToSave - noVerticesAssignedToBlocks > NO_VERTICES_FOR_BLOCK_TRIANGULATION)
		{
			noWallsInBlocksBfrList.push_back(wall);
			noVerticesAssignedToBlocks = out_verticesInWallsBfr[wall];
		}
		for (float2 v : outList)
		{
			out_holesAndContours[outIndex] = v;
			outIndex++;
		}
		out_verticesInWallsBfr[wall + 1] = outIndex;
	}
	noWallsInBlocksBfrList.push_back(noWalls);
	*out_noBlocks = noWallsInBlocksBfrList.size() - 1;
	*out_noWallsInBlocksBfr = (int*)malloc(sizeof(int) * (*out_noBlocks + 1));
	int index = 0;
	for (int v : noWallsInBlocksBfrList)
	{
		(*out_noWallsInBlocksBfr)[index] = v;
		index++;
	}
}