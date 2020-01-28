#pragma once

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <cuda_runtime.h>
#include <string>
#include <array>
#include <fstream>
#include <iostream>
#include <map>


struct assembleFinalJsonArgs
{
	int noBuildings;
	int noWalls;
	int* buildingOfWalls;
	int* noHolesInWallsBfr;
	int* noVerticesInWallsBfr;
	int* noVerticesInHolesBfr;
	int* noVerticesInContoursBfr;

	int* triangles;
	float3* normalsOfWalls;
	float3* widthOfWalls;
	float3* shiftOfWalls;
	float3* verticesValues;
	float3* normalsValues;
	float3* holesVerticesValues;

	float3* contourVerticesValues;
	float3* contourNormalValues;
	float2* frontUvsValues;
	float2* backUvsValues;
	float2* innerUvsValues;
	float2* outerUvsValues;

	float3* buildingPositions;

	int* frontMaterials;
	int* backMaterials;
	int* innerMaterials;
	int* outerMaterials;

	std::map<int, std::string>* materialsInfo;
};

rapidjson::Document readDocumentFromFile(std::string path)
{
	std::ifstream ifs(path);
	std::string content((std::istreambuf_iterator<char>(ifs)),
		(std::istreambuf_iterator<char>()));
	rapidjson::Document document;
	document.Parse(content.c_str());
	ifs.close();
	return document;
}
void writeDocumentToFile(rapidjson::Document& document, std::string path)
{
	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	document.Accept(writer);
	//std::cout << buffer.GetString() << std::endl;
	std::ofstream ofs(path);
	ofs << buffer.GetString();
	ofs.close();
}


void assembleFinalJson(assembleFinalJsonArgs args, rapidjson::Document& outputDoc)
{
	rapidjson::Document::AllocatorType& allocator = outputDoc.GetAllocator();
	outputDoc.SetObject();
	rapidjson::Value plots(rapidjson::kArrayType);
	rapidjson::Value* buildingsWalls = (rapidjson::Value*)malloc(sizeof(rapidjson::Value)*args.noBuildings);
	for (int buildingIndex = 0; buildingIndex < args.noBuildings; buildingIndex++)
	{
		rapidjson::Value walls(rapidjson::kArrayType);
		buildingsWalls[buildingIndex] = walls;
	}

	for (int wallIndex = 0; wallIndex < args.noWalls; wallIndex++)
	{
		rapidjson::Value wall(rapidjson::kObjectType);
		rapidjson::Value meshes(rapidjson::kArrayType);

		rapidjson::Value frontMesh(rapidjson::kObjectType);
		rapidjson::Value backMesh(rapidjson::kObjectType);
		rapidjson::Value innerMesh(rapidjson::kObjectType);
		rapidjson::Value outerMesh(rapidjson::kObjectType);

		rapidjson::Value frontVertices(rapidjson::kArrayType);
		rapidjson::Value backVertices(rapidjson::kArrayType);
		rapidjson::Value innerVertices(rapidjson::kArrayType);
		rapidjson::Value outerVertices(rapidjson::kArrayType);

		rapidjson::Value frontTriangles(rapidjson::kArrayType);
		rapidjson::Value backTriangles(rapidjson::kArrayType);
		rapidjson::Value innerTriangles(rapidjson::kArrayType);
		rapidjson::Value outerTriangles(rapidjson::kArrayType);

		rapidjson::Value frontNormals(rapidjson::kArrayType);
		rapidjson::Value backNormals(rapidjson::kArrayType);
		rapidjson::Value innerNormals(rapidjson::kArrayType);
		rapidjson::Value outerNormals(rapidjson::kArrayType);

		rapidjson::Value frontUvs(rapidjson::kArrayType);
		rapidjson::Value backUvs(rapidjson::kArrayType);
		rapidjson::Value innerUvs(rapidjson::kArrayType);
		rapidjson::Value outerUvs(rapidjson::kArrayType);

		for (int vertexIndex = args.noVerticesInWallsBfr[wallIndex]; vertexIndex < args.noVerticesInWallsBfr[wallIndex + 1]; vertexIndex++)
		{
			rapidjson::Value frontVertex(rapidjson::kObjectType);
			frontVertex.AddMember("x", rapidjson::Value().SetDouble(args.verticesValues[vertexIndex].x), allocator);
			frontVertex.AddMember("y", rapidjson::Value().SetDouble(args.verticesValues[vertexIndex].y), allocator);
			frontVertex.AddMember("z", rapidjson::Value().SetDouble(args.verticesValues[vertexIndex].z), allocator);
			frontVertices.PushBack(frontVertex, allocator);

			rapidjson::Value backVertex(rapidjson::kObjectType);
			backVertex.AddMember("x", rapidjson::Value().SetDouble(args.verticesValues[vertexIndex].x + args.widthOfWalls[wallIndex].x), allocator);
			backVertex.AddMember("y", rapidjson::Value().SetDouble(args.verticesValues[vertexIndex].y + args.widthOfWalls[wallIndex].y), allocator);
			backVertex.AddMember("z", rapidjson::Value().SetDouble(args.verticesValues[vertexIndex].z + args.widthOfWalls[wallIndex].z), allocator);
			backVertices.PushBack(backVertex, allocator);

			rapidjson::Value frontNormal(rapidjson::kObjectType);
			frontNormal.AddMember("x", rapidjson::Value().SetDouble(args.normalsOfWalls[wallIndex].x), allocator);
			frontNormal.AddMember("y", rapidjson::Value().SetDouble(args.normalsOfWalls[wallIndex].y), allocator);
			frontNormal.AddMember("z", rapidjson::Value().SetDouble(args.normalsOfWalls[wallIndex].z), allocator);

			rapidjson::Value backNormal(rapidjson::kObjectType);
			backNormal.AddMember("x", rapidjson::Value().SetDouble(-args.normalsOfWalls[wallIndex].x), allocator);
			backNormal.AddMember("y", rapidjson::Value().SetDouble(-args.normalsOfWalls[wallIndex].y), allocator);
			backNormal.AddMember("z", rapidjson::Value().SetDouble(-args.normalsOfWalls[wallIndex].z), allocator);

			rapidjson::Value frontUv(rapidjson::kObjectType);
			frontUv.AddMember("x", rapidjson::Value().SetDouble(args.frontUvsValues[vertexIndex].x), allocator);
			frontUv.AddMember("y", rapidjson::Value().SetDouble(args.frontUvsValues[vertexIndex].y), allocator);

			rapidjson::Value backUv(rapidjson::kObjectType);
			backUv.AddMember("x", rapidjson::Value().SetDouble(args.backUvsValues[vertexIndex].x), allocator);
			backUv.AddMember("y", rapidjson::Value().SetDouble(args.backUvsValues[vertexIndex].y), allocator);

			frontNormals.PushBack(frontNormal, allocator);
			backNormals.PushBack(backNormal, allocator);

			frontUvs.PushBack(frontUv, allocator);
			backUvs.PushBack(backUv, allocator);
		}

		int noVerticesInTrianglesBfrThisWall = 3 * (args.noVerticesInWallsBfr[wallIndex] - wallIndex * 2);
		int noVerticesInTrianglesBfrNextWall = 3 * (args.noVerticesInWallsBfr[wallIndex + 1] - (wallIndex + 1) * 2);

		for (int triangleIndex = noVerticesInTrianglesBfrThisWall; triangleIndex < noVerticesInTrianglesBfrNextWall; triangleIndex++)
		{
			frontTriangles.PushBack(rapidjson::Value().SetInt(args.triangles[triangleIndex]), allocator);
			backTriangles.PushBack(rapidjson::Value().SetInt(args.triangles[noVerticesInTrianglesBfrNextWall + noVerticesInTrianglesBfrThisWall - 1 - triangleIndex]), allocator);
		}
		int localVertexBottom;
		int localNoVertices;
		for (int holeIndex = args.noHolesInWallsBfr[wallIndex]; holeIndex < args.noHolesInWallsBfr[wallIndex + 1]; holeIndex++)
		{
			localVertexBottom = args.noVerticesInHolesBfr[holeIndex];
			int holeLift = localVertexBottom - args.noVerticesInHolesBfr[args.noHolesInWallsBfr[wallIndex]];
			localNoVertices = args.noVerticesInHolesBfr[holeIndex + 1] - localVertexBottom;
			for (int vertexIndex = localVertexBottom; vertexIndex < args.noVerticesInHolesBfr[holeIndex + 1]; vertexIndex++)
			{
				rapidjson::Value innerVertex1(rapidjson::kObjectType);
				innerVertex1.AddMember("x", rapidjson::Value().SetDouble(args.holesVerticesValues[vertexIndex].x), allocator);
				innerVertex1.AddMember("y", rapidjson::Value().SetDouble(args.holesVerticesValues[vertexIndex].y), allocator);
				innerVertex1.AddMember("z", rapidjson::Value().SetDouble(args.holesVerticesValues[vertexIndex].z), allocator);
				rapidjson::Value innerVertex2(rapidjson::kObjectType);
				innerVertex2.AddMember("x", rapidjson::Value().SetDouble(args.holesVerticesValues[vertexIndex].x + args.widthOfWalls[wallIndex].x), allocator);
				innerVertex2.AddMember("y", rapidjson::Value().SetDouble(args.holesVerticesValues[vertexIndex].y + args.widthOfWalls[wallIndex].y), allocator);
				innerVertex2.AddMember("z", rapidjson::Value().SetDouble(args.holesVerticesValues[vertexIndex].z + args.widthOfWalls[wallIndex].z), allocator);
				innerVertices.PushBack(innerVertex1, allocator);
				innerVertices.PushBack(innerVertex2, allocator);
				rapidjson::Value innerUv1(rapidjson::kObjectType);
				innerUv1.AddMember("x", rapidjson::Value().SetDouble(args.innerUvsValues[vertexIndex].x), allocator);
				innerUv1.AddMember("y", rapidjson::Value().SetDouble(args.innerUvsValues[vertexIndex].y), allocator);
				rapidjson::Value innerUv2(rapidjson::kObjectType);
				innerUv2.AddMember("x", rapidjson::Value().SetDouble(args.innerUvsValues[vertexIndex].x), allocator);
				innerUv2.AddMember("y", rapidjson::Value().SetDouble(args.innerUvsValues[vertexIndex].y), allocator);
				innerUvs.PushBack(innerUv1, allocator);
				innerUvs.PushBack(innerUv2, allocator);
				int nextVertex = ((vertexIndex - localVertexBottom + 1) % localNoVertices + holeLift) * 2;
				innerTriangles.PushBack(rapidjson::Value().SetInt(nextVertex), allocator);
				innerTriangles.PushBack(rapidjson::Value().SetInt((vertexIndex - localVertexBottom + holeLift) * 2), allocator);
				innerTriangles.PushBack(rapidjson::Value().SetInt((vertexIndex - localVertexBottom + holeLift) * 2 + 1), allocator);
				innerTriangles.PushBack(rapidjson::Value().SetInt(nextVertex + 1), allocator);
				innerTriangles.PushBack(rapidjson::Value().SetInt(nextVertex), allocator);
				innerTriangles.PushBack(rapidjson::Value().SetInt((vertexIndex - localVertexBottom + holeLift) * 2 + 1), allocator);
				rapidjson::Value innerNormal1(rapidjson::kObjectType);
				innerNormal1.AddMember("x", rapidjson::Value().SetDouble(args.normalsValues[vertexIndex].x), allocator);
				innerNormal1.AddMember("y", rapidjson::Value().SetDouble(args.normalsValues[vertexIndex].y), allocator);
				innerNormal1.AddMember("z", rapidjson::Value().SetDouble(args.normalsValues[vertexIndex].z), allocator);
				rapidjson::Value innerNormal2(rapidjson::kObjectType);
				innerNormal2.AddMember("x", rapidjson::Value().SetDouble(args.normalsValues[vertexIndex].x), allocator);
				innerNormal2.AddMember("y", rapidjson::Value().SetDouble(args.normalsValues[vertexIndex].y), allocator);
				innerNormal2.AddMember("z", rapidjson::Value().SetDouble(args.normalsValues[vertexIndex].z), allocator);
				innerNormals.PushBack(innerNormal1, allocator);
				innerNormals.PushBack(innerNormal2, allocator);
			}
		}


		localVertexBottom = args.noVerticesInContoursBfr[wallIndex];
		localNoVertices = args.noVerticesInContoursBfr[wallIndex + 1] - localVertexBottom;
		for (int vertexIndex = localVertexBottom; vertexIndex < args.noVerticesInContoursBfr[wallIndex + 1]; vertexIndex++)
		{
			rapidjson::Value outerVertex1(rapidjson::kObjectType);
			outerVertex1.AddMember("x", rapidjson::Value().SetDouble(args.contourVerticesValues[vertexIndex].x), allocator);
			outerVertex1.AddMember("y", rapidjson::Value().SetDouble(args.contourVerticesValues[vertexIndex].y), allocator);
			outerVertex1.AddMember("z", rapidjson::Value().SetDouble(args.contourVerticesValues[vertexIndex].z), allocator);
			rapidjson::Value outerVertex2(rapidjson::kObjectType);
			outerVertex2.AddMember("x", rapidjson::Value().SetDouble(args.contourVerticesValues[vertexIndex].x + args.widthOfWalls[wallIndex].x), allocator);
			outerVertex2.AddMember("y", rapidjson::Value().SetDouble(args.contourVerticesValues[vertexIndex].y + args.widthOfWalls[wallIndex].y), allocator);
			outerVertex2.AddMember("z", rapidjson::Value().SetDouble(args.contourVerticesValues[vertexIndex].z + args.widthOfWalls[wallIndex].z), allocator);
			outerVertices.PushBack(outerVertex1, allocator);
			outerVertices.PushBack(outerVertex2, allocator);
			rapidjson::Value outerUv1(rapidjson::kObjectType);
			outerUv1.AddMember("x", rapidjson::Value().SetDouble(args.outerUvsValues[vertexIndex].x), allocator);
			outerUv1.AddMember("y", rapidjson::Value().SetDouble(args.outerUvsValues[vertexIndex].y), allocator);
			rapidjson::Value outerUv2(rapidjson::kObjectType);
			outerUv2.AddMember("x", rapidjson::Value().SetDouble(args.outerUvsValues[vertexIndex].x), allocator);
			outerUv2.AddMember("y", rapidjson::Value().SetDouble(args.outerUvsValues[vertexIndex].y), allocator);
			outerUvs.PushBack(outerUv1, allocator);
			outerUvs.PushBack(outerUv2, allocator);
			int nextVertex = ((vertexIndex - localVertexBottom + 1) % localNoVertices) * 2;
			outerTriangles.PushBack(rapidjson::Value().SetInt(nextVertex), allocator);
			outerTriangles.PushBack(rapidjson::Value().SetInt((vertexIndex - localVertexBottom) * 2), allocator);
			outerTriangles.PushBack(rapidjson::Value().SetInt((vertexIndex - localVertexBottom) * 2 + 1), allocator);
			outerTriangles.PushBack(rapidjson::Value().SetInt(nextVertex + 1), allocator);
			outerTriangles.PushBack(rapidjson::Value().SetInt(nextVertex), allocator);
			outerTriangles.PushBack(rapidjson::Value().SetInt((vertexIndex - localVertexBottom) * 2 + 1), allocator);
			rapidjson::Value outerNormal1(rapidjson::kObjectType);
			outerNormal1.AddMember("x", rapidjson::Value().SetDouble(args.contourNormalValues[vertexIndex].x), allocator);
			outerNormal1.AddMember("y", rapidjson::Value().SetDouble(args.contourNormalValues[vertexIndex].y), allocator);
			outerNormal1.AddMember("z", rapidjson::Value().SetDouble(args.contourNormalValues[vertexIndex].z), allocator);
			rapidjson::Value outerNormal2(rapidjson::kObjectType);
			outerNormal2.AddMember("x", rapidjson::Value().SetDouble(args.contourNormalValues[vertexIndex].x), allocator);
			outerNormal2.AddMember("y", rapidjson::Value().SetDouble(args.contourNormalValues[vertexIndex].y), allocator);
			outerNormal2.AddMember("z", rapidjson::Value().SetDouble(args.contourNormalValues[vertexIndex].z), allocator);
			outerNormals.PushBack(outerNormal1, allocator);
			outerNormals.PushBack(outerNormal2, allocator);
		}

		frontMesh.AddMember("vertices", frontVertices, allocator);
		frontMesh.AddMember("triangles", frontTriangles, allocator);
		frontMesh.AddMember("normals", frontNormals, allocator);
		frontMesh.AddMember("uvs", frontUvs, allocator);
		frontMesh.AddMember("material", rapidjson::Value().SetString(rapidjson::StringRef((*args.materialsInfo)[args.backMaterials[wallIndex]].c_str())), allocator);

		backMesh.AddMember("vertices", backVertices, allocator);
		backMesh.AddMember("triangles", backTriangles, allocator);
		backMesh.AddMember("normals", backNormals, allocator);
		backMesh.AddMember("uvs", backUvs, allocator);
		backMesh.AddMember("material", rapidjson::Value().SetString(rapidjson::StringRef((*args.materialsInfo)[args.frontMaterials[wallIndex]].c_str())), allocator);

		innerMesh.AddMember("vertices", innerVertices, allocator);
		innerMesh.AddMember("triangles", innerTriangles, allocator);
		innerMesh.AddMember("normals", innerNormals, allocator);
		innerMesh.AddMember("uvs", innerUvs, allocator);
		innerMesh.AddMember("material", rapidjson::Value().SetString(rapidjson::StringRef((*args.materialsInfo)[args.innerMaterials[wallIndex]].c_str())), allocator);

		outerMesh.AddMember("vertices", outerVertices, allocator);
		outerMesh.AddMember("triangles", outerTriangles, allocator);
		outerMesh.AddMember("normals", outerNormals, allocator);
		outerMesh.AddMember("uvs", outerUvs, allocator);
		outerMesh.AddMember("material", rapidjson::Value().SetString(rapidjson::StringRef((*args.materialsInfo)[args.outerMaterials[wallIndex]].c_str())), allocator);

		meshes.PushBack(frontMesh, allocator);
		meshes.PushBack(backMesh, allocator);
		meshes.PushBack(innerMesh, allocator);
		meshes.PushBack(outerMesh, allocator);
		wall.AddMember("meshes", meshes, allocator);
		rapidjson::Value wallPosition(rapidjson::kObjectType);
		wallPosition.AddMember("x", rapidjson::Value().SetDouble(args.shiftOfWalls[wallIndex].x), allocator);
		wallPosition.AddMember("y", rapidjson::Value().SetDouble(args.shiftOfWalls[wallIndex].y), allocator);
		wallPosition.AddMember("z", rapidjson::Value().SetDouble(args.shiftOfWalls[wallIndex].z), allocator);
		wall.AddMember("position", wallPosition, allocator);
		(buildingsWalls[args.buildingOfWalls[wallIndex]]).PushBack(wall, allocator);
	}

	for (int buildingIndex = 0; buildingIndex < args.noBuildings; buildingIndex++)
	{
		rapidjson::Value plot(rapidjson::kObjectType);
		rapidjson::Value plotPosition(rapidjson::kObjectType);
		plotPosition.AddMember("x", rapidjson::Value().SetDouble(args.buildingPositions[buildingIndex].x), allocator);
		plotPosition.AddMember("y", rapidjson::Value().SetDouble(args.buildingPositions[buildingIndex].y), allocator);
		plotPosition.AddMember("z", rapidjson::Value().SetDouble(args.buildingPositions[buildingIndex].z), allocator);
		plot.AddMember("position", plotPosition, allocator);
		plot.AddMember("models", rapidjson::Value().SetArray(), allocator);
		plot.AddMember("type", rapidjson::Value().SetString("example"), allocator);

		plot.AddMember("walls", buildingsWalls[buildingIndex], allocator);
		plots.PushBack(plot, allocator);
	}
	outputDoc.AddMember("plots", plots, allocator);
}