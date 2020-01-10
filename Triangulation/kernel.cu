
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <fstream>
#include <iostream>

#include "triangulation.cuh"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

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
	std::cout << buffer.GetString() << std::endl;
	std::ofstream ofs(path);
	ofs << buffer.GetString();
	ofs.close();
}
rapidjson::Document createExample()
{
	rapidjson::Document outputDoc;
	outputDoc.SetObject();
	rapidjson::Value plots(rapidjson::kArrayType);
	rapidjson::Value plot(rapidjson::kObjectType);
	rapidjson::Value walls(rapidjson::kArrayType);
	rapidjson::Value wall(rapidjson::kObjectType);
	rapidjson::Value meshes(rapidjson::kArrayType);
	rapidjson::Value vertexX(12.54);
	//rapidjson::Value vertexY(10.54);
	//rapidjson::Value vertexZ(8.5432);

	rapidjson::Document::AllocatorType& allocator = outputDoc.GetAllocator();

	//front and back meshes
	double frontVerticesX[8]{ 0, 100, 100, 0, 25, 50, 50, 25 };
	double frontVerticesY[8]{ 0, 0, 100, 100, 50, 50, 75, 75 };
	rapidjson::Value frontMesh(rapidjson::kObjectType);
	rapidjson::Value backMesh(rapidjson::kObjectType);
	rapidjson::Value frontVertices(rapidjson::kArrayType);
	rapidjson::Value backVertices(rapidjson::kArrayType);
	rapidjson::Value frontUvs(rapidjson::kArrayType);
	rapidjson::Value backUvs(rapidjson::kArrayType);
	for (size_t i = 0; i < 8; i++)
	{
		rapidjson::Value frontVertex(rapidjson::kObjectType);
		frontVertex.AddMember("x", rapidjson::Value().SetDouble(frontVerticesX[i]), allocator);
		frontVertex.AddMember("y", rapidjson::Value().SetDouble(frontVerticesY[i]), allocator);
		frontVertex.AddMember("z", rapidjson::Value().SetDouble(0), allocator);
		frontVertices.PushBack(frontVertex, allocator);

		rapidjson::Value frontUv(rapidjson::kObjectType);
		frontUv.AddMember("x", rapidjson::Value().SetDouble(frontVerticesX[i]/100), allocator);
		frontUv.AddMember("y", rapidjson::Value().SetDouble(frontVerticesY[i]/100), allocator);
		frontUvs.PushBack(frontUv, allocator);
		
		rapidjson::Value backVertex(rapidjson::kObjectType);
		backVertex.AddMember("x", rapidjson::Value().SetDouble(frontVerticesX[i]), allocator);
		backVertex.AddMember("y", rapidjson::Value().SetDouble(frontVerticesY[i]), allocator);
		backVertex.AddMember("z", rapidjson::Value().SetDouble(10), allocator);
		backVertices.PushBack(backVertex, allocator);

		rapidjson::Value backUv(rapidjson::kObjectType);
		backUv.AddMember("x", rapidjson::Value().SetDouble(frontVerticesX[i] / 100), allocator);
		backUv.AddMember("y", rapidjson::Value().SetDouble(frontVerticesY[i] / 100), allocator);
		backUvs.PushBack(backUv, allocator);
	}
	rapidjson::Value frontTriangles(rapidjson::kArrayType);
	rapidjson::Value backTriangles(rapidjson::kArrayType);
	rapidjson::Value frontNormals(rapidjson::kArrayType);
	rapidjson::Value backNormals(rapidjson::kArrayType);

	int triangles[24]{ 0,3,4,0,4,5,0,5,1,1,5,2,2,5,6,2,6,3,6,7,3,4,3,7 };
	for (size_t i = 0; i < 24; i++)
	{
		frontTriangles.PushBack(rapidjson::Value().SetInt(triangles[i]), allocator);
		backTriangles.PushBack(rapidjson::Value().SetInt(triangles[23-i]), allocator);
		rapidjson::Value frontNormalVector(rapidjson::kObjectType);
		rapidjson::Value backNormalVector(rapidjson::kObjectType);
		frontNormalVector.AddMember("x", rapidjson::Value().SetDouble(0), allocator);
		frontNormalVector.AddMember("y", rapidjson::Value().SetDouble(0), allocator);
		frontNormalVector.AddMember("z", rapidjson::Value().SetDouble(-1), allocator);
		backNormalVector.AddMember("x", rapidjson::Value().SetDouble(0), allocator);
		backNormalVector.AddMember("y", rapidjson::Value().SetDouble(0), allocator);
		backNormalVector.AddMember("z", rapidjson::Value().SetDouble(1), allocator);
		frontNormals.PushBack(frontNormalVector, allocator);
		backNormals.PushBack(backNormalVector, allocator);
	}

	frontMesh.AddMember("vertices", frontVertices, allocator);
	frontMesh.AddMember("triangles", frontTriangles, allocator);
	frontMesh.AddMember("normals", frontNormals, allocator);
	frontMesh.AddMember("uvs", frontUvs, allocator);
	frontMesh.AddMember("material", rapidjson::Value().SetString("plaster_blue_damaged"), allocator);

	backMesh.AddMember("vertices", backVertices, allocator);
	backMesh.AddMember("triangles", backTriangles, allocator);
	backMesh.AddMember("normals", backNormals, allocator);
	backMesh.AddMember("uvs", backUvs, allocator);
	backMesh.AddMember("material", rapidjson::Value().SetString("plaster_blue_damaged"), allocator);

	meshes.PushBack(frontMesh, allocator);
	meshes.PushBack(backMesh, allocator);
	wall.AddMember("meshes", meshes, allocator);
	walls.PushBack(wall, allocator);
	plot.AddMember("walls", walls, allocator);
	plot.AddMember("models", rapidjson::Value().SetArray(), allocator);
	plot.AddMember("type", rapidjson::Value().SetString("example"), allocator);
	plots.PushBack(plot, allocator);
	outputDoc.AddMember("plots", plots, allocator);
	return outputDoc;
}

int main()
{
	//// The number type to use for tessellation
	//using Coord = double;

	//// The index type. Defaults to uint32_t, but you can also pass uint16_t if you know that your
	//// data won't have more than 65536 vertices.
	//using N = uint32_t;

	//// Create array
	//using Point = std::array<Coord, 2>;
	//std::vector<std::vector<Point>> polygon;

	//rapidjson::Document document = readDocumentFromFile("TriangulationInput.json");
	//auto outlineJson = document["outline"].GetArray();
	//std::vector<Point> outline;
	//double x, y;
	//std::cout << "Outline\n";
	//for (size_t i = 0; i < outlineJson.Size(); i++)
	//{
	//	x = outlineJson[i]["x"].GetDouble();
	//	y = outlineJson[i]["y"].GetDouble();
	//	outline.push_back({ x,y });
	//	std::cout << x << " " << y << std::endl;
	//}
	//auto holeJson = document["hole"].GetArray();
	//std::vector<Point> hole;
	//std::cout << "Hole\n";
	//for (size_t i = 0; i < holeJson.Size(); i++)
	//{
	//	x = holeJson[i]["x"].GetDouble();
	//	y = holeJson[i]["y"].GetDouble();
	//	hole.push_back({ x,y });
	//	std::cout << x << " " << y << std::endl;
	//}

	//polygon.push_back(outline);
	//polygon.push_back(hole);

	////triangles are ccw
	//std::vector<N> triangles = mapbox::earcut<N>(polygon);
	//for (size_t i = 0; i < triangles.size(); i++)
	//{
	//	std::cout << triangles[i];
	//}
	//std::cout << std::endl;
	//outline.insert(outline.end(), hole.begin(), hole.end());

	
	const int duplicates = 200;
	const int noVertices = 6;
	int noVerticesInWallsBfr[duplicates+1];
	for (size_t i = 0; i < duplicates + 1; i++)
	{
		noVerticesInWallsBfr[i] = i * noVertices;
	}
	int* d_noVerticesInWallsBfr;
	cudaMalloc(&d_noVerticesInWallsBfr, sizeof(int) * (duplicates + 1));
	cudaMemcpy(d_noVerticesInWallsBfr, noVerticesInWallsBfr, sizeof(int) * (duplicates + 1), cudaMemcpyHostToDevice);

	//around 300 vertices for a block is too much for a shared memory
	//while 240 is fine

	const int noWallsForBlock = 40;
	const int noBlocks = (duplicates - 1) / noWallsForBlock + 1;
	int noWallsInBlocksBfr[noBlocks + 1];
	for (size_t i = 0; i < noBlocks + 1; i++)
	{
		noWallsInBlocksBfr[i] = i * noWallsForBlock;
	}
	int* d_noWallsInBlocksBfr;
	cudaMalloc(&d_noWallsInBlocksBfr, sizeof(int)*(noBlocks + 1));
	cudaMemcpy(d_noWallsInBlocksBfr, noWallsInBlocksBfr, sizeof(int)*(noBlocks + 1), cudaMemcpyHostToDevice);

	float3 verticesInWalls[duplicates * noVertices];
	for (size_t i = 0; i < duplicates * noVertices; i+=noVertices)
	{
		verticesInWalls[i].x = 0;
		verticesInWalls[i].y = 3;
		verticesInWalls[i + 1].x = 2;
		verticesInWalls[i + 1].y = 0;
		verticesInWalls[i + 2].x = 1;
		verticesInWalls[i + 2].y = 2;
		verticesInWalls[i + 3].x = 3;
		verticesInWalls[i + 3].y = 2;
		verticesInWalls[i + 4].x = 2;
		verticesInWalls[i + 4].y = 0;
		verticesInWalls[i + 5].x = 4;
		verticesInWalls[i + 5].y = 3;
	}
	float3* d_verticesInWalls;
	cudaMalloc(&d_verticesInWalls, sizeof(float3) * duplicates * noVertices);
	cudaMemcpy(d_verticesInWalls, verticesInWalls, sizeof(float3) * duplicates * noVertices, cudaMemcpyHostToDevice);
	int sizeOfSharedMemoryPerBlock = sizeof(float3) * noWallsForBlock * noVertices + sizeof(int) * noWallsForBlock * noVertices * 7 + sizeof(int) * noWallsForBlock * 2;
	int* d_triangles;
	int trianglesSize = 3 * (noVertices - 2) * duplicates;
	cudaMalloc(&d_triangles, sizeof(int) * trianglesSize);
	triangulatePolygon << <noBlocks, NO_THREADS, sizeOfSharedMemoryPerBlock >> > (duplicates, d_noVerticesInWallsBfr, d_noWallsInBlocksBfr, d_verticesInWalls, noVertices*noWallsForBlock, d_triangles);
	int* triangles = (int*)malloc(sizeof(int) * trianglesSize);
	cudaMemcpy(triangles, d_triangles, sizeof(int) * trianglesSize, cudaMemcpyDeviceToHost);
	printf("Triangles:\n");
	for (size_t i = 0; i < trianglesSize; i+=3)
	{
		printf("%d %d %d\n", triangles[i], triangles[i + 1], triangles[i + 2]);
	}
	//rapidjson::Document& example = createExample();
	//writeDocumentToFile(example, "TriangulationOutput.json");


	return 0;
}