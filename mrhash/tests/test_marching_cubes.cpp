#include <sdf/marching_cubes.cuh>

#include "test_utils.cuh"

using namespace cupanutils::cugeoutils;

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(COMBINE, simple_combine) {
  // generate two simple meshes
  Eigen::MatrixXd vertices1(4, 3);
  vertices1 << 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0;
  Eigen::MatrixXi faces1(2, 3);
  faces1 << 0, 1, 2, 1, 3, 2;

  Eigen::MatrixXd vertices2(4, 3);
  vertices2 << 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1;
  Eigen::MatrixXi faces2(2, 3);
  faces2 << 0, 1, 2, 1, 3, 2;

  Eigen::MatrixXd combinedVertices;
  Eigen::MatrixXi combinedFaces;
  GeometricMarchingCubes mesh_extractor(0.f, false, 0, 0.f);
  mesh_extractor.combine({vertices1, vertices2}, {faces1, faces2}, combinedVertices, combinedFaces);

  ASSERT_EQ(combinedVertices.rows(), 8);
  ASSERT_EQ(combinedVertices.cols(), 3);
  for (int i = 0; i < 4; ++i) {
    ASSERT_FLOAT_EQ(combinedVertices(i, 0), vertices1(i, 0));
    ASSERT_FLOAT_EQ(combinedVertices(i, 1), vertices1(i, 1));
    ASSERT_FLOAT_EQ(combinedVertices(i, 2), vertices1(i, 2));
    ASSERT_FLOAT_EQ(combinedVertices(i + 4, 0), vertices2(i, 0));
    ASSERT_FLOAT_EQ(combinedVertices(i + 4, 1), vertices2(i, 1));
    ASSERT_FLOAT_EQ(combinedVertices(i + 4, 2), vertices2(i, 2));
  }
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(combinedFaces(i, 0), faces1(i, 0));
    ASSERT_EQ(combinedFaces(i, 1), faces1(i, 1));
    ASSERT_EQ(combinedFaces(i, 2), faces1(i, 2));
    ASSERT_EQ(combinedFaces(i + 2, 0), faces2(i, 0) + 4); // adjust indices
    ASSERT_EQ(combinedFaces(i + 2, 1), faces2(i, 1) + 4);
    ASSERT_EQ(combinedFaces(i + 2, 2), faces2(i, 2) + 4);
  }
}

TEST(COMBINE, single_combine) {
  Eigen::MatrixXd vertices1(4, 3);
  vertices1 << 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0;
  Eigen::MatrixXi faces1(2, 3);
  faces1 << 0, 1, 2, 1, 3, 2;

  Eigen::MatrixXd vertices2(0, 3);
  Eigen::MatrixXi faces2(0, 3);

  Eigen::MatrixXd combinedVertices;
  Eigen::MatrixXi combinedFaces;

  GeometricMarchingCubes mesh_extractor(0.f, false, 0, 0.f);
  mesh_extractor.combine({vertices1, vertices2}, {faces1, faces2}, combinedVertices, combinedFaces);

  ASSERT_EQ(combinedVertices.rows(), 4);
  ASSERT_EQ(combinedVertices.cols(), 3);
  for (int i = 0; i < 4; ++i) {
    ASSERT_FLOAT_EQ(combinedVertices(i, 0), vertices1(i, 0));
    ASSERT_FLOAT_EQ(combinedVertices(i, 1), vertices1(i, 1));
    ASSERT_FLOAT_EQ(combinedVertices(i, 2), vertices1(i, 2));
  }
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(combinedFaces(i, 0), faces1(i, 0));
    ASSERT_EQ(combinedFaces(i, 1), faces1(i, 1));
    ASSERT_EQ(combinedFaces(i, 2), faces1(i, 2));
  }
}

TEST(COMBINE, empty_lists) {
  Eigen::MatrixXd combinedVertices;
  Eigen::MatrixXi combinedFaces;

  GeometricMarchingCubes mesh_extractor(0.f, false, 0, 0.f);
  mesh_extractor.combine({}, {}, combinedVertices, combinedFaces);

  ASSERT_EQ(combinedVertices.rows(), 0);
  ASSERT_EQ(combinedVertices.cols(), 0);
  ASSERT_EQ(combinedFaces.rows(), 0);
  ASSERT_EQ(combinedFaces.cols(), 0);
}

TEST(REMOVE_DUPL_FACES, duplicates) {
  Eigen::MatrixXi faces(4, 3);
  faces << 0, 1, 2, 1, 2, 3, 0, 1, 2, 1, 2, 3;

  Eigen::MatrixXi uniqueFaces;

  GeometricMarchingCubes mesh_extractor(0.f, false, 0, 0.f);
  mesh_extractor.removeDuplicateFacesTriangle(faces, uniqueFaces);

  ASSERT_EQ(uniqueFaces.rows(), faces.rows() - 2);
}

TEST(REMOVE_DUPL_FACES, no_duplicates) {
  Eigen::MatrixXi faces(3, 3);
  faces << 0, 1, 2, 1, 2, 3, 0, 2, 3;

  Eigen::MatrixXi uniqueFaces;

  GeometricMarchingCubes mesh_extractor(0.f, false, 0, 0.f);
  mesh_extractor.removeDuplicateFacesTriangle(faces, uniqueFaces);

  ASSERT_EQ(uniqueFaces.rows(), faces.rows());
}

TEST(REMOVE_DUPL_FACES, empty) {
  Eigen::MatrixXi faces(0, 3);
  Eigen::MatrixXi uniqueFaces;

  GeometricMarchingCubes mesh_extractor(0.f, false, 0, 0.f);
  mesh_extractor.removeDuplicateFacesTriangle(faces, uniqueFaces);

  ASSERT_EQ(uniqueFaces.rows(), 0);
  ASSERT_EQ(uniqueFaces.cols(), 3);
}

TEST(REMOVE_DUPL_VERTICES, basic_zero) {
  Eigen::MatrixXd vertices(6, 3);
  vertices << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0;

  Eigen::MatrixXi faces(2, 3);
  faces << 0, 1, 2, 1, 3, 2;

  Eigen::MatrixXd uniqueVertices;
  Eigen::MatrixXi updatedFaces;
  Eigen::VectorXi oldToNewMapping;

  GeometricMarchingCubes mesh_extractor(0.f, false, 0, 0.f);
  mesh_extractor.removeDuplicateVerticesTriangle(vertices, faces, 0, uniqueVertices, updatedFaces, oldToNewMapping);

  ASSERT_EQ(uniqueVertices.rows(), 3);
  ASSERT_EQ(uniqueVertices.cols(), 3);
  ASSERT_EQ(oldToNewMapping.rows(), vertices.rows());
  ASSERT_EQ(oldToNewMapping.cols(), 1);
  ASSERT_EQ(updatedFaces.rows(), faces.rows());
  ASSERT_EQ(updatedFaces.cols(), faces.cols());

  for (int i = 0; i < 3; ++i) {
    ASSERT_NEAR(uniqueVertices(i, 0), vertices(i, 0), 1e-12);
    ASSERT_NEAR(uniqueVertices(i, 1), vertices(i, 1), 1e-12);
    ASSERT_NEAR(uniqueVertices(i, 2), vertices(i, 2), 1e-12);
  }

  for (int i = 0; i < vertices.rows(); ++i) {
    int mappedIndex = oldToNewMapping(i, 0);
    ASSERT_GE(mappedIndex, 0);
    ASSERT_LT(mappedIndex, uniqueVertices.rows());
    ASSERT_NEAR(uniqueVertices(mappedIndex, 0), vertices(i, 0), 1e-12);
    ASSERT_NEAR(uniqueVertices(mappedIndex, 1), vertices(i, 1), 1e-12);
    ASSERT_NEAR(uniqueVertices(mappedIndex, 2), vertices(i, 2), 1e-12);
  }

  for (int i = 0; i < faces.rows(); ++i) {
    for (int j = 0; j < faces.cols(); ++j) {
      int oldIdx = faces(i, j);
      int newIdx = updatedFaces(i, j);
      ASSERT_EQ(newIdx, oldToNewMapping(oldIdx));
    }
  }
}

TEST(REMOVE_DUPL_VERTICES, basic_nonzero) {
  Eigen::MatrixXd vertices(6, 3);
  vertices << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0;

  Eigen::MatrixXi faces(2, 3);
  faces << 0, 1, 2, 1, 3, 2;

  Eigen::MatrixXd uniqueVertices;
  Eigen::MatrixXi updatedFaces;
  Eigen::VectorXi oldToNewMapping;

  GeometricMarchingCubes mesh_extractor(0.f, false, 0, 0.f);
  mesh_extractor.removeDuplicateVerticesTriangle(vertices, faces, 1e-12, uniqueVertices, updatedFaces, oldToNewMapping);

  ASSERT_EQ(uniqueVertices.rows(), 3);
  ASSERT_EQ(uniqueVertices.cols(), 3);
  ASSERT_EQ(oldToNewMapping.rows(), vertices.rows());
  ASSERT_EQ(oldToNewMapping.cols(), 1);
  ASSERT_EQ(updatedFaces.rows(), faces.rows());
  ASSERT_EQ(updatedFaces.cols(), faces.cols());

  for (int i = 0; i < 3; ++i) {
    ASSERT_NEAR(uniqueVertices(i, 0), vertices(i, 0), 1e-12);
    ASSERT_NEAR(uniqueVertices(i, 1), vertices(i, 1), 1e-12);
    ASSERT_NEAR(uniqueVertices(i, 2), vertices(i, 2), 1e-12);
  }

  for (int i = 0; i < vertices.rows(); ++i) {
    int mappedIndex = oldToNewMapping(i, 0);
    ASSERT_GE(mappedIndex, 0);
    ASSERT_LT(mappedIndex, uniqueVertices.rows());
    ASSERT_NEAR(uniqueVertices(mappedIndex, 0), vertices(i, 0), 1e-12);
    ASSERT_NEAR(uniqueVertices(mappedIndex, 1), vertices(i, 1), 1e-12);
    ASSERT_NEAR(uniqueVertices(mappedIndex, 2), vertices(i, 2), 1e-12);
  }

  for (int i = 0; i < faces.rows(); ++i) {
    for (int j = 0; j < faces.cols(); ++j) {
      int oldIdx = faces(i, j);
      int newIdx = updatedFaces(i, j);
      ASSERT_EQ(newIdx, oldToNewMapping(oldIdx));
    }
  }
}

TEST(REMOVE_DUPL_VERTICES, empty_vertices) {
  Eigen::MatrixXd vertices(0, 3);
  Eigen::MatrixXi faces(0, 3);

  Eigen::MatrixXd uniqueVertices;
  Eigen::MatrixXi updatedFaces;
  Eigen::VectorXi oldToNewMapping;

  GeometricMarchingCubes mesh_extractor(0.f, false, 0, 0.f);
  mesh_extractor.removeDuplicateVerticesTriangle(vertices, faces, 1e-12, uniqueVertices, updatedFaces, oldToNewMapping);

  ASSERT_EQ(uniqueVertices.rows(), 0);
  ASSERT_EQ(uniqueVertices.cols(), 3);
  ASSERT_EQ(oldToNewMapping.rows(), 0);
  ASSERT_EQ(oldToNewMapping.cols(), 1);
  ASSERT_EQ(updatedFaces.rows(), 0);
  ASSERT_EQ(updatedFaces.cols(), 3);
}

TEST(REMOVE_DUPL_VERTICES, no_duplicates) {
  Eigen::MatrixXd vertices(3, 3);
  vertices << 0, 0, 0, 1, 0, 0, 0, 1, 0;

  Eigen::MatrixXi faces(1, 3);
  faces << 0, 1, 2;

  Eigen::MatrixXd uniqueVertices;
  Eigen::MatrixXi updatedFaces;
  Eigen::VectorXi oldToNewMapping;

  GeometricMarchingCubes mesh_extractor(0.f, false, 0, 0.f);
  mesh_extractor.removeDuplicateVerticesTriangle(vertices, faces, 1e-12, uniqueVertices, updatedFaces, oldToNewMapping);

  ASSERT_EQ(uniqueVertices.rows(), 3);
  ASSERT_EQ(uniqueVertices.cols(), 3);
  ASSERT_EQ(oldToNewMapping.rows(), vertices.rows());
  ASSERT_EQ(oldToNewMapping.cols(), 1);
  ASSERT_EQ(updatedFaces.rows(), faces.rows());
  ASSERT_EQ(updatedFaces.cols(), faces.cols());

  for (int i = 0; i < vertices.rows(); ++i) {
    int mappedIndex = oldToNewMapping(i, 0);
    ASSERT_EQ(mappedIndex, i);
    ASSERT_NEAR(uniqueVertices(i, 0), vertices(i, 0), 1e-12);
    ASSERT_NEAR(uniqueVertices(i, 1), vertices(i, 1), 1e-12);
    ASSERT_NEAR(uniqueVertices(i, 2), vertices(i, 2), 1e-12);
  }

  for (int i = 0; i < faces.rows(); ++i) {
    for (int j = 0; j < faces.cols(); ++j) {
      int oldIdx = faces(i, j);
      int newIdx = updatedFaces(i, j);
      ASSERT_EQ(newIdx, oldToNewMapping(oldIdx));
    }
  }
}
