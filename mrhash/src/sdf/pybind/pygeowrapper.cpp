#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/string.h>

#include <geowrapper.h>
#include <serializer.h>

using namespace pygeowrapper;

NB_MODULE(pygeowrapper, m) {
  nb::class_<GeoWrapper>(m, "GeoWrapper")
    .def(nb::init<float, float, int, float, int, int, bool, float, uchar, float, float, std::string, float, float, bool>(),
         nb::arg("sdf_truncation"),
         nb::arg("sdf_truncation_scale"),
         nb::arg("integration_weight_sample"),
         nb::arg("virtual_voxel_size"),
         nb::arg("n_frames_invalidate_voxels"),
         nb::arg("voxel_extents_scale"),
         nb::arg("viewer_active"),
         nb::arg("marching_cubes_threshold"),
         nb::arg("min_weight_threshold"),
         nb::arg("min_depth"),
         nb::arg("max_depth"),
         nb::arg("gs_optimization_param_path") = default_gs_optimization_param_path,
         nb::arg("sdf_var_threshold")          = default_sdf_var_threshold,
         nb::arg("vertices_merging_threshold") = default_vertices_merging_threshold,
         nb::arg("projective_sdf")             = default_projective_sdf)

    // getters
    .def("getHashNumBuckets", &GeoWrapper::getHashNumBuckets)
    .def("getNumSdfBlocks", &GeoWrapper::getNumSdfBlocks)
    .def("getHashBucketSize", &GeoWrapper::getHashBucketSize)
    .def("getSdfTruncation", &GeoWrapper::getSdfTruncation)
    .def("getSdfTruncationScale", &GeoWrapper::getSdfTruncationScale)
    .def("getIntegrationWeightSample", &GeoWrapper::getIntegrationWeightSample)
    .def("getIntegrationWeightMax", &GeoWrapper::getIntegrationWeightMax)
    .def("getVirtualVoxelSize", &GeoWrapper::getVirtualVoxelSize)
    .def("getLinkedListSize", &GeoWrapper::getLinkedListSize)
    .def("getNFramesInvalidateVoxels", &GeoWrapper::getNFramesInvalidateVoxels)
    .def("getMaxNumSdfBlockIntegrateFromGlobalHash", &GeoWrapper::getMaxNumSdfBlockIntegrateFromGlobalHash)
    .def("getVoxelExtentsScale", &GeoWrapper::getVoxelExtentsScale)
    .def("getCurrPose", &GeoWrapper::getCurrPose)
    .def("getPointCloud", &GeoWrapper::getPointCloud)
    .def("getNormals", &GeoWrapper::getNormals)
    .def("getVertices", &GeoWrapper::getVertices)
    .def("getFaces", &GeoWrapper::getFaces)
    .def("getColors", &GeoWrapper::getColors)

    // setters
    .def("setHashNumBuckets", &GeoWrapper::setHashNumBuckets)
    .def("setNumSdfBlocks", &GeoWrapper::setNumSdfBlocks)
    .def("setHashBucketSize", &GeoWrapper::setHashBucketSize)
    .def("setSdfTruncation", &GeoWrapper::setSdfTruncation)
    .def("setSdfTruncationScale", &GeoWrapper::setSdfTruncationScale)
    .def("setIntegrationWeightSample", &GeoWrapper::setIntegrationWeightSample)
    .def("setIntegrationWeightMax", &GeoWrapper::setIntegrationWeightMax)
    .def("setVirtualVoxelSize", &GeoWrapper::setVirtualVoxelSize)
    .def("setLinkedListSize", &GeoWrapper::setLinkedListSize)
    .def("setNFramesInvalidateVoxels", &GeoWrapper::setNFramesInvalidateVoxels)
    .def("setMaxNumSdfBlockIntegrateFromGlobalHash", &GeoWrapper::setMaxNumSdfBlockIntegrateFromGlobalHash)
    .def("setVoxelExtentsScale", &GeoWrapper::setVoxelExtentsScale)
    .def("setRGBImage", nb::overload_cast<nb::ndarray<uint8_t>>(&GeoWrapper::setRGBImage))
    .def("setDepthImage", nb::overload_cast<nb::ndarray<float>>(&GeoWrapper::setDepthImage))
    .def("setPointCloud", nb::overload_cast<nb::ndarray<float>, bool>(&GeoWrapper::setPointCloud))
    .def("setPointCloud", nb::overload_cast<nb::ndarray<float>, nb::ndarray<float>>(&GeoWrapper::setPointCloud))
    .def("setCamera", &GeoWrapper::setCamera)
    .def("setCurrPose", &GeoWrapper::setCurrPose)
    .def("setCameraInLidar", &GeoWrapper::setCameraInLidar)
    .def("compute", (&GeoWrapper::compute))
    .def("extractMesh", &GeoWrapper::extractMesh)
    .def("GSSavePointCloud", &GeoWrapper::GSSavePointCloud)
    .def("GSFinalOpt", &GeoWrapper::GSFinalOpt)
    .def("streamAllOut", &GeoWrapper::streamAllOut)
    .def("clearBuffers", &GeoWrapper::clearBuffers)
    .def("serializeData",
         &GeoWrapper::serializeData,
         nb::arg("filename_hash")  = "./data/hash_points.ply",
         nb::arg("filename_voxel") = "./data/voxel_points.ply")
    .def("serializeGrid", &GeoWrapper::serializeGrid, nb::arg("filename") = cupanutils::cugeoutils::default_serializer_filename)
    .def(
      "deserializeGrid", &GeoWrapper::deserializeGrid, nb::arg("filename") = cupanutils::cugeoutils::default_serializer_filename);
}
