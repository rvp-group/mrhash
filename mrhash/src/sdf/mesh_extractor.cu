#include "mesh_extractor.cuh"

namespace cupanutils {
  namespace cugeoutils {
    template <typename T>
    __device__ Vertex MeshExtractor<T>::vertexInterp(const float isolevel,
                                                     const float3& p1,
                                                     const float3& p2,
                                                     const float d1,
                                                     const float d2,
                                                     const uchar3& c1,
                                                     const uchar3& c2) {
      Vertex r1;
      r1.p = p1;
      r1.c = make_float3(c1.x, c1.y, c1.z) / 255.f;
      Vertex r2;
      r2.p = p2;
      r2.c = make_float3(c2.x, c2.y, c2.z) / 255.f;
      if (abs(isolevel - d1) < 0.00001f)
        return r1;
      if (abs(isolevel - d2) < 0.00001f)
        return r2;
      if (abs(d1 - d2) < 0.00001f)
        return r1;
      const float mu = (isolevel - d1) / (d2 - d1);

      Vertex res;
      res.p.x = p1.x + mu * (p2.x - p1.x);
      res.p.y = p1.y + mu * (p2.y - p1.y);
      res.p.z = p1.z + mu * (p2.z - p1.z);

      res.c.x = c1.x + mu * (c2.x - c1.x) / 255.f;
      res.c.y = c1.y + mu * (c2.y - c1.y) / 255.f;
      res.c.z = c1.z + mu * (c2.z - c1.z) / 255.f;
      return res;
    }

    template <typename T>
    __device__ uint MeshExtractor<T>::append() {
      const uint addr = atomicAdd(&d_trianglesCounter_[0], 1);
      return addr;
    }

    template <typename T>
    __device__ void MeshExtractor<T>::appendTriangle(const Triangle& triangle) {
      const uint addr = append();
      if (addr >= max_num_triangles_mesh_) {
        printf("appendTriangle | exceeded max triangles: %u >= %u\n", addr, max_num_triangles_mesh_);
        return;
      }
      Triangle& d_triangle = d_triangles_[addr];
      d_triangle.v0        = triangle.v0;
      d_triangle.v1        = triangle.v1;
      d_triangle.v2        = triangle.v2;
    }
  } // namespace cugeoutils
} // namespace cupanutils
