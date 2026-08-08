// M0 hello-world: load a Gmsh .msh v4.1 mesh with MFEM, report per-attribute
// volumes, and probe the requested device (cpu / cuda).
//
// Usage: hello_mfem <mesh.msh> [device]

#include "mfem.hpp"

#include <cstdio>
#include <map>

int main(int argc, char *argv[])
{
   if (argc < 2)
   {
      std::fprintf(stderr, "usage: %s <mesh.msh> [device]\n", argv[0]);
      return 1;
   }
   const char *device_config = (argc > 2) ? argv[2] : "cpu";

   mfem::Device device(device_config);
   device.Print();

   mfem::Mesh mesh(argv[1], 1, 1);
   std::printf("dimension: %d\n", mesh.Dimension());
   std::printf("elements: %d\n", mesh.GetNE());
   std::printf("vertices: %d\n", mesh.GetNV());
   std::printf("boundary elements: %d\n", mesh.GetNBE());

   std::map<int, double> vol_by_attr;
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      vol_by_attr[mesh.GetAttribute(i)] += mesh.GetElementVolume(i);
   }
   for (const auto &kv : vol_by_attr)
   {
      std::printf("attribute %d volume: %.15e\n", kv.first, kv.second);
   }
   std::printf("hello_mfem: OK\n");
   return 0;
}
