// Link stub for one MFEM translation unit whose nvcc compilation is
// pathologically slow on this machine (fem/integ/bilininteg_convection_ea.cpp,
// 12h+ then killed; permanently abandoned by design decision) and whose
// functionality LithoFEM never uses (element-assembly convection kernels).
// The symbol is referenced only through the ConvectionIntegrator vtable, so
// the stub is needed to link but is never called; if it ever were reached it
// aborts loudly. The Makefile drops this file automatically if a real
// bilininteg_convection_ea.o is ever archived into libmfem.a.

#include "mfem.hpp"

namespace mfem
{

void ConvectionIntegrator::AssembleEA(const FiniteElementSpace &, Vector &,
                                      const bool)
{
   MFEM_ABORT("stub: MFEM built without bilininteg_convection_ea.o");
}

} // namespace mfem
