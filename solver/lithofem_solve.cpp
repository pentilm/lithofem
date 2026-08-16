// LithoFEM production solver core (M6): scattered-field formulation with
// Bloch periodicity via the envelope substitution E_sc = u e^{i kpar.r},
// z-PML as material tensor Lambda, per-region complex eps, TMM incident
// tables from solve.json (evaluated exactly), UMFPACK complex direct solve.
//
// Inputs:  -m mesh.per.msh  -j solve.json  -o outdir  [-g group]
// Outputs (per group g):
//   outdir/plane_g<g>_p<k>.bin : row-major (ny, nx, 3, 2) float64 samples of
//       the ENVELOPE u of E_sc on the requested z-plane (re/im interleaved)
//   outdir/solve_meta_g<g>.json : residual, per-region |u|^2 integrals, dofs
//
// The total physical field is reconstructed in Python:
//   E_sc = u e^{i kpar.r},  E_total = E_sc + E_inc.

#include "mfem.hpp"
#include "thirdparty/json.hpp"

#include <cholmod.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <vector>

#ifdef LITHOFEM_HAVE_CUDSS
#include <cuda_runtime.h>
#include <cudss.h>
#endif

using namespace mfem;
using json = nlohmann::json;
using C = std::complex<double>;

namespace
{

// v2.5 segment timing (M1 profile): wall-clock seconds per solver stage,
// printed as `timing_<seg>_s` lines and written into solve_meta.
using SegClock = std::chrono::steady_clock;
SegClock::time_point seg_t0;
std::vector<std::pair<std::string, double>> seg_times;
void seg_mark(const char *name)
{
   const auto now = SegClock::now();
   seg_times.emplace_back(
      name, std::chrono::duration<double>(now - seg_t0).count());
   std::printf("timing_%s_s %.3f\n", name, seg_times.back().second);
   seg_t0 = now;
}

#include "problem_common.inc"


// ---- local current sources (M6b) --------------------------------------
// weak form contribution: + i k0 (Z0 J) . conj(v) with the envelope phase
// e^{-i kpar.r} and any user phase gradient; point = basis point value,
// line = 1D quadrature, sheet = conforming-face quadrature.

bool find_point(Mesh &mesh, const double x[3], int &elem, IntegrationPoint &ip)
{
   DenseMatrix pts(3, 1);
   const double oxs[3] = {0.0, -prob.lx, prob.lx};
   const double oys[3] = {0.0, -prob.ly, prob.ly};
   for (int im = 0; im < 9; ++im)
   {
      pts(0, 0) = x[0] + oxs[im % 3];
      pts(1, 0) = x[1] + oys[im / 3];
      pts(2, 0) = x[2];
      Array<int> e1;
      Array<IntegrationPoint> ip1;
      mesh.FindPoints(pts, e1, ip1, im == 0);
      if (e1[0] >= 0)
      {
         elem = e1[0];
         ip = ip1[0];
         return true;
      }
   }
   return false;
}

void accumulate_at(FiniteElementSpace &fes, Mesh &mesh, const double x[3],
                   const C amp[3], LinearForm &br, LinearForm &bi)
{
   int elem;
   IntegrationPoint ip;
   MFEM_VERIFY(find_point(mesh, x, elem, ip), "local source point not found");
   const FiniteElement *fe = fes.GetFE(elem);
   ElementTransformation *tr = mesh.GetElementTransformation(elem);
   tr->SetIntPoint(&ip);
   DenseMatrix vshape(fe->GetDof(), 3);
   fe->CalcVShape(*tr, vshape);
   Array<int> dofs;
   DofTransformation *dt = fes.GetElementDofs(elem, dofs);
   Vector vr(fe->GetDof()), vi(fe->GetDof());
   for (int k = 0; k < fe->GetDof(); ++k)
   {
      C val(0.0, 0.0);
      for (int d = 0; d < 3; ++d) { val += vshape(k, d) * amp[d]; }
      vr(k) = val.real();
      vi(k) = val.imag();
   }
   if (dt)
   {
      dt->TransformDual(vr);
      dt->TransformDual(vi);
   }
   br.AddElementVector(dofs, vr);
   bi.AddElementVector(dofs, vi);
}

C bloch_phase(double x, double y)
{
   return std::exp(C(0.0, -(prob.kx * x + prob.ky * y)));
}

void add_local_sources(FiniteElementSpace &fes, Mesh &mesh,
                       LinearForm &br, LinearForm &bi)
{
   // Under MFEM device runs (-d cuda) Assemble() leaves the vectors valid on
   // the device only; the element-wise writes below happen host-side. Pull
   // to host and invalidate the device copies, or these contributions are
   // silently lost when FormLinearSystem reads the device data (found by
   // the V2-M2 GPU smoke test; v1 never ran local sources with -d cuda).
   br.HostReadWrite();
   bi.HostReadWrite();
   const C ik0(0.0, prob.k0);
   for (size_t li = 0; li < prob.local.size(); ++li)
   {
      if (!prob.active_local.empty() &&
          std::find(prob.active_local.begin(), prob.active_local.end(),
                    (int)li) == prob.active_local.end())
      {
         continue;
      }
      const LocalSource &ls = prob.local[li];
      if (ls.type == 0) // point dipole
      {
         C amp[3];
         const C ph = bloch_phase(ls.p0[0], ls.p0[1]);
         for (int d = 0; d < 3; ++d) { amp[d] = ik0 * ls.cur[d] * ph; }
         accumulate_at(fes, mesh, ls.p0, amp, br, bi);
      }
      else if (ls.type == 1) // line current, composite midpoint rule
      {
         double dir[3], len = 0.0;
         for (int d = 0; d < 3; ++d)
         {
            dir[d] = ls.p1[d] - ls.p0[d];
            len += dir[d] * dir[d];
         }
         len = std::sqrt(len);
         const int nq = std::max(200, (int)(len / 0.05));
         const double w = len / nq;
         for (int q = 0; q < nq; ++q)
         {
            const double t = (q + 0.5) / nq;
            double x[3];
            for (int d = 0; d < 3; ++d) { x[d] = ls.p0[d] + t * dir[d]; }
            const C ph = bloch_phase(x[0], x[1]) *
                         std::exp(C(0.0, ls.pg[0] * t * len));
            C amp[3];
            for (int d = 0; d < 3; ++d) { amp[d] = ik0 * ls.cur[d] * ph * w; }
            accumulate_at(fes, mesh, x, amp, br, bi);
         }
      }
      else // horizontal sheet: conforming-face Gauss quadrature
      {
         const double z0 = ls.p0[2];
         // in-plane parametrization: r = corner + s1 e1 + s2 e2
         const double e1n = std::hypot(ls.p1[0], ls.p1[1]);
         const double e2n = std::hypot(ls.e2[0], ls.e2[1]);
         const double det = ls.p1[0] * ls.e2[1] - ls.p1[1] * ls.e2[0];
         MFEM_VERIFY(std::abs(det) > 1e-12, "degenerate sheet");
         int order = 2 * prob.order + 2;
         int n_faces = 0;
         for (int f = 0; f < mesh.GetNumFaces(); ++f)
         {
            FaceElementTransformations *T =
               mesh.GetInteriorFaceTransformations(f);
            if (!T) { continue; }
            // face on the sheet plane?
            {
               const IntegrationRule &ir0 = IntRules.Get(T->GetGeometryType(), 1);
               Vector xc(3);
               T->Transform(ir0.IntPoint(0), xc);
               if (std::abs(xc(2) - z0) > 1e-7) { continue; }
               // inside the rectangle (wrapped into the cell)?
               double rx = xc(0) - ls.p0[0], ry = xc(1) - ls.p0[1];
               double s1 = (rx * ls.e2[1] - ry * ls.e2[0]) / det;
               double s2 = (ls.p1[0] * ry - ls.p1[1] * rx) / det;
               if (s1 < -1e-9 || s1 > 1 + 1e-9 || s2 < -1e-9 || s2 > 1 + 1e-9)
               {
                  continue;
               }
            }
            ++n_faces;
            const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(), order);
            const FiniteElement *fe1 = fes.GetFE(T->Elem1No);
            Array<int> dofs;
            DofTransformation *dt = fes.GetElementDofs(T->Elem1No, dofs);
            Vector vr(fe1->GetDof()), vi(fe1->GetDof());
            vr = 0.0;
            vi = 0.0;
            DenseMatrix vshape(fe1->GetDof(), 3);
            for (int q = 0; q < ir.GetNPoints(); ++q)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               T->SetAllIntPoints(&ip);
               Vector x(3);
               T->Transform(ip, x);
               const double rx = x(0) - ls.p0[0], ry = x(1) - ls.p0[1];
               const double s1 = (rx * ls.e2[1] - ry * ls.e2[0]) / det;
               const double s2 = (ls.p1[0] * ry - ls.p1[1] * rx) / det;
               const C ph = bloch_phase(x(0), x(1)) *
                            std::exp(C(0.0, ls.pg[0] * s1 * e1n +
                                            ls.pg[1] * s2 * e2n));
               ElementTransformation &etr = T->GetElement1Transformation();
               fe1->CalcVShape(etr, vshape);
               const double w = ip.weight * T->Weight();
               for (int k = 0; k < fe1->GetDof(); ++k)
               {
                  C val(0.0, 0.0);
                  for (int d = 0; d < 3; ++d)
                  {
                     val += vshape(k, d) * (ik0 * ls.cur[d] * ph);
                  }
                  vr(k) += w * val.real();
                  vi(k) += w * val.imag();
               }
            }
            if (dt)
            {
               dt->TransformDual(vr);
               dt->TransformDual(vi);
            }
            br.AddElementVector(dofs, vr);
            bi.AddElementVector(dofs, vi);
         }
         MFEM_VERIFY(n_faces > 0, "sheet source found no conforming faces");
      }
   }
}


// ---- M9: system export/import + memory estimate ------------------------

void export_system(const ComplexSparseMatrix &Ac, const Vector &B,
                   const std::string &prefix)
{
   const SparseMatrix &Ar = Ac.real();
   const SparseMatrix &Ai = Ac.imag();
   const int n = Ar.Height();
   const int *I = Ar.GetI();
   const int *J = Ar.GetJ();
   const double *vr = Ar.GetData();
   const double *vi = Ai.GetData();
   {
      std::ofstream f(prefix + ".mtx");
      f << "%%MatrixMarket matrix coordinate complex general\n";
      f << n << " " << n << " " << Ar.NumNonZeroElems() << "\n";
      f.precision(17);
      for (int r = 0; r < n; ++r)
      {
         for (int k = I[r]; k < I[r + 1]; ++k)
         {
            f << r + 1 << " " << J[k] + 1 << " " << vr[k] << " " << vi[k]
              << "\n";
         }
      }
   }
   {
      std::ofstream f(prefix + ".rhs.mtx");
      f << "%%MatrixMarket matrix array complex general\n";
      f << n << " 1\n";
      f.precision(17);
      for (int r = 0; r < n; ++r)
      {
         f << B(r) << " " << B(n + r) << "\n";
      }
   }
}

void import_vector(const std::string &path, Vector &X, int n)
{
   std::ifstream f(path);
   MFEM_VERIFY(f.good(), "cannot open imported solution");
   std::string line;
   while (std::getline(f, line))
   {
      if (!line.empty() && line[0] != '%') { break; }
   }
   // `line` now holds the size header "n 1"
   for (int r = 0; r < n; ++r)
   {
      double re, im;
      MFEM_VERIFY(bool(f >> re >> im), "short imported solution file");
      X(r) = re;
      X(n + r) = im;
   }
}

double umfpack_symbolic_estimate_gb(const ComplexSparseMatrix &Ac)
{
   // Pre-factorization memory estimate. UMFPACK's own symbolic bound assumes
   // worst-case off-diagonal pivoting and overshoots complex curl-curl
   // systems by >10x. The pattern is structurally symmetric and diagonal
   // pivoting succeeds in practice (measured lnz == unz), so the CHOLMOD
   // symbolic Cholesky fill of the pattern is the right predictor:
   //   bytes ~= 2 * lnz * 16 (complex L and U) * 1.25 working overhead.
   const SparseMatrix &Ar = Ac.real();
   const SuiteSparse_long n = Ar.Height();
   const int *I = Ar.GetI();
   const int *J = Ar.GetJ();
   const SuiteSparse_long nnz = Ar.NumNonZeroElems();
   std::vector<SuiteSparse_long> Ap(n + 1), Aj(nnz);
   for (SuiteSparse_long r = 0; r <= n; ++r) { Ap[r] = I[r]; }
   for (SuiteSparse_long k = 0; k < nnz; ++k) { Aj[k] = J[k]; }
   for (SuiteSparse_long r = 0; r < n; ++r)
   {
      std::sort(Aj.begin() + Ap[r], Aj.begin() + Ap[r + 1]);
   }

   cholmod_common cm;
   cholmod_l_start(&cm);
   cholmod_sparse A;
   std::memset(&A, 0, sizeof(A));
   A.nrow = (size_t)n;
   A.ncol = (size_t)n;
   A.nzmax = (size_t)nnz;
   A.p = Ap.data();
   A.i = Aj.data();
   A.stype = 1;                  // use the upper-triangular part
   A.itype = CHOLMOD_LONG;
   A.xtype = CHOLMOD_PATTERN;
   A.dtype = CHOLMOD_DOUBLE;
   A.sorted = 1;
   A.packed = 1;
   cholmod_factor *L = cholmod_l_analyze(&A, &cm);
   const double lnz = cm.lnz;
   if (L) { cholmod_l_free_factor(&L, &cm); }
   cholmod_l_finish(&cm);
   return 2.0 * lnz * 16.0 * 1.25 / 1e9;
}


// ---- docs/physics.md thin internal interfaces (soft contract) ---------------
// IAssembler: mesh + parameters -> (complex operator, eliminated X/B).
// ILinearSolver: operator + RHS -> solution. MFEM/UMFPACK are the default
// implementations; replacing either only requires honoring these calls.

class ILinearSolver
{
public:
   virtual ~ILinearSolver() = default;
   virtual bool Solve(ComplexSparseMatrix &A, const Vector &B, Vector &X) = 0;
   virtual const char *Name() const = 0;
};

class UmfpackDirectSolver : public ILinearSolver
{
public:
   UmfpackDirectSolver() : umf_(true)
   {
      umf_.Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
      umf_.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD;
   }
   bool Solve(ComplexSparseMatrix &A, const Vector &B, Vector &X) override
   {
      if (!ready_)
      {
         umf_.SetOperator(A);
         ready_ = true;
         std::printf("umfpack_peak_gb %.3f\n",
                     umf_.Info[UMFPACK_PEAK_MEMORY] *
                     umf_.Info[UMFPACK_SIZE_OF_UNIT] / 1e9);
         const double rcond = umf_.Info[UMFPACK_RCOND];
         std::printf("umfpack_rcond %.3e\n", rcond);
         if (rcond > 0.0 && rcond < 1e-12)
         {
            std::printf("WARNING: factorization looks ill-conditioned "
                        "(rcond ~ %.1e); the reported residual is the "
                        "authoritative accuracy check\n", rcond);
         }
      }
      umf_.Mult(B, X);
      return true;
   }
   const char *Name() const override { return "umfpack-direct"; }

private:
   ComplexUMFPackSolver umf_;
   bool ready_ = false;
};

class GmresIterativeSolver : public ILinearSolver
{
public:
   GmresIterativeSolver(double rtol, int maxit) : rtol_(rtol), maxit_(maxit) {}
   bool Solve(ComplexSparseMatrix &A, const Vector &B, Vector &X) override
   {
      const int n_c = A.real().Height();
      GMRESSolver gmres;
      gmres.SetRelTol(rtol_);
      gmres.SetMaxIter(maxit_);
      gmres.SetKDim(200);
      gmres.SetPrintLevel(1);
      DSmoother jac(const_cast<SparseMatrix &>(A.real()));
      Array<int> offs(3);
      offs[0] = 0;
      offs[1] = n_c;
      offs[2] = 2 * n_c;
      BlockDiagonalPreconditioner pc(offs);
      pc.SetDiagonalBlock(0, &jac);
      pc.SetDiagonalBlock(1, &jac);
      gmres.SetPreconditioner(pc);
      gmres.SetOperator(A);
      X = 0.0;
      gmres.Mult(B, X);
      if (gmres.GetConverged())
      {
         std::printf("iterative solve converged in %d iters\n",
                     gmres.GetNumIterations());
         return true;
      }
      std::printf("iterative solve did NOT converge (%d iters); "
                  "falling back to the direct solver\n",
                  gmres.GetNumIterations());
      return false;
   }
   const char *Name() const override { return "gmres-jacobi"; }

private:
   double rtol_;
   int maxit_;
};

#include "cudss_solver.inc"

// v2.5 GPU assembly (host extraction + device kernels + zero-copy cuDSS).
// Only meaningful when the cuDSS direct path is available.
#ifdef LITHOFEM_HAVE_CUDSS
#include "asm_data.inc"
#include "asm_gpu.inc"
#endif

} // namespace

int main(int argc, char *argv[])
{
   const char *mesh_file = nullptr, *json_file = nullptr, *outdir = ".";
   int group = 0;
   double shift_x = 0.0;
   const char *device_config = "";
   const char *assembly_config = "";
   const char *export_prefix = "";
   const char *import_solution = "";
   double mem_limit_gb = 0.0;
   double gpu_mem_limit_gb = 0.0;
   bool probe_only = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "periodic mesh (.per.msh)");
   args.AddOption(&json_file, "-j", "--json", "solve.json");
   args.AddOption(&outdir, "-o", "--outdir", "output directory");
   args.AddOption(&group, "-g", "--group", "solve group index");
   args.AddOption(&shift_x, "-sx", "--shift-x",
                  "rigidly translate the mesh by dx with periodic wrap "
                  "(translation-covariance checks, M6-3)");
   args.AddOption(&device_config, "-d", "--device", "cpu | cuda (overrides solve.json)");
   args.AddOption(&assembly_config, "-a", "--assembly",
                  "cpu | gpu matrix assembly (overrides solve.json fem.assembly)");
   args.AddOption(&probe_only, "-probe", "--probe", "-no-probe", "--no-probe",
                  "report the problem size and exit before assembly "
                  "(pre-flight cost estimate)");
   args.AddOption(&export_prefix, "-es", "--export-system",
                  "write A (MatrixMarket complex) + RHS, then exit");
   args.AddOption(&import_solution, "-is", "--import-solution",
                  "read solution vector (MatrixMarket array) and skip the solve");
   args.AddOption(&mem_limit_gb, "-ml", "--mem-limit-gb",
                  "abort before factorization if the UMFPACK estimate exceeds this");
   args.AddOption(&gpu_mem_limit_gb, "-gml", "--gpu-mem-limit-gb",
                  "cuDSS VRAM cap in GB (over-estimate -> CPU fallback); "
                  "0 = auto (free VRAM), overrides solve.json gpu_mem_gb");
   args.Parse();
   if (!args.Good() || !mesh_file || !json_file)
   {
      args.PrintUsage(std::cout);
      return 1;
   }

   json doc;
   {
      std::ifstream f(json_file);
      MFEM_VERIFY(f.good(), "cannot open solve.json");
      f >> doc;
   }

   parse_problem_core(doc, group);
   // v2.5: fem.assembly cpu|gpu (default cpu; CLI -a overrides)
   std::string assembly = doc["fem"].value("assembly", std::string("cpu"));
   if (assembly_config[0]) { assembly = assembly_config; }
   if (doc.contains("solver"))
   {
      const auto &sj = doc["solver"];
      prob.solver_type = sj.value("type", std::string("direct"));
      prob.device = sj.value("device", std::string("cpu"));
      if (sj.contains("gpu_ids") && !sj["gpu_ids"].empty())
      {
         prob.gpu_id = sj["gpu_ids"][0].get<int>();
      }
      prob.it_rtol = sj.value("rtol", 1e-8);
      prob.it_maxit = sj.value("max_iter", 2000);
      prob.gpu_mem_gb = sj.value("gpu_mem_gb", 0.0);
   }
   if (gpu_mem_limit_gb > 0.0) { prob.gpu_mem_gb = gpu_mem_limit_gb; }
   // Sweep tasks (V2-M4) pin one physical card via CUDA_VISIBLE_DEVICES;
   // inside such a process the only valid device ordinal is 0.
   if (const char *cvd = std::getenv("CUDA_VISIBLE_DEVICES"))
   {
      if (*cvd && !std::strchr(cvd, ',') && prob.gpu_id != 0)
      {
         std::printf("CUDA_VISIBLE_DEVICES=%s pins a single card; "
                     "using device ordinal 0 (was gpu_ids[0]=%d)\n",
                     cvd, prob.gpu_id);
         prob.gpu_id = 0;
      }
   }

   const auto &gj = doc["groups"][group];
   for (const auto &si : gj["source_indices"]) { prob.group_sources.push_back(si.get<int>()); }
   prob.per_source = doc["output"].value("per_source", false);
   const auto &all_sources = doc["sources"];
   for (int si : prob.group_sources)
   {
      const auto &sj = all_sources[si];
      const std::string st = sj["type"].get<std::string>();
      if (st == "planewave") { continue; }
      LocalSource ls;
      ls.source_index = si;
      auto getv3 = [](const json &a, double *out)
      { for (int d = 0; d < 3; ++d) { out[d] = a[d].get<double>(); } };
      for (int d = 0; d < 3; ++d) { ls.cur[d] = jc(sj["current"][d]); }
      if (st == "point")
      {
         ls.type = 0;
         getv3(sj["position"], ls.p0);
      }
      else if (st == "line")
      {
         ls.type = 1;
         getv3(sj["endpoints"][0], ls.p0);
         getv3(sj["endpoints"][1], ls.p1);
         ls.pg[0] = sj.value("phase_gradient", 0.0);
      }
      else if (st == "sheet")
      {
         ls.type = 2;
         getv3(sj["corner"], ls.p0);
         getv3(sj["edges"][0], ls.p1);
         getv3(sj["edges"][1], ls.e2);
         if (sj.contains("phase_gradient"))
         {
            ls.pg[0] = sj["phase_gradient"][0].get<double>();
            ls.pg[1] = sj["phase_gradient"][1].get<double>();
         }
      }
      else { MFEM_ABORT("unknown source type"); }
      prob.local.push_back(ls);
   }
   for (const auto &part : gj["incident"])
   {
      prob.incident_src.push_back(part["source_index"].get<int>());
      std::vector<SlabWave> sws;
      for (const auto &sj : part["slabs"])
      {
         SlabWave sw;
         sw.z_lo = sj["z"][0].get<double>();
         sw.z_hi = sj["z"][1].get<double>();
         sw.qA = jc(sj["qA"]);
         sw.zA = sj["zA"].get<double>();
         sw.qB = jc(sj["qB"]);
         sw.zB = sj["zB"].get<double>();
         for (int i = 0; i < 3; ++i)
         {
            sw.A[i] = jc(sj["A"][i]);
            sw.B[i] = jc(sj["B"][i]);
         }
         sws.push_back(sw);
      }
      prob.incident.push_back(sws);
   }
   for (const auto &pj : doc["output"]["planes"])
   {
      prob.plane_z.push_back(pj["z"].get<double>());
      prob.plane_nx.push_back(pj["resolution"][0].get<int>());
      prob.plane_ny.push_back(pj["resolution"][1].get<int>());
   }

   // v2 semantics (docs/gpu.md): solver.device gpu + type direct = cuDSS
   // direct solve; the MFEM device stays CPU (tet assembly has no device
   // kernels, and host-side RHS/sampling paths are only validated there).
   // The MFEM cuda device engages for iterative+gpu (v1 GMRES path) or an
   // explicit -d cuda override.
   std::string devstr = device_config;
   if (devstr.empty())
   {
      devstr = (prob.device == "gpu" && prob.solver_type == "iterative")
                  ? "cuda" : "cpu";
   }
   if (devstr == "cuda" && prob.gpu_id != 0)
   {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "cuda:%d", prob.gpu_id);
      devstr = buf;
   }
   Device device(devstr.c_str());
   device.Print();

   seg_t0 = SegClock::now();
   Mesh mesh(mesh_file, 1, 1);
   std::printf("mesh: %d elements, %d vertices\n", mesh.GetNE(), mesh.GetNV());

   if (shift_x != 0.0)
   {
      // translate the periodic mesh on its torus: per-element rigid shift of
      // the (discontinuous) nodal coordinates, wrapped back into [0, Lx)
      GridFunction *nodes = mesh.GetNodes();
      MFEM_VERIFY(nodes, "expected a periodic (curved-node) mesh");
      FiniteElementSpace *nfes = nodes->FESpace();
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         Array<int> dofs;
         nfes->GetElementDofs(e, dofs);
         double cx = 0.0;
         for (int j = 0; j < dofs.Size(); ++j)
         {
            cx += (*nodes)(nfes->DofToVDof(dofs[j], 0));
         }
         cx = cx / dofs.Size() + shift_x;
         const double wrap = prob.lx * std::floor(cx / prob.lx);
         for (int j = 0; j < dofs.Size(); ++j)
         {
            const int vd = nfes->DofToVDof(dofs[j], 0);
            (*nodes)(vd) += shift_x - wrap;
         }
      }
   }

   seg_mark("mesh_read");
   ND_FECollection fec(prob.order, 3);
   FiniteElementSpace fes(&mesh, &fec);
   std::printf("ndof: %d\n", fes.GetTrueVSize());

   if (probe_only)
   {
      // Pre-flight: the caller wants the problem size (and from it the memory
      // cost) before committing to a run. Everything expensive is downstream.
      std::printf("probe_ndof %d\nprobe_elements %d\nlithofem_solve: OK\n",
                  fes.GetTrueVSize(), mesh.GetNE());
      return 0;
   }

   Array<int> ess_bdr(mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0);
   ess_bdr = 1;
   Array<int> ess_tdof_list;
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   seg_mark("fespace");

   const ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;

   ScatterSource fr(false), fi(true);
   ComplexLinearForm b(&fes, conv);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fr),
                         new VectorFEDomainLFIntegrator(fi));
   b.Assemble();
   add_local_sources(fes, mesh, b.real(), b.imag());
   seg_mark("rhs");

   LamInvPart lam_re(false), lam_im(true);
   MassPart mass_re(false), mass_im(true);
   CrossPart c0_re(0, false), c0_im(0, true), c1_re(1, false), c1_im(1, true);
   KLKPart klk_re(false), klk_im(true);

   SesquilinearForm a(&fes, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(lam_re),
                         new CurlCurlIntegrator(lam_im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(mass_re),
                         new VectorFEMassIntegrator(mass_im));
   if (prob.kx != 0.0 || prob.ky != 0.0)
   {
      a.AddDomainIntegrator(new MixedVectorCurlIntegrator(c0_re),
                            new MixedVectorCurlIntegrator(c0_im));
      a.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(c1_re),
                            new MixedVectorWeakCurlIntegrator(c1_im));
      a.AddDomainIntegrator(new VectorFEMassIntegrator(klk_re),
                            new VectorFEMassIntegrator(klk_im));
   }
   ComplexGridFunction u(&fes);
   u = std::complex<real_t>(0.0, 0.0);

   OperatorHandle A;
   Vector X, B;
   ComplexSparseMatrix *Ac = nullptr;
   int n_c = fes.GetTrueVSize();

   bool solved = false;
   UmfpackDirectSolver umf_solver;
   // v2 routing (docs/gpu.md): direct + gpu -> cuDSS, any failure -> UMFPACK.
   const bool want_gpu_direct =
      prob.device == "gpu" || devstr.rfind("cuda", 0) == 0;
#ifdef LITHOFEM_HAVE_CUDSS
   CuDssDirectSolver cudss_solver(prob.gpu_id, prob.gpu_mem_gb);
   CuDssDirectSolver *cudss = want_gpu_direct ? &cudss_solver : nullptr;
#else
   ILinearSolver *cudss = nullptr;
   if (want_gpu_direct)
   {
      std::printf("cudss: support not built into this binary\n");
   }
#endif

   // build the eliminated RHS without a host matrix: X = 0 and PEC values
   // are zero, so B = [b_re; b_im] with essential rows zeroed (exactly what
   // FormLinearSystem produces here; verified element-wise by U6)
   auto form_rhs_only = [&](const Vector &bf, Vector &Bv)
   {
      Bv.SetSize(2 * n_c);
      const double *bd = bf.HostRead();
      std::memcpy(Bv.HostWrite(), bd, sizeof(double) * 2 * n_c);
      for (int i = 0; i < ess_tdof_list.Size(); ++i)
      {
         Bv(ess_tdof_list[i]) = 0.0;
         Bv(n_c + ess_tdof_list[i]) = 0.0;
      }
   };

   // ---- v2.5 GPU assembly (fem.assembly: gpu) ---------------------------
   bool gpu_assembled = false;
#ifdef LITHOFEM_HAVE_CUDSS
   AsmData asm_data;
   AsmGpuGlobal asm_gpu;
   if (assembly == "gpu")
   {
      if (!want_gpu_direct || prob.solver_type != "direct" ||
          export_prefix[0] || import_solution[0])
      {
         std::printf("asm_gpu: gpu assembly needs solver device=gpu, "
                     "type=direct, no export/import; using cpu assembly\n");
      }
      else if (std::getenv("LITHOFEM_ASM_FORCE_FAIL"))
      {
         std::printf("asm_gpu: forced failure injected "
                     "(LITHOFEM_ASM_FORCE_FAIL)\n");
         std::printf("asm_gpu: falling back to MFEM cpu assembly\n");
      }
      else
      {
         gpu_assembled = [&]()
         {
            int ndev = 0;
            if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0)
            {
               std::printf("asm_gpu: no CUDA device\n");
               return false;
            }
            StopWatch sw;
            sw.Start();
            extract_asm_data(mesh, fes, asm_data);
            std::printf("asm_gpu_extract_s %.3f\n", sw.RealTime());
            sw.Clear();
            sw.Start();
            build_csr_symbolic(asm_data);
            std::printf("asm_gpu_csr_s %.3f\n", sw.RealTime());
            sw.Clear();
            sw.Start();
            if (!asm_gpu.Init(asm_data, prob.gpu_id)) { return false; }
            if (!asm_gpu.BuildGlobal(asm_data, ess_tdof_list)) { return false; }
            std::printf("asm_gpu_upload_s %.3f\n", sw.RealTime());
            sw.Clear();
            sw.Start();
            const bool kpar = prob.kx != 0.0 || prob.ky != 0.0;
            const int mm = ASM_MASS |
                           (kpar ? ASM_CROSS0 | ASM_CROSS1 | ASM_KLK : 0);
            if (!asm_gpu.ComputeLocal(ASM_CURLCURL, mm, true)) { return false; }
            if (!asm_gpu.ScatterEliminate(true)) { return false; }
            std::printf("asm_gpu_kernel_s %.3f\n", sw.RealTime());
            return true;
         }();
         if (!gpu_assembled)
         {
            std::printf("asm_gpu: falling back to MFEM cpu assembly\n");
         }
      }
   }
#else
   if (assembly == "gpu")
   {
      std::printf("asm_gpu: support not built into this binary; "
                  "using cpu assembly\n");
   }
#endif

   auto cpu_assemble_form = [&]()
   {
      a.Assemble(0);  // skip_zeros=0: keep re/im sparsity identical
      seg_mark("assemble");
      a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);
      Ac = A.As<ComplexSparseMatrix>();
      n_c = Ac->real().Height();
      std::printf("system size (complex): %d, nnz(re): %d\n",
                  n_c, Ac->real().NumNonZeroElems());
      seg_mark("form");
   };
   if (!gpu_assembled)
   {
      cpu_assemble_form();
   }
#ifdef LITHOFEM_HAVE_CUDSS
   else
   {
      seg_mark("assemble");
      X.SetSize(2 * n_c);
      X = 0.0;
      form_rhs_only(b, B);
      std::printf("system size (complex): %d, nnz(re): %d\n",
                  n_c, asm_gpu.nnz());
      seg_mark("form");
   }
#endif

   if (export_prefix[0])
   {
      export_system(*Ac, B, export_prefix);
      std::printf("system exported to %s.{mtx,rhs.mtx}; exiting\n",
                  export_prefix);
      return 0;
   }

   auto direct_solve = [&](Vector &Bv, Vector &Xv)
   {
      if (cudss)
      {
         if (cudss->Solve(*Ac, Bv, Xv)) { return; }
         std::printf("cudss failed; falling back to UMFPACK\n");
      }
      umf_solver.Solve(*Ac, Bv, Xv);
   };
   if (import_solution[0])
   {
      import_vector(import_solution, X, n_c);
      std::printf("solution imported from %s\n", import_solution);
      solved = true;
   }

   if (!solved && prob.solver_type == "iterative")
   {
      GmresIterativeSolver it_solver(prob.it_rtol, prob.it_maxit);
      solved = it_solver.Solve(*Ac, B, X);
   }

   if (!solved)
   {
#ifdef LITHOFEM_HAVE_CUDSS
      if (gpu_assembled)
      {
         solved = cudss_solver.SolveDeviceCsr(
            n_c, asm_gpu.nnz(), asm_gpu.DeviceI(), asm_gpu.DeviceJ(),
            asm_gpu.DeviceValues(), B, X);
         if (!solved)
         {
            std::printf("asm_gpu: device-CSR solve failed; falling back to "
                        "cpu assembly + direct solve\n");
            gpu_assembled = false;
            cpu_assemble_form();
         }
      }
#endif
      if (!solved)
      {
         if (mem_limit_gb > 0.0)
         {
            const double est_gb = umfpack_symbolic_estimate_gb(*Ac);
            std::printf("umfpack_peak_estimate_gb %.3f\n", est_gb);
            if (est_gb > mem_limit_gb)
            {
               std::printf("MEMORY LIMIT: estimated %.1f GB exceeds the limit "
                           "%.1f GB; aborting (use --export-system, a coarser "
                           "mesh, or a machine with more memory)\n",
                           est_gb, mem_limit_gb);
               return 3;
            }
         }
         direct_solve(B, X);
      }
   }
   seg_mark("solve");

   double rel_res = 0.0;
#ifdef LITHOFEM_HAVE_CUDSS
   if (gpu_assembled)
   {
      // self-residual via device SpMV: the matrix never exists on the host
      MFEM_VERIFY(asm_gpu.Residual(X, B, rel_res), "device residual failed");
   }
   else
#endif
   {
      Vector R(B.Size());
      Ac->Mult(X, R);
      R -= B;
      const double bnorm = B.Norml2();
      rel_res = bnorm > 0.0 ? R.Norml2() / bnorm : 0.0;
   }
   std::printf("residual %.3e\n", rel_res);

   a.RecoverFEMSolution(X, b, u);

   // per-region L2^2 of the scattered envelope (|u| = |E_sc|)
   std::vector<double> l2sq(prob.regions.size(), 0.0);
   {
      GridFunction &ur = u.real();
      GridFunction &ui = u.imag();
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         const int attr = mesh.GetAttribute(e);
         const FiniteElement *fe = fes.GetFE(e);
         ElementTransformation *tr = mesh.GetElementTransformation(e);
         const IntegrationRule &ir =
            IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + 2);
         for (int q = 0; q < ir.GetNPoints(); ++q)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            tr->SetIntPoint(&ip);
            Vector vr(3), vi(3);
            ur.GetVectorValue(*tr, ip, vr);
            ui.GetVectorValue(*tr, ip, vi);
            l2sq[attr - 1] +=
               ip.weight * tr->Weight() * (vr * vr + vi * vi);
         }
      }
   }

   // sample requested planes (reused for per-source solutions)
   auto sample_planes = [&](ComplexGridFunction &sol, const char *suffix)
   {
      for (size_t p = 0; p < prob.plane_z.size(); ++p)
      {
         const int nx = prob.plane_nx[p], ny = prob.plane_ny[p];
         std::vector<double> buf((size_t)nx * ny * 6, 0.0);
         std::vector<double> cbuf((size_t)nx * ny * 6, 0.0);
         DenseMatrix pts(3, nx);
         Array<int> elem_ids;
         Array<IntegrationPoint> ips;
         for (int j = 0; j < ny; ++j)
         {
            for (int i = 0; i < nx; ++i)
            {
               // deterministic side selection: tiny fixed nudge keeps sample
               // points off element faces/edges
               pts(0, i) = (i + 0.5) * prob.lx / nx + 3.1e-7;
               pts(1, i) = (j + 0.5) * prob.ly / ny + 1.7e-7;
               pts(2, i) = prob.plane_z[p] - 4.9e-7;
            }
            mesh.FindPoints(pts, elem_ids, ips, false);
            for (int i = 0; i < nx; ++i)
            {
               if (elem_ids[i] >= 0) { continue; }
               double x1[3] = {pts(0, i), pts(1, i), pts(2, i)};
               int e1;
               IntegrationPoint ip1;
               MFEM_VERIFY(find_point(mesh, x1, e1, ip1),
                           "sample point not found");
               elem_ids[i] = e1;
               ips[i] = ip1;
            }
            for (int i = 0; i < nx; ++i)
            {
               Vector vr(3), vi(3), cr(3), ci(3);
               sol.real().GetVectorValue(elem_ids[i], ips[i], vr);
               sol.imag().GetVectorValue(elem_ids[i], ips[i], vi);
               ElementTransformation *tr =
                  mesh.GetElementTransformation(elem_ids[i]);
               tr->SetIntPoint(&ips[i]);
               sol.real().GetCurl(*tr, cr);
               sol.imag().GetCurl(*tr, ci);
               const size_t base = ((size_t)j * nx + i) * 6;
               for (int c = 0; c < 3; ++c)
               {
                  buf[base + 2 * c] = vr(c);
                  buf[base + 2 * c + 1] = vi(c);
                  cbuf[base + 2 * c] = cr(c);
                  cbuf[base + 2 * c + 1] = ci(c);
               }
            }
         }
         char path[512];
         std::snprintf(path, sizeof(path), "%s/plane_g%d_p%zu%s.bin",
                       outdir, group, p, suffix);
         std::ofstream out(path, std::ios::binary);
         out.write(reinterpret_cast<const char *>(buf.data()),
                   (std::streamsize)(buf.size() * sizeof(double)));
         std::snprintf(path, sizeof(path), "%s/plane_g%d_p%zu%s_curl.bin",
                       outdir, group, p, suffix);
         std::ofstream outc(path, std::ios::binary);
         outc.write(reinterpret_cast<const char *>(cbuf.data()),
                    (std::streamsize)(cbuf.size() * sizeof(double)));
      }
   };
   seg_mark("postproc");
   sample_planes(u, "");

   if (doc["output"]["volume"].value("enabled", false))
   {
      const bool with_pml = doc["output"]["volume"].value("include_pml", false);
      const std::string vfile =
         doc["output"]["volume"].value("file", std::string("field_full"));
      Array<int> attrs;
      for (size_t k = 0; k < prob.regions.size(); ++k)
      {
         if (with_pml || !prob.regions[k].is_pml) { attrs.Append((int)k + 1); }
      }
      SubMesh smesh = SubMesh::CreateFromDomain(mesh, attrs);
      ND_FECollection sfec(prob.order, 3);
      FiniteElementSpace sfes(&smesh, &sfec);
      GridFunction ur(&sfes), ui(&sfes);
      smesh.Transfer(u.real(), ur);
      smesh.Transfer(u.imag(), ui);
      char vpath[512];
      std::snprintf(vpath, sizeof(vpath), "%s_g%d", vfile.c_str(), group);
      ParaViewDataCollection pvd(vpath, &smesh);
      pvd.SetPrefixPath(outdir);
      pvd.SetHighOrderOutput(true);
      pvd.SetLevelsOfDetail(prob.order);
      pvd.RegisterField("Esc_env_re", &ur);
      pvd.RegisterField("Esc_env_im", &ui);
      pvd.SetCycle(0);
      pvd.SetTime(0.0);
      pvd.Save();
      std::printf("paraview volume written: %s/%s\n", outdir, vpath);
   }

   seg_mark("output");

   // optional per-source outputs: fresh RHS per source, factorization reused
   if (prob.per_source && prob.group_sources.size() > 1)
   {
      for (int si : prob.group_sources)
      {
         prob.active_parts.clear();
         for (size_t pi = 0; pi < prob.incident_src.size(); ++pi)
         {
            if (prob.incident_src[pi] == si) { prob.active_parts.push_back((int)pi); }
         }
         if (prob.active_parts.empty()) { prob.active_parts.push_back(-1); }
         prob.active_local.clear();
         for (size_t li = 0; li < prob.local.size(); ++li)
         {
            if (prob.local[li].source_index == si)
            {
               prob.active_local.push_back((int)li);
            }
         }
         if (prob.active_local.empty()) { prob.active_local.push_back(-1); }
         ComplexLinearForm bs(&fes, conv);
         ScatterSource fr_s(false), fi_s(true);
         bs.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fr_s),
                                new VectorFEDomainLFIntegrator(fi_s));
         bs.Assemble();
         add_local_sources(fes, mesh, bs.real(), bs.imag());
         ComplexGridFunction us(&fes);
         us = std::complex<real_t>(0.0, 0.0);
         OperatorHandle As;
         Vector Xs, Bs;
#ifdef LITHOFEM_HAVE_CUDSS
         if (gpu_assembled)
         {
            Xs.SetSize(2 * n_c);
            Xs = 0.0;
            form_rhs_only(bs, Bs);
            if (!cudss_solver.SolveDeviceCsr(
                   n_c, asm_gpu.nnz(), asm_gpu.DeviceI(), asm_gpu.DeviceJ(),
                   asm_gpu.DeviceValues(), Bs, Xs))
            {
               std::printf("asm_gpu: per-source device solve failed; "
                           "falling back to cpu assembly\n");
               gpu_assembled = false;
               cpu_assemble_form();
            }
         }
         if (!gpu_assembled)
#endif
         {
            a.FormLinearSystem(ess_tdof_list, us, bs, As, Xs, Bs);
            direct_solve(Bs, Xs);
         }
         a.RecoverFEMSolution(Xs, bs, us);
         char suffix[32];
         std::snprintf(suffix, sizeof(suffix), "_s%d", si);
         sample_planes(us, suffix);
      }
      prob.active_parts.clear();
      prob.active_local.clear();
      seg_mark("per_source");
   }

   // meta
   {
      json meta;
      meta["group"] = group;
      meta["residual"] = rel_res;
      meta["ndof"] = fes.GetTrueVSize();
      meta["region_l2sq"] = l2sq;
      json timing;
      double total = 0.0;
      for (const auto &st : seg_times)
      {
         timing[st.first] = st.second;
         total += st.second;
      }
      timing["total"] = total;
      meta["timing_s"] = timing;
      meta["planes"] = json::array();
      for (size_t p = 0; p < prob.plane_z.size(); ++p)
      {
         meta["planes"].push_back({{"z", prob.plane_z[p]},
                                   {"nx", prob.plane_nx[p]},
                                   {"ny", prob.plane_ny[p]}});
      }
      char path[512];
      std::snprintf(path, sizeof(path), "%s/solve_meta_g%d.json", outdir, group);
      std::ofstream out(path);
      out << meta.dump(1);
   }
   std::printf("lithofem_solve: OK\n");
   return 0;
}
