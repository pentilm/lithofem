// v2.5 assembly correctness harness (docs/gpu.md, pytest-wrapped).
//
// Runs, for one prepared case (mesh + solve.json + group):
//   U1: reference basis/curl tables vs direct MFEM CalcVShape/CalcCurlShape,
//       p = 1..4, all points of both default integration rules;
//   U2: extracted affine geometry (J, detJ, origin) vs the MFEM element
//       transformation at interior sample points + total mesh volume;
//   U7: CPU reference reassembler (extracted data, GPU operation order) vs
//       the production SesquilinearForm::Assemble(0): bitwise-equal sparsity
//       and element-wise value comparison (target rel < 1e-13).
//
// Prints `key value` lines consumed by tests/test_asm_v25.py.

#include "mfem.hpp"
#include "thirdparty/json.hpp"

#include <cuda_runtime.h>

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

using namespace mfem;
using json = nlohmann::json;
using C = std::complex<double>;

namespace
{

#include "problem_common.inc"
#include "asm_data.inc"
#include "asm_gpu.inc"

double now_s()
{
   using clk = std::chrono::steady_clock;
   return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// U1: re-extract the reference tables and compare against direct MFEM calls
// point by point (validates table layout/indexing for p = 1..4).
double u1_ref_tables()
{
   double max_diff = 0.0;
   for (int p = 1; p <= 4; ++p)
   {
      ND_TetrahedronElement fe(p);
      const int nd = fe.GetDof();
      for (int rule = 0; rule < 2; ++rule)
      {
         const IntegrationRule &ir = IntRules.Get(
            Geometry::TETRAHEDRON, rule == 0 ? std::max(0, 2 * p - 2) : 2 * p);
         AsmRefRule rr;
         extract_ref_rule(fe, ir, rr);
         DenseMatrix vs(nd, 3), cs(nd, 3);
         for (int q = 0; q < rr.nq; ++q)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            fe.CalcVShape(ip, vs);
            fe.CalcCurlShape(ip, cs);
            for (int k = 0; k < nd; ++k)
               for (int d = 0; d < 3; ++d)
               {
                  max_diff = std::max(max_diff,
                     std::abs(rr.vshape[((size_t)q * nd + k) * 3 + d] - vs(k, d)));
                  max_diff = std::max(max_diff,
                     std::abs(rr.curl[((size_t)q * nd + k) * 3 + d] - cs(k, d)));
               }
         }
      }
   }
   return max_diff;
}

// U2: extracted geometry vs MFEM element transformations.
void u2_geometry(Mesh &mesh, const AsmData &ad, double &max_J_rel,
                 double &max_x_abs, double &vol_rel)
{
   max_J_rel = 0.0;
   max_x_abs = 0.0;
   // three interior reference points (affine: J constant, x = v0 + J xi)
   const double pts[3][3] = {{0.25, 0.25, 0.25},
                             {0.13, 0.42, 0.11},
                             {0.61, 0.05, 0.17}};
   double vol_mfem = 0.0, vol_ext = 0.0;
   for (int e = 0; e < ad.ne; ++e)
   {
      ElementTransformation *tr = mesh.GetElementTransformation(e);
      const double *J = ad.J.data() + (size_t)e * 9;
      const double *v0 = ad.v0.data() + (size_t)e * 3;
      double Jnorm = 0.0;
      for (int i = 0; i < 9; ++i) { Jnorm = std::max(Jnorm, std::abs(J[i])); }
      for (int s = 0; s < 3; ++s)
      {
         IntegrationPoint ip;
         ip.Set3(pts[s][0], pts[s][1], pts[s][2]);
         tr->SetIntPoint(&ip);
         const DenseMatrix &Jm = tr->Jacobian();
         for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
               max_J_rel = std::max(max_J_rel,
                  std::abs(Jm(i, j) - J[i * 3 + j]) / Jnorm);
            }
         Vector x(3);
         tr->Transform(ip, x);
         for (int i = 0; i < 3; ++i)
         {
            const double xe = v0[i] + J[i * 3 + 0] * pts[s][0] +
                              J[i * 3 + 1] * pts[s][1] + J[i * 3 + 2] * pts[s][2];
            max_x_abs = std::max(max_x_abs, std::abs(x(i) - xe));
         }
      }
      vol_mfem += mesh.GetElementVolume(e);
      vol_ext += ad.detJ[e] / 6.0;   // reference tet volume 1/6
   }
   vol_rel = std::abs(vol_ext - vol_mfem) / std::abs(vol_mfem);
}

} // namespace

int main(int argc, char *argv[])
{
   const char *mesh_file = nullptr, *json_file = nullptr;
   int group = 0;
   int order_override = 0;
   bool do_gpu = false, skip_cpu = false, do_gpu_global = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "periodic mesh (.per.msh)");
   args.AddOption(&json_file, "-j", "--json", "solve.json");
   args.AddOption(&group, "-g", "--group", "solve group index");
   args.AddOption(&order_override, "-p", "--order",
                  "override the ND order from solve.json (U3/U4 p=1..4)");
   args.AddOption(&do_gpu, "-gpu", "--gpu", "-no-gpu", "--no-gpu",
                  "run the U3/U4 GPU local-matrix comparisons");
   args.AddOption(&do_gpu_global, "-gpu-global", "--gpu-global",
                  "-no-gpu-global", "--no-gpu-global",
                  "run the U5/U6 GPU global assembly + elimination comparisons");
   args.AddOption(&skip_cpu, "-skip-cpu", "--skip-cpu", "-with-cpu",
                  "--with-cpu", "skip the U7 CPU reassembly comparison");
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
   if (order_override > 0) { prob.order = order_override; }

   std::printf("u1_max_abs %.3e\n", u1_ref_tables());

   Mesh mesh(mesh_file, 1, 1);
   ND_FECollection fec(prob.order, 3);
   FiniteElementSpace fes(&mesh, &fec);
   std::printf("ndof %d\nkpar %.6e %.6e\norder %d\n",
               fes.GetTrueVSize(), prob.kx, prob.ky, prob.order);

   // ---- extraction (timed: the per-mesh one-off cost of the GPU path) ----
   AsmData ad;
   double t0 = now_s();
   extract_asm_data(mesh, fes, ad);
   const double t_extract = now_s() - t0;
   t0 = now_s();
   build_csr_symbolic(ad);
   const double t_csr = now_s() - t0;
   std::printf("timing_extract_s %.3f\ntiming_csr_s %.3f\n", t_extract, t_csr);

   double max_J_rel, max_x_abs, vol_rel;
   u2_geometry(mesh, ad, max_J_rel, max_x_abs, vol_rel);
   std::printf("u2_J_rel %.3e\nu2_x_abs %.3e\nu2_vol_rel %.3e\n",
               max_J_rel, max_x_abs, vol_rel);

   if (!skip_cpu)
   {
      // ---- MFEM production assembly (the U7 reference) -------------------
      LamInvPart lam_re(false), lam_im(true);
      MassPart mass_re(false), mass_im(true);
      CrossPart c0_re(0, false), c0_im(0, true), c1_re(1, false), c1_im(1, true);
      KLKPart klk_re(false), klk_im(true);

      const ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;
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
      t0 = now_s();
      a.Assemble(0);   // skip_zeros=0, as in production
      const double t_mfem = now_s() - t0;
      a.real().Finalize(0);
      a.imag().Finalize(0);
      SparseMatrix &Ar = a.real().SpMat();
      SparseMatrix &Ai = a.imag().SpMat();
      Ar.SortColumnIndices();
      Ai.SortColumnIndices();
      std::printf("timing_mfem_assemble_s %.3f\n", t_mfem);

      // ---- CPU reference reassembly (timed) ------------------------------
      std::vector<double> vre, vim;
      t0 = now_s();
      reassemble_all(ad, vre, vim);
      const double t_re = now_s() - t0;
      std::printf("timing_reassemble_s %.3f\n", t_re);

      // ---- U7 comparison -------------------------------------------------
      const int n = Ar.Height();
      bool struct_ok = n == ad.height &&
                       Ar.NumNonZeroElems() == (int)ad.Jcol.size();
      if (struct_ok)
      {
         const int *I = Ar.GetI();
         const int *Jc = Ar.GetJ();
         for (int r = 0; r <= n && struct_ok; ++r)
         {
            struct_ok = I[r] == ad.I[r];
         }
         for (size_t k = 0; k < ad.Jcol.size() && struct_ok; ++k)
         {
            struct_ok = Jc[k] == ad.Jcol[k];
         }
         // the imaginary part must share the sparsity (skip_zeros=0 invariant)
         const int *Ii = Ai.GetI();
         for (int r = 0; r <= n && struct_ok; ++r)
         {
            struct_ok = Ii[r] == ad.I[r];
         }
      }
      std::printf("u7_struct_ok %d\n", struct_ok ? 1 : 0);
      if (!struct_ok)
      {
         std::printf("asm_test: FAIL (sparsity mismatch)\n");
         return 2;
      }
      double max_abs = 0.0, max_diff = 0.0;
      const double *ar = Ar.GetData();
      const double *ai = Ai.GetData();
      for (size_t k = 0; k < ad.Jcol.size(); ++k)
      {
         max_abs = std::max(max_abs, std::hypot(ar[k], ai[k]));
         max_diff = std::max(max_diff,
                             std::hypot(vre[k] - ar[k], vim[k] - ai[k]));
      }
      std::printf("u7_rel %.3e\n", max_diff / max_abs);
   }

   if (do_gpu)
   {
      // ---- U3/U4: GPU local-matrix kernels vs MFEM element matrices ------
      int devcount = 0;
      if (cudaGetDeviceCount(&devcount) != cudaSuccess || devcount == 0)
      {
         std::printf("asm_test: no CUDA device\n");
         return 4;
      }
      AsmGpu gpu;
      MFEM_VERIFY(gpu.Init(ad, 0), "asm gpu init failed");
      const bool kpar = prob.kx != 0.0 || prob.ky != 0.0;
      const int nd = ad.nd;

      // sample elements: up to 4 per attribute (bulk / frustum / PML slabs)
      std::vector<int> sample;
      {
         std::vector<int> cnt(prob.regions.size(), 0);
         for (int e = 0; e < ad.ne; ++e)
         {
            if (cnt[ad.attr[e] - 1]++ < 4) { sample.push_back(e); }
         }
      }

      LamInvPart lam_re(false), lam_im(true);
      MassPart mass_re(false), mass_im(true);
      CrossPart c0_re(0, false), c0_im(0, true), c1_re(1, false), c1_im(1, true);
      KLKPart klk_re(false), klk_im(true);
      CurlCurlIntegrator icc_re(lam_re), icc_im(lam_im);
      VectorFEMassIntegrator ims_re(mass_re), ims_im(mass_im);
      MixedVectorCurlIntegrator ic0_re(c0_re), ic0_im(c0_im);
      MixedVectorWeakCurlIntegrator ic1_re(c1_re), ic1_im(c1_im);
      VectorFEMassIntegrator ikl_re(klk_re), ikl_im(klk_im);

      struct U3Case
      {
         const char *name;
         int mask_cc, mask_mass;
         BilinearFormIntegrator *ire, *iim;
         bool need_kpar;
      };
      U3Case cases[5] = {
         {"curlcurl", ASM_CURLCURL, 0, &icc_re, &icc_im, false},
         {"mass", 0, ASM_MASS, &ims_re, &ims_im, false},
         {"cross0", 0, ASM_CROSS0, &ic0_re, &ic0_im, true},
         {"cross1", 0, ASM_CROSS1, &ic1_re, &ic1_im, true},
         {"klk", 0, ASM_KLK, &ikl_re, &ikl_im, true},
      };
      std::vector<DC> Ag;
      DenseMatrix mre, mim;
      for (const U3Case &uc : cases)
      {
         if (uc.need_kpar && !kpar) { continue; }
         MFEM_VERIFY(gpu.ComputeLocal(uc.mask_cc, uc.mask_mass, false),
                     "gpu local kernel failed");
         MFEM_VERIFY(gpu.DownloadLocal(Ag), "gpu download failed");
         double max_abs = 0.0, max_diff = 0.0;
         for (int e : sample)
         {
            const FiniteElement &fe = *fes.GetFE(e);
            ElementTransformation *tr = mesh.GetElementTransformation(e);
            uc.ire->AssembleElementMatrix(fe, *tr, mre);
            uc.iim->AssembleElementMatrix(fe, *tr, mim);
            const DC *Ae = Ag.data() + (size_t)e * nd * nd;
            for (int k = 0; k < nd; ++k)
               for (int l = 0; l < nd; ++l)
               {
                  const double ar = mre(k, l), ai = mim(k, l);
                  const DC g = Ae[(size_t)k * nd + l];
                  max_abs = std::max(max_abs, std::hypot(ar, ai));
                  max_diff = std::max(max_diff,
                                      std::hypot(g.re - ar, g.im - ai));
               }
         }
         std::printf("u3_%s_rel %.3e\n", uc.name, max_diff / max_abs);
      }

      // U4: full sum + dual DofTransformation, every element of the mesh
      const int mask_mass_all =
         ASM_MASS | (kpar ? ASM_CROSS0 | ASM_CROSS1 | ASM_KLK : 0);
      t0 = now_s();
      MFEM_VERIFY(gpu.ComputeLocal(ASM_CURLCURL, mask_mass_all, true),
                  "gpu local+te failed");
      std::printf("timing_gpu_local_s %.3f\n", now_s() - t0);
      MFEM_VERIFY(gpu.DownloadLocal(Ag), "gpu download failed");
      double max_abs = 0.0, max_diff = 0.0;
      int fo_seen = 0;
      Array<int> dofs;
      DofTransformation doftrans;
      DenseMatrix sre, sim;
      for (int e = 0; e < ad.ne; ++e)
      {
         const FiniteElement &fe = *fes.GetFE(e);
         ElementTransformation *tr = mesh.GetElementTransformation(e);
         sre.SetSize(nd);
         sim.SetSize(nd);
         sre = 0.0;
         sim = 0.0;
         for (const U3Case &uc : cases)
         {
            if (uc.need_kpar && !kpar) { continue; }
            uc.ire->AssembleElementMatrix(fe, *tr, mre);
            uc.iim->AssembleElementMatrix(fe, *tr, mim);
            sre += mre;
            sim += mim;
         }
         fes.GetElementDofs(e, dofs, doftrans);
         doftrans.TransformDual(sre);
         doftrans.TransformDual(sim);
         if (ad.p >= 2)
         {
            for (int f = 0; f < 4; ++f) { fo_seen |= 1 << ad.fo[(size_t)e * 4 + f]; }
         }
         const DC *Ae = Ag.data() + (size_t)e * nd * nd;
         for (int k = 0; k < nd; ++k)
            for (int l = 0; l < nd; ++l)
            {
               const double ar = sre(k, l), ai = sim(k, l);
               const DC g = Ae[(size_t)k * nd + l];
               max_abs = std::max(max_abs, std::hypot(ar, ai));
               max_diff = std::max(max_diff, std::hypot(g.re - ar, g.im - ai));
            }
      }
      std::printf("u4_rel %.3e\nu4_elems %d\nfo_coverage %d\n",
                  max_diff / max_abs, ad.ne, fo_seen);
   }

   if (do_gpu_global)
   {
      // ---- U5/U6: GPU global CSR + essential elimination vs MFEM ---------
      int devcount = 0;
      if (cudaGetDeviceCount(&devcount) != cudaSuccess || devcount == 0)
      {
         std::printf("asm_test: no CUDA device\n");
         return 4;
      }
      Array<int> ess_bdr(mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max()
                                                    : 0);
      ess_bdr = 1;
      Array<int> ess_tdof;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof);

      AsmGpuGlobal g;
      MFEM_VERIFY(g.Init(ad, 0), "gpu init failed");
      MFEM_VERIFY(g.BuildGlobal(ad, ess_tdof), "gpu global init failed");
      const bool kpar = prob.kx != 0.0 || prob.ky != 0.0;
      const int mm = ASM_MASS | (kpar ? ASM_CROSS0 | ASM_CROSS1 | ASM_KLK : 0);
      t0 = now_s();
      MFEM_VERIFY(g.ComputeLocal(ASM_CURLCURL, mm, true), "gpu local failed");
      MFEM_VERIFY(g.ScatterEliminate(false), "gpu scatter failed");
      std::printf("timing_gpu_assemble_s %.3f\n", now_s() - t0);
      std::vector<DC> gv;
      MFEM_VERIFY(g.DownloadValues(gv), "gpu download failed");

      // MFEM reference (production integrator set)
      LamInvPart lam_re(false), lam_im(true);
      MassPart mass_re(false), mass_im(true);
      CrossPart c0_re(0, false), c0_im(0, true), c1_re(1, false), c1_im(1, true);
      KLKPart klk_re(false), klk_im(true);
      const ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;
      SesquilinearForm a2(&fes, conv);
      a2.AddDomainIntegrator(new CurlCurlIntegrator(lam_re),
                             new CurlCurlIntegrator(lam_im));
      a2.AddDomainIntegrator(new VectorFEMassIntegrator(mass_re),
                             new VectorFEMassIntegrator(mass_im));
      if (kpar)
      {
         a2.AddDomainIntegrator(new MixedVectorCurlIntegrator(c0_re),
                                new MixedVectorCurlIntegrator(c0_im));
         a2.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(c1_re),
                                new MixedVectorWeakCurlIntegrator(c1_im));
         a2.AddDomainIntegrator(new VectorFEMassIntegrator(klk_re),
                                new VectorFEMassIntegrator(klk_im));
      }
      a2.Assemble(0);
      a2.real().Finalize(0);
      a2.imag().Finalize(0);
      a2.real().SpMat().SortColumnIndices();
      a2.imag().SpMat().SortColumnIndices();

      auto compare_csr = [&](const SparseMatrix &Ar, const SparseMatrix &Ai,
                             const std::vector<DC> &mine, const char *tag)
      {
         bool ok = Ar.Height() == ad.height &&
                   Ar.NumNonZeroElems() == (int)ad.Jcol.size();
         const int *I = Ar.GetI();
         const int *Jc = Ar.GetJ();
         for (int r = 0; r <= ad.height && ok; ++r) { ok = I[r] == ad.I[r]; }
         for (size_t k = 0; k < ad.Jcol.size() && ok; ++k)
         {
            ok = Jc[k] == ad.Jcol[k];
         }
         std::printf("%s_struct_ok %d\n", tag, ok ? 1 : 0);
         if (!ok) { return false; }
         const double *ar = Ar.GetData();
         const double *ai = Ai.GetData();
         double mabs = 0.0, mdiff = 0.0;
         for (size_t k = 0; k < ad.Jcol.size(); ++k)
         {
            mabs = std::max(mabs, std::hypot(ar[k], ai[k]));
            mdiff = std::max(mdiff, std::hypot(mine[k].re - ar[k],
                                               mine[k].im - ai[k]));
         }
         std::printf("%s_rel %.3e\n", tag, mdiff / mabs);
         return true;
      };
      if (!compare_csr(a2.real().SpMat(), a2.imag().SpMat(), gv, "u5"))
      {
         std::printf("asm_test: FAIL (u5 sparsity mismatch)\n");
         return 2;
      }

      // U6: eliminated system + RHS vs FormLinearSystem (synthetic RHS,
      // deterministic; X = 0 so only the essential-row zeroing acts on B)
      MFEM_VERIFY(g.ScatterEliminate(true), "gpu eliminate failed");
      MFEM_VERIFY(g.DownloadValues(gv), "gpu download failed");
      ComplexGridFunction u2(&fes);
      u2 = std::complex<real_t>(0.0, 0.0);
      ComplexLinearForm b2(&fes, conv);
      b2.Assemble();
      double *b2d = b2.GetData();
      for (int i = 0; i < b2.Size(); ++i)
      {
         b2d[i] = std::sin(0.7 * i) + 0.3 * std::cos(0.13 * i);
      }
      OperatorHandle A2;
      Vector X2, B2;
      a2.FormLinearSystem(ess_tdof, u2, b2, A2, X2, B2);
      ComplexSparseMatrix *Ac2 = A2.As<ComplexSparseMatrix>();
      Ac2->real().SortColumnIndices();
      Ac2->imag().SortColumnIndices();
      if (!compare_csr(Ac2->real(), Ac2->imag(), gv, "u6"))
      {
         std::printf("asm_test: FAIL (u6 sparsity mismatch)\n");
         return 2;
      }
      // my RHS construction: copy + zero essential rows (both halves)
      double b_diff = 0.0;
      {
         const int n = ad.height;
         Vector Bref(2 * n);
         std::memcpy(Bref.GetData(), b2.GetData(), sizeof(double) * 2 * n);
         for (int i = 0; i < ess_tdof.Size(); ++i)
         {
            Bref(ess_tdof[i]) = 0.0;
            Bref(n + ess_tdof[i]) = 0.0;
         }
         for (int i = 0; i < 2 * n; ++i)
         {
            b_diff = std::max(b_diff, std::abs(Bref(i) - B2(i)));
         }
      }
      std::printf("u6_b_max %.3e\n", b_diff);
   }

   std::printf("asm_test: OK\n");
   return 0;
}
