// M5: z-PML verification (docs/physics.md).
//
// Periodic box [0,L]^2 x [0,Z] with z-PML slabs at both ends (PEC outside).
// A uniform volumetric current slab (thickness d, mesh-aligned) radiates
// plane waves up/down; any wave coming back from the PML shows up as the
// counter-propagating amplitude in the fit  u(z) ~ a e^{i q z} + b e^{-i q z}
// sampled between source and PML, giving |r_PML| = |b/a| (top region).
//
// Bloch/oblique incidence via the envelope substitution E = u e^{i kpar x}:
// u is strictly periodic; the operator gains cross terms with K = kpar x.
// Optional metal slab near the bottom ("vacuum above metal substrate").
//
// Output: line "pml_reflection <|r|>" plus diagnostics.

#include "mfem.hpp"

#include <cmath>
#include <complex>
#include <cstdio>

using namespace mfem;

namespace
{

struct Params
{
   double k0 = 2.0 * M_PI / 1.0;  // wavelength = 1 (all lengths in lambda)
   double eps_re = 1.0, eps_im = 0.0;
   double theta_deg = 0.0;        // incidence angle (plane of incidence: xz)
   int pol = 0;                   // 0: s (E||y), 1: p (E in xz)
   double pml_thick = 1.0;        // in wavelengths
   int pml_order = 2;
   double target_ref = 1e-8;
   int metal = 0;                 // 1: metal slab at bottom of the interior
   int order = 3;
   int nx = 4;  // >= 4: periodic identification needs enough layers (see docs/gpu.md)
   int nz_per_wl = 8;             // elements per wavelength in z
};

Params P;

// geometry (units of wavelength): [pml][gap][<metal?>][src][gap][pml]
double z_pml_b, z_metal_top, z_src_lo, z_src_hi, z_pml_t, z_top;

std::complex<double> eps_at(double z)
{
   using C = std::complex<double>;
   // metal substrate extends through the bottom PML (material continuation)
   if (P.metal && z < z_metal_top) { return C(-20.0, 2.0); }
   return C(P.eps_re, P.eps_im);
}

// PML stretch s_z(z) = 1 + i sigma(z)/k0, polynomial profile
std::complex<double> s_z(double z)
{
   double d = -1.0;
   if (z < z_pml_b) { d = (z_pml_b - z) / P.pml_thick; }
   else if (z > z_pml_t) { d = (z - z_pml_t) / P.pml_thick; }
   if (d < 0.0) { return {1.0, 0.0}; }
   const double n_med = std::sqrt(std::abs(eps_at(z))); // decay uses Re(n)~n
   const double smax =
      -std::log(P.target_ref) * (P.pml_order + 1) /
      (2.0 * P.k0 * n_med * P.pml_thick);
   return {1.0, smax * std::pow(d, P.pml_order)};
}

// coefficient matrices at a point (complex), envelope form
// Lambda = diag(s, s, 1/s); Lambda^-1 = diag(1/s, 1/s, s)
void lam_inv(double z, std::complex<double> m[3])
{
   const std::complex<double> s = s_z(z);
   m[0] = 1.0 / s;
   m[1] = 1.0 / s;
   m[2] = s;
}

void mass_eps_lam(double z, std::complex<double> m[3])
{
   const std::complex<double> s = s_z(z);
   const std::complex<double> e = eps_at(z);
   m[0] = -P.k0 * P.k0 * e * s;
   m[1] = -P.k0 * P.k0 * e * s;
   m[2] = -P.k0 * P.k0 * e / s;
}

double kpar() { return P.k0 * std::sin(P.theta_deg * M_PI / 180.0); }

// K = kpar x  (cross-product matrix for kpar along x)
// K v = (0, -kp*v_z, kp*v_y)^T ... K = [[0,0,0],[0,0,-kp],[0,kp,0]]
void fill_K(DenseMatrix &K)
{
   K.SetSize(3);
   K = 0.0;
   K(1, 2) = -kpar();
   K(2, 1) = kpar();
}

// matrix coefficient helpers: real/imag parts of diagonal complex functions
class DiagPart : public MatrixCoefficient
{
public:
   using Fn = void (*)(double, std::complex<double>[3]);
   DiagPart(Fn f, bool imag) : MatrixCoefficient(3), f_(f), imag_(imag) {}
   void Eval(DenseMatrix &m, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      double x[3];
      Vector tx(x, 3);
      T.Transform(ip, tx);
      std::complex<double> d[3];
      f_(x[2], d);
      m.SetSize(3);
      m = 0.0;
      for (int i = 0; i < 3; ++i) { m(i, i) = imag_ ? d[i].imag() : d[i].real(); }
   }

private:
   Fn f_;
   bool imag_;
};

// full matrix product parts for cross terms: Q = i K * LamInv  (and i LamInv K)
class CrossPart : public MatrixCoefficient
{
public:
   // side: 0 -> i K LamInv (tested against v; use with MixedVectorCurlIntegrator)
   //       1 -> i LamInv K (acts on u; use with MixedVectorWeakCurlIntegrator)
   CrossPart(int side, bool imag) : MatrixCoefficient(3), side_(side), imag_(imag) {}
   void Eval(DenseMatrix &m, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      double x[3];
      Vector tx(x, 3);
      T.Transform(ip, tx);
      std::complex<double> li[3];
      lam_inv(x[2], li);
      DenseMatrix K;
      fill_K(K);
      m.SetSize(3);
      m = 0.0;
      // K * diag(li) or diag(li) * K, times i
      for (int r = 0; r < 3; ++r)
         for (int c = 0; c < 3; ++c)
         {
            const std::complex<double> v =
               std::complex<double>(0.0, 1.0) *
               (side_ == 0 ? K(r, c) * li[c] : li[r] * K(r, c));
            m(r, c) = imag_ ? v.imag() : v.real();
         }
   }

private:
   int side_;
   bool imag_;
};

// mass extra term: -K LamInv K (complex)
class KLKPart : public MatrixCoefficient
{
public:
   KLKPart(bool imag) : MatrixCoefficient(3), imag_(imag) {}
   void Eval(DenseMatrix &m, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      double x[3];
      Vector tx(x, 3);
      T.Transform(ip, tx);
      std::complex<double> li[3];
      lam_inv(x[2], li);
      DenseMatrix K;
      fill_K(K);
      m.SetSize(3);
      for (int r = 0; r < 3; ++r)
         for (int c = 0; c < 3; ++c)
         {
            std::complex<double> v(0.0, 0.0);
            for (int j = 0; j < 3; ++j) { v += K(r, j) * li[j] * K(j, c); }
            v = -v;
            m(r, c) = imag_ ? v.imag() : v.real();
         }
   }

private:
   bool imag_;
};

// source: envelope RHS  i k0 (Z0 J) in the source slab; J direction by pol
void src_re(const Vector &x, Vector &f)
{
   f.SetSize(3);
   f = 0.0;
   (void)x;
}

void src_im(const Vector &x, Vector &f)
{
   f.SetSize(3);
   f = 0.0;
   if (x(2) > z_src_lo && x(2) < z_src_hi)
   {
      if (P.pol == 0) { f(1) = P.k0; }        // i*k0*J_y -> imag part k0
      else { f(0) = P.k0; }                    // p-pol: J along x
   }
}

} // namespace

int main(int argc, char *argv[])
{
   OptionsParser args(argc, argv);
   args.AddOption(&P.eps_re, "-er", "--eps-re", "medium eps real part");
   args.AddOption(&P.eps_im, "-ei", "--eps-im", "medium eps imag part");
   args.AddOption(&P.theta_deg, "-t", "--theta", "incidence angle (deg)");
   args.AddOption(&P.pol, "-pol", "--pol", "0=s 1=p");
   args.AddOption(&P.pml_thick, "-pt", "--pml-thickness", "PML thickness (wl)");
   args.AddOption(&P.pml_order, "-po", "--pml-order", "sigma polynomial order");
   args.AddOption(&P.target_ref, "-tr", "--target-reflection", "target R");
   args.AddOption(&P.metal, "-metal", "--metal", "1: metal slab at bottom");
   args.AddOption(&P.order, "-p", "--order", "FE order");
   args.AddOption(&P.nz_per_wl, "-nz", "--nz-per-wl", "z elements per wl");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }

   // geometry (wavelength units): interior gap sizes
   const double gap = 1.5, metal_h = P.metal ? 0.75 : 0.0, src_h = 0.125;
   z_pml_b = P.pml_thick;
   z_metal_top = z_pml_b + metal_h;
   z_src_lo = z_metal_top + gap;
   z_src_hi = z_src_lo + src_h;
   z_pml_t = z_src_hi + gap;
   z_top = z_pml_t + P.pml_thick;

   const double lx = 0.5;
   // element-aligned z breakpoints: choose nz so every breakpoint is a multiple
   const double dz0 = 1.0 / P.nz_per_wl;
   const int nz = (int)std::round(z_top / (src_h < dz0 ? src_h : dz0));
   const double dz = z_top / nz;
   // snap breakpoints to the grid
   auto snap = [&](double &z) { z = std::round(z / dz) * dz; };
   snap(z_pml_b);
   snap(z_metal_top);
   snap(z_src_lo);
   snap(z_src_hi);
   snap(z_pml_t);

   Mesh serial = Mesh::MakeCartesian3D(P.nx, P.nx, nz, Element::TETRAHEDRON,
                                       lx, lx, z_top);
   std::vector<Vector> translations(2);
   translations[0].SetSize(3);
   translations[0] = 0.0;
   translations[0](0) = lx;
   translations[1].SetSize(3);
   translations[1] = 0.0;
   translations[1](1) = lx;
   Mesh mesh = Mesh::MakePeriodic(
      serial, serial.CreatePeriodicVertexMapping(translations));

   ND_FECollection fec(P.order, 3);
   FiniteElementSpace fes(&mesh, &fec);

   // PEC only on the z-end faces; the x/y boundary elements survive
   // MakePeriodic (vertices identified) and must NOT be constrained.
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      Array<int> vtx;
      mesh.GetBdrElementVertices(be, vtx);
      double zc = 0.0;
      // use the original (pre-periodic) copy's coordinates via serial mesh
      // is unavailable here; the periodic mesh keeps vertex coords, so:
      for (int j = 0; j < vtx.Size(); ++j) { zc += mesh.GetVertex(vtx[j])[2]; }
      zc /= vtx.Size();
      if (std::abs(zc) < 1e-9 || std::abs(zc - z_top) < 1e-9)
      {
         ess_bdr[mesh.GetBdrAttribute(be) - 1] = 1;
      }
   }
   Array<int> ess_tdof_list;
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   const ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;

   VectorFunctionCoefficient fr(3, src_re), fi(3, src_im);
   ComplexLinearForm b(&fes, conv);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fr),
                         new VectorFEDomainLFIntegrator(fi));
   b.Assemble();

   DiagPart lam_re(lam_inv, false), lam_im(lam_inv, true);
   DiagPart mass_re(mass_eps_lam, false), mass_im(mass_eps_lam, true);
   CrossPart c0_re(0, false), c0_im(0, true), c1_re(1, false), c1_im(1, true);
   KLKPart klk_re(false), klk_im(true);

   SesquilinearForm a(&fes, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(lam_re),
                         new CurlCurlIntegrator(lam_im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(mass_re),
                         new VectorFEMassIntegrator(mass_im));
   if (kpar() != 0.0)
   {
      a.AddDomainIntegrator(new MixedVectorCurlIntegrator(c0_re),
                            new MixedVectorCurlIntegrator(c0_im));
      a.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(c1_re),
                            new MixedVectorWeakCurlIntegrator(c1_im));
      a.AddDomainIntegrator(new VectorFEMassIntegrator(klk_re),
                            new VectorFEMassIntegrator(klk_im));
   }
   a.Assemble(0);  // skip_zeros=0: keep re/im sparsity identical for UMFPACK

   ComplexGridFunction u(&fes);
   u = std::complex<real_t>(0.0, 0.0);

   OperatorHandle A;
   Vector X, B;
   a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);
   ComplexSparseMatrix *Ac = A.As<ComplexSparseMatrix>();
   ComplexUMFPackSolver umf(true);  // long ints: avoid int32 workspace overflow
   // structurally symmetric matrix: symmetric strategy + best-of-AMD/METIS
   umf.Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
   umf.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD;
   umf.SetOperator(*Ac);
   umf.Mult(B, X);

   Vector R(B.Size());
   Ac->Mult(X, R);
   R -= B;
   std::printf("residual %.3e\n", R.Norml2() / B.Norml2());

   a.RecoverFEMSolution(X, b, u);

   // sample the relevant field component along z between source and top PML
   std::complex<double> qz = std::sqrt(
      std::complex<double>(P.k0 * P.k0 * P.eps_re - kpar() * kpar(),
                           P.k0 * P.k0 * P.eps_im));
   if (qz.imag() < 0.0) { qz = -qz; }
   const int nsamp = 200;
   const double za = z_src_hi + 0.15, zb = z_pml_t - 0.15;
   std::vector<std::complex<double>> vals(nsamp);
   std::vector<double> zs(nsamp);
   DenseMatrix point(3, 1);
   for (int i = 0; i < nsamp; ++i)
   {
      const double z = za + (zb - za) * i / (nsamp - 1.0);
      zs[i] = z;
      point(0, 0) = 0.2371 * lx;   // generic transverse spot
      point(1, 0) = 0.6173 * lx;
      point(2, 0) = z;
      Array<int> elem_ids;
      Array<IntegrationPoint> ips;
      mesh.FindPoints(point, elem_ids, ips);
      Vector vre(3), vim(3);
      u.real().GetVectorValue(elem_ids[0], ips[0], vre);
      u.imag().GetVectorValue(elem_ids[0], ips[0], vim);
      const int comp = (P.pol == 0) ? 1 : 0;
      vals[i] = {vre(comp), vim(comp)};
   }

   // least-squares fit  vals ~ a e^{i qz z} + b e^{-i qz z}
   std::complex<double> m00(0, 0), m01(0, 0), m11(0, 0), r0(0, 0), r1(0, 0);
   for (int i = 0; i < nsamp; ++i)
   {
      const std::complex<double> e1 = std::exp(std::complex<double>(0, 1) * qz * zs[i]);
      const std::complex<double> e2 = std::exp(std::complex<double>(0, -1) * qz * zs[i]);
      m00 += std::conj(e1) * e1;
      m01 += std::conj(e1) * e2;
      m11 += std::conj(e2) * e2;
      r0 += std::conj(e1) * vals[i];
      r1 += std::conj(e2) * vals[i];
   }
   const std::complex<double> det = m00 * m11 - m01 * std::conj(m01);
   const std::complex<double> ca = (m11 * r0 - m01 * r1) / det;
   const std::complex<double> cb = (m00 * r1 - std::conj(m01) * r0) / det;

   std::printf("up_amp %.6e down_amp %.6e\n", std::abs(ca), std::abs(cb));
   std::printf("pml_reflection %.6e\n", std::abs(cb) / std::abs(ca));
   return 0;
}
