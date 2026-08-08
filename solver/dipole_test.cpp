// M6b-2/3: open-domain local sources vs analytic Green functions.
//
// point mode: cube with PML on all six sides (full 3D stretch tensor
//   Lambda = diag(sy*sz/sx, sx*sz/sy, sx*sy/sz)); electric point dipole at
//   the centre; sampled field on a sphere r >= lambda/2 vs the free-space
//   dyadic Green function.
// line mode: y-periodic cell, PML in x and z; line current along y through
//   the centre; sampled field on a circle in the xz-plane vs the 2D Green
//   function  E_y = -(k0/4) (Z0 J) H0^(1)(k0 rho).
//
// Units: lengths in wavelengths, k0 = 2*pi; current normalized (Z0 J = 1).
// Output: "rel_l2 <err>" over the sample set.

#include "mfem.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>

using namespace mfem;
using C = std::complex<double>;

namespace
{

struct Params
{
   double k0 = 2.0 * M_PI;
   int mode = 0;          // 0: point dipole, 1: line current
   int orient = 0;        // dipole orientation: 0 x, 1 y, 2 z
   double inner = 2.0;    // inner box size (wavelengths)
   double pml_t = 0.6;
   int pml_order = 2;
   double target_ref = 1e-8;
   int order = 3;
   double epw = 5.0;      // elements per wavelength
   double r_samp = 0.5;   // sampling radius
};

Params P;
double box = 0.0;   // total box size, source at centre
double ly_line = 0.0;

C s_axis(double c)
{
   // stretch along one axis given the coordinate c in [0, box]
   double d = -1.0;
   if (c < P.pml_t) { d = (P.pml_t - c) / P.pml_t; }
   else if (c > box - P.pml_t) { d = (c - (box - P.pml_t)) / P.pml_t; }
   if (d < 0.0) { return {1.0, 0.0}; }
   const double smax = -std::log(P.target_ref) * (P.pml_order + 1) /
                       (2.0 * P.k0 * P.pml_t);
   return {1.0, smax * std::pow(d, P.pml_order)};
}

void stretches(const Vector &x, C s[3])
{
   s[0] = s_axis(x(0));
   s[1] = (P.mode == 0) ? s_axis(x(1)) : C(1.0, 0.0); // line: y periodic
   s[2] = s_axis(x(2));
}

// Lambda^-1 = diag(sx/(sy sz), sy/(sx sz), sz/(sx sy))
void lam_inv(const Vector &x, C m[3])
{
   C s[3];
   stretches(x, s);
   m[0] = s[0] / (s[1] * s[2]);
   m[1] = s[1] / (s[0] * s[2]);
   m[2] = s[2] / (s[0] * s[1]);
}

void lam(const Vector &x, C m[3])
{
   C s[3];
   stretches(x, s);
   m[0] = s[1] * s[2] / s[0];
   m[1] = s[0] * s[2] / s[1];
   m[2] = s[0] * s[1] / s[2];
}

class DiagPart : public MatrixCoefficient
{
public:
   using Fn = void (*)(const Vector &, C[3]);
   DiagPart(Fn f, bool imag, double scale = 1.0)
      : MatrixCoefficient(3), f_(f), imag_(imag), scale_(scale) {}
   void Eval(DenseMatrix &m, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector x(3);
      T.Transform(ip, x);
      C d[3];
      f_(x, d);
      m.SetSize(3);
      m = 0.0;
      for (int i = 0; i < 3; ++i)
      {
         const C v = scale_ * d[i];
         m(i, i) = imag_ ? v.imag() : v.real();
      }
   }

private:
   Fn f_;
   bool imag_;
   double scale_;
};

bool find_point(Mesh &mesh, const double x[3], int &elem, IntegrationPoint &ip)
{
   DenseMatrix pts(3, 1);
   pts(0, 0) = x[0];
   pts(1, 0) = x[1];
   pts(2, 0) = x[2];
   Array<int> e1;
   Array<IntegrationPoint> ip1;
   mesh.FindPoints(pts, e1, ip1, false);
   if (e1[0] < 0) { return false; }
   elem = e1[0];
   ip = ip1[0];
   return true;
}

void accumulate_at(FiniteElementSpace &fes, Mesh &mesh, const double x[3],
                   const C amp[3], LinearForm &br, LinearForm &bi)
{
   int elem;
   IntegrationPoint ip;
   MFEM_VERIFY(find_point(mesh, x, elem, ip), "source point not found");
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

// free-space dyadic Green: E_i = i k0 (Z0 Il)_j [ (delta_ij + didj/k^2) g ]
void dipole_field(const double rvec[3], int orient, C e[3])
{
   const double r = std::sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] +
                              rvec[2] * rvec[2]);
   const double k = P.k0;
   const double kr = k * r;
   const C g = std::exp(C(0.0, kr)) / (4.0 * M_PI * r);
   // (I + grad grad / k^2) g = g [A delta_ij + B rhat_i rhat_j] with
   // A = 1 + i/kr - 1/kr^2,  B = -1 - 3i/kr + 3/kr^2   (Jackson 9.18)
   const C a = C(1.0 - 1.0 / (kr * kr), 1.0 / kr);
   const C b = C(-1.0 + 3.0 / (kr * kr), -3.0 / kr);
   for (int i = 0; i < 3; ++i)
   {
      const double rh_i = rvec[i] / r, rh_j = rvec[orient] / r;
      const C G = g * (a * (i == orient ? 1.0 : 0.0) + b * rh_i * rh_j);
      e[i] = C(0.0, k) * G;
   }
}

} // namespace

int main(int argc, char *argv[])
{
   OptionsParser args(argc, argv);
   args.AddOption(&P.mode, "-mode", "--mode", "0 point, 1 line");
   args.AddOption(&P.orient, "-or", "--orient", "dipole orientation 0/1/2");
   args.AddOption(&P.order, "-p", "--order", "FE order");
   args.AddOption(&P.epw, "-e", "--epw", "elements per wavelength");
   args.AddOption(&P.inner, "-in", "--inner", "inner box (wavelengths)");
   args.AddOption(&P.pml_t, "-pt", "--pml", "PML thickness (wavelengths)");
   args.AddOption(&P.r_samp, "-r", "--r-samp", "sampling radius (wavelengths)");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }

   box = P.inner + 2.0 * P.pml_t;
   const double h = 1.0 / P.epw;
   const int n = std::max(6, (int)std::round(box / h));
   const int ny = (P.mode == 0) ? n : 4;
   ly_line = (P.mode == 0) ? box : 4.0 * h;

   Mesh serial = Mesh::MakeCartesian3D(n, ny, n, Element::TETRAHEDRON,
                                       box, ly_line, box);
   Mesh mesh = [&]()
   {
      if (P.mode == 0) { return std::move(serial); }
      std::vector<Vector> tr(1);
      tr[0].SetSize(3);
      tr[0] = 0.0;
      tr[0](1) = ly_line;
      return Mesh::MakePeriodic(serial,
                                serial.CreatePeriodicVertexMapping(tr));
   }();

   ND_FECollection fec(P.order, 3);
   FiniteElementSpace fes(&mesh, &fec);
   std::printf("ndof %d\n", fes.GetTrueVSize());

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   if (P.mode == 0) { ess_bdr = 1; }
   else
   {
      // PEC only on x/z ends (y is periodic; its bdr elements are identified)
      ess_bdr = 0;
      for (int be = 0; be < mesh.GetNBE(); ++be)
      {
         Array<int> vtx;
         mesh.GetBdrElementVertices(be, vtx);
         double cx = 0.0, cz = 0.0;
         for (int j = 0; j < vtx.Size(); ++j)
         {
            cx += mesh.GetVertex(vtx[j])[0];
            cz += mesh.GetVertex(vtx[j])[2];
         }
         cx /= vtx.Size();
         cz /= vtx.Size();
         const bool xend = std::abs(cx) < 1e-9 || std::abs(cx - box) < 1e-9;
         const bool zend = std::abs(cz) < 1e-9 || std::abs(cz - box) < 1e-9;
         if (xend || zend) { ess_bdr[mesh.GetBdrAttribute(be) - 1] = 1; }
      }
   }
   Array<int> ess_tdof_list;
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   const ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;

   ComplexLinearForm b(&fes, conv);
   b.Assemble();
   const double ctr = box / 2.0;
   // keep the point source strictly inside an element: a delta on a mesh
   // face/vertex is one-sided for ND bases (normal components jump there)
   const double hh = box / n;
   const double src[3] = {ctr + 0.31 * hh, ctr + 0.17 * hh, ctr + 0.243 * hh};
   if (P.mode == 0)
   {
      const double x0[3] = {src[0], src[1], src[2]};
      C amp[3] = {C(0, 0), C(0, 0), C(0, 0)};
      amp[P.orient] = C(0.0, P.k0); // i k0 (Z0 Il)
      accumulate_at(fes, mesh, x0, amp, b.real(), b.imag());
   }
   else
   {
      const int nq = 400;
      const double w = ly_line / nq;
      for (int q = 0; q < nq; ++q)
      {
         const double x0[3] = {ctr, (q + 0.5) * w, ctr};
         C amp[3] = {C(0, 0), C(0.0, P.k0 * w), C(0, 0)};
         accumulate_at(fes, mesh, x0, amp, b.real(), b.imag());
      }
   }

   DiagPart lam_re(lam_inv, false), lam_im(lam_inv, true);
   DiagPart mass_re(lam, false, 1.0), mass_im(lam, true, 1.0);
   // mass coefficient must be -k0^2 * Lambda (vacuum eps = 1)
   DiagPart mass2_re(lam, false, -P.k0 * P.k0), mass2_im(lam, true, -P.k0 * P.k0);

   SesquilinearForm a(&fes, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(lam_re),
                         new CurlCurlIntegrator(lam_im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(mass2_re),
                         new VectorFEMassIntegrator(mass2_im));
   a.Assemble(0);

   ComplexGridFunction u(&fes);
   u = std::complex<real_t>(0.0, 0.0);
   OperatorHandle A;
   Vector X, B;
   a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);
   ComplexSparseMatrix *Ac = A.As<ComplexSparseMatrix>();
   ComplexUMFPackSolver umf(true);
   umf.Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
   umf.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD;
   umf.SetOperator(*Ac);
   umf.Mult(B, X);
   a.RecoverFEMSolution(X, b, u);

   // sample and compare
   double num2 = 0.0, den2 = 0.0, nnum2 = 0.0;
   C corr(0, 0), corr_c(0, 0);
   const int ns = 200;
   for (int i = 0; i < ns; ++i)
   {
      double xs[3];
      C eref[3];
      if (P.mode == 0)
      {
         // spiral point set on the sphere of radius r_samp
         const double th = std::acos(1.0 - 2.0 * (i + 0.5) / ns);
         const double phi = M_PI * (1.0 + std::sqrt(5.0)) * i;
         const double rv[3] = {P.r_samp * std::sin(th) * std::cos(phi),
                               P.r_samp * std::sin(th) * std::sin(phi),
                               P.r_samp * std::cos(th)};
         xs[0] = src[0] + rv[0];
         xs[1] = src[1] + rv[1];
         xs[2] = src[2] + rv[2];
         dipole_field(rv, P.orient, eref);
      }
      else
      {
         const double phi = 2.0 * M_PI * (i + 0.5) / ns;
         const double rv[2] = {P.r_samp * std::cos(phi), P.r_samp * std::sin(phi)};
         xs[0] = ctr + rv[0];
         xs[1] = 0.5 * ly_line + 1e-4;
         xs[2] = ctr + rv[1];
         const double rho = P.r_samp;
         const double J0 = std::cyl_bessel_j(0.0, P.k0 * rho);
         const double Y0 = std::cyl_neumann(0.0, P.k0 * rho);
         eref[0] = C(0, 0);
         eref[1] = -(P.k0 / 4.0) * C(J0, Y0);
         eref[2] = C(0, 0);
      }
      int elem;
      IntegrationPoint ip;
      if (!find_point(mesh, xs, elem, ip)) { continue; }
      Vector vr(3), vi(3);
      u.real().GetVectorValue(elem, ip, vr);
      u.imag().GetVectorValue(elem, ip, vi);
      for (int d = 0; d < 3; ++d)
      {
         const C num = C(vr(d), vi(d)) - eref[d];
         num2 += std::norm(num);
         nnum2 += std::norm(C(vr(d), vi(d)));
         den2 += std::norm(eref[d]);
         corr += C(vr(d), vi(d)) * std::conj(eref[d]);
         corr_c += std::conj(C(vr(d), vi(d))) * std::conj(eref[d]);
      }
   }
   std::printf("corr %.3f%+.3fi  corr_conj %.3f%+.3fi\n",
               corr.real() / std::sqrt(nnum2 * den2),
               corr.imag() / std::sqrt(nnum2 * den2),
               corr_c.real() / std::sqrt(nnum2 * den2),
               corr_c.imag() / std::sqrt(nnum2 * den2));
   std::printf("norm_num %.4e norm_ref %.4e\n",
               std::sqrt(nnum2), std::sqrt(den2));
   std::printf("rel_l2 %.6e\n", std::sqrt(num2 / den2));
   return 0;
}
