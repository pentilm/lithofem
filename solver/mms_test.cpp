// M4: Method-of-manufactured-solutions verification of the complex
// curl-curl assembly core (docs/physics.md).
//
// Solves  curl(curl E) - k0^2 eps E = f  on the unit cube with
// non-homogeneous PEC boundary (tangential trace of E_exact), where
// E_exact is a smooth manufactured field and f is derived analytically.
// eps is a 3x3 complex tensor (cases: real scalar, complex scalar,
// complex symmetric tensor). The curl-curl coefficient uses the full
// matrix-coefficient infrastructure with Lambda = I (PML plugs in later).
//
// Output (machine-readable): one line per refinement level
//   level <l> ndof <n> h <h> err <L2err>
// plus at the finest level:
//   symmetry <rel_asym>   residual <rel_res>
//
// Usage: mms_test -p <order> -e <real|complex|tensor> -n <n0> -l <levels>

#include "mfem.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>

using namespace mfem;

namespace
{

constexpr double K0 = 3.0; // arbitrary wavenumber on the unit box

// permittivity tensor (complex symmetric), selected by case name
std::complex<double> eps_tensor[3][3];

void set_eps_case(const char *name)
{
   using C = std::complex<double>;
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) { eps_tensor[i][j] = C(0.0, 0.0); }
   if (std::strcmp(name, "real") == 0)
   {
      for (int i = 0; i < 3; ++i) { eps_tensor[i][i] = C(2.25, 0.0); }
   }
   else if (std::strcmp(name, "complex") == 0)
   {
      for (int i = 0; i < 3; ++i) { eps_tensor[i][i] = C(2.25, 0.75); }
   }
   else if (std::strcmp(name, "tensor") == 0)
   {
      const C t[3][3] = {{C(2.3, 0.4), C(0.2, 0.05), C(0.1, 0.0)},
                         {C(0.2, 0.05), C(2.0, 0.3), C(0.15, 0.02)},
                         {C(0.1, 0.0), C(0.15, 0.02), C(1.8, 0.5)}};
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j) { eps_tensor[i][j] = t[i][j]; }
   }
   else
   {
      mfem_error("unknown eps case");
   }
}

// Divergence-free part S: each component independent of its own coordinate,
// so curl(curl S) = -Laplacian(S) with per-component factor (a^2 + b^2).
// Real part uses (ka, kb) = (pi, 2pi)-style integers; imag part differs.
struct SField
{
   double a[3], b[3]; // wavenumbers per component

   void eval(const Vector &x, Vector &S) const
   {
      S(0) = sin(a[0] * x(1)) * sin(b[0] * x(2));
      S(1) = sin(a[1] * x(2)) * sin(b[1] * x(0));
      S(2) = sin(a[2] * x(0)) * sin(b[2] * x(1));
   }
   void curlcurl(const Vector &x, Vector &cc) const
   {
      Vector S(3);
      eval(x, S);
      for (int i = 0; i < 3; ++i) { cc(i) = (a[i] * a[i] + b[i] * b[i]) * S(i); }
   }
};

// gradient part grad(phi), phi = sin(c0 x) sin(c1 y) sin(c2 z): curl-free
struct GradField
{
   double c[3];

   void eval(const Vector &x, Vector &G) const
   {
      G(0) = c[0] * cos(c[0] * x(0)) * sin(c[1] * x(1)) * sin(c[2] * x(2));
      G(1) = c[1] * sin(c[0] * x(0)) * cos(c[1] * x(1)) * sin(c[2] * x(2));
      G(2) = c[2] * sin(c[0] * x(0)) * sin(c[1] * x(1)) * cos(c[2] * x(2));
   }
};

const double PI = M_PI;
const SField S_re = {{PI, 2 * PI, PI}, {2 * PI, PI, PI}};
const SField S_im = {{2 * PI, PI, 2 * PI}, {PI, PI, 2 * PI}};
const GradField G_re = {{PI, PI, PI}};
const GradField G_im = {{2 * PI, PI, PI}};

void E_re(const Vector &x, Vector &E)
{
   Vector S(3), G(3);
   S_re.eval(x, S);
   G_re.eval(x, G);
   for (int i = 0; i < 3; ++i) { E(i) = S(i) + G(i); }
}

void E_im(const Vector &x, Vector &E)
{
   Vector S(3), G(3);
   S_im.eval(x, S);
   G_im.eval(x, G);
   for (int i = 0; i < 3; ++i) { E(i) = S(i) + G(i); }
}

// f = curl(curl E) - k0^2 eps E, split into real/imag parts
void f_part(const Vector &x, Vector &f, bool imag)
{
   Vector ccr(3), cci(3), er(3), ei(3);
   S_re.curlcurl(x, ccr);
   S_im.curlcurl(x, cci);
   E_re(x, er);
   E_im(x, ei);
   for (int i = 0; i < 3; ++i)
   {
      std::complex<double> epsE(0.0, 0.0);
      for (int j = 0; j < 3; ++j)
      {
         epsE += eps_tensor[i][j] * std::complex<double>(er(j), ei(j));
      }
      const std::complex<double> cc(ccr(i), cci(i));
      const std::complex<double> val = cc - K0 * K0 * epsE;
      f(i) = imag ? val.imag() : val.real();
   }
}

void f_re(const Vector &x, Vector &f) { f_part(x, f, false); }
void f_im(const Vector &x, Vector &f) { f_part(x, f, true); }

double frob(const SparseMatrix &m)
{
   double s = 0.0;
   const double *data = m.GetData();
   for (int i = 0; i < m.NumNonZeroElems(); ++i) { s += data[i] * data[i]; }
   return sqrt(s);
}

// relative Frobenius asymmetry of a sparse matrix
double rel_asymmetry(const SparseMatrix &m)
{
   SparseMatrix *mt = Transpose(m);
   SparseMatrix *diff = Add(1.0, m, -1.0, *mt);
   const double r = frob(*diff) / frob(m);
   delete diff;
   delete mt;
   return r;
}

} // namespace

int main(int argc, char *argv[])
{
   int order = 2;
   int n0 = 2;
   int levels = 3;
   const char *eps_case = "real";
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-p", "--order", "Nedelec order (any positive int)");
   args.AddOption(&n0, "-n", "--n0", "coarsest cells per direction");
   args.AddOption(&levels, "-l", "--levels", "number of refinement levels");
   args.AddOption(&eps_case, "-e", "--eps", "real | complex | tensor");
   args.AddOption(&device_config, "-d", "--device", "device (cpu/cuda)");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }

   Device device(device_config);
   set_eps_case(eps_case);

   // matrix coefficients: Lambda^-1 = I (real), -k0^2 eps split re/im
   DenseMatrix lam_inv_re(3);
   lam_inv_re = 0.0;
   for (int i = 0; i < 3; ++i) { lam_inv_re(i, i) = 1.0; }
   DenseMatrix mass_re(3), mass_im(3);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
         mass_re(i, j) = -K0 * K0 * eps_tensor[i][j].real();
         mass_im(i, j) = -K0 * K0 * eps_tensor[i][j].imag();
      }

   for (int lev = 0; lev < levels; ++lev)
   {
      const int n = n0 << lev;
      Mesh mesh = Mesh::MakeCartesian3D(n, n, n, Element::TETRAHEDRON,
                                        1.0, 1.0, 1.0);
      ND_FECollection fec(order, mesh.Dimension());
      FiniteElementSpace fes(&mesh, &fec);

      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      Array<int> ess_tdof_list;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      const ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;

      VectorFunctionCoefficient f_re_coeff(3, f_re);
      VectorFunctionCoefficient f_im_coeff(3, f_im);
      ComplexLinearForm b(&fes, conv);
      b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_re_coeff),
                            new VectorFEDomainLFIntegrator(f_im_coeff));
      b.Assemble();

      VectorFunctionCoefficient e_re_coeff(3, E_re);
      VectorFunctionCoefficient e_im_coeff(3, E_im);
      ComplexGridFunction e(&fes);
      e = std::complex<real_t>(0.0, 0.0);
      e.ProjectBdrCoefficientTangent(e_re_coeff, e_im_coeff, ess_bdr);

      MatrixConstantCoefficient lam_inv_re_c(lam_inv_re);
      MatrixConstantCoefficient mass_re_c(mass_re);
      MatrixConstantCoefficient mass_im_c(mass_im);

      SesquilinearForm a(&fes, conv);
      a.AddDomainIntegrator(new CurlCurlIntegrator(lam_inv_re_c), nullptr);
      a.AddDomainIntegrator(new VectorFEMassIntegrator(mass_re_c),
                            new VectorFEMassIntegrator(mass_im_c));
      a.Assemble(0);  // skip_zeros=0: keep re/im sparsity identical for UMFPACK

      OperatorHandle A;
      Vector X, B;
      a.FormLinearSystem(ess_tdof_list, e, b, A, X, B);

      ComplexSparseMatrix *Ac = A.As<ComplexSparseMatrix>();

      ComplexUMFPackSolver umf(true);  // long ints: avoid int32 workspace overflow
   // structurally symmetric matrix: symmetric strategy + best-of-AMD/METIS
   umf.Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
   umf.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD;
   umf.SetOperator(*Ac);
      umf.Mult(B, X);

      // residual ||Ax - b|| / ||b||
      Vector R(B.Size());
      Ac->Mult(X, R);
      R -= B;
      const double rel_res = R.Norml2() / B.Norml2();

      a.RecoverFEMSolution(X, b, e);

      const double err_r = e.real().ComputeL2Error(e_re_coeff);
      const double err_i = e.imag().ComputeL2Error(e_im_coeff);
      const double err = sqrt(err_r * err_r + err_i * err_i);

      std::printf("level %d ndof %d h %.8e err %.12e\n",
                  lev, fes.GetTrueVSize(), 1.0 / n, err);

      if (lev == levels - 1)
      {
         const double asym = std::max(rel_asymmetry(Ac->real()),
                                      Ac->imag().NumNonZeroElems() > 0
                                      ? rel_asymmetry(Ac->imag()) : 0.0);
         std::printf("symmetry %.3e\n", asym);
         std::printf("residual %.3e\n", rel_res);
      }
   }
   return 0;
}
