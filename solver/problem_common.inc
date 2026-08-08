// Shared problem definition for the production solver and the v2.5 assembly
// test binary (asm_test): Problem/Region structs, PML stretch, the five
// coefficient classes of the sesquilinear form, incident-field tables, and
// the core solve.json parsing. Included inside an anonymous namespace of
// each translation unit (same pattern as cudss_solver.inc). Both binaries
// MUST evaluate coefficients through this exact code so that the v2.5
// reassembly comparisons are against the production discretization.

C jc(const json &v) { return C(v[0].get<double>(), v[1].get<double>()); }

struct SlabWave
{
   double z_lo, z_hi, zA, zB;
   C qA, qB;
   C A[3], B[3];
};

struct Region
{
   int slab = -1;          // -1 for pml_bottom, n_slab for pml_top
   C eps;
   bool is_frustum = false;
   C delta_eps;            // eps - eps_bg(slab), frustum regions only
   bool is_pml = false;
   double z0 = 0.0, z1 = 0.0; // pml slab extent
   bool pml_top = false;
};

struct LocalSource
{
   int type = 0;            // 0: point, 1: line, 2: sheet
   int source_index = -1;
   double p0[3] = {0, 0, 0};
   double p1[3] = {0, 0, 0};  // line endpoint / sheet edge1
   double e2[3] = {0, 0, 0};  // sheet edge2
   C cur[3];
   double pg[2] = {0, 0};     // phase gradient (line: [0]; sheet: both)
};

struct Problem
{
   double k0 = 0.0;
   double lx = 0.0, ly = 0.0, z_min = 0.0, z_max = 0.0;
   double kx = 0.0, ky = 0.0;
   int order = 3;
   double pml_order = 2.0;
   double pml_target = 1e-8;
   double pml_thick = 0.0;
   std::vector<Region> regions;                 // by attribute-1
   std::vector<std::vector<SlabWave>> incident; // parts x slabs
   std::vector<int> incident_src;               // source index per part
   std::vector<LocalSource> local;              // local sources of the group
   std::vector<int> group_sources;              // source indices in the group
   bool per_source = false;
   std::vector<int> active_parts;               // incident part filter (RHS)
   std::vector<int> active_local;               // local source filter (RHS)
   std::string solver_type = "direct";          // direct | iterative
   std::string device = "cpu";
   int gpu_id = 0;
   double it_rtol = 1e-8;
   int it_maxit = 2000;
   double gpu_mem_gb = 0.0;  // cuDSS VRAM cap (V2-M3); 0 = auto (free VRAM)
   std::vector<double> plane_z;
   std::vector<int> plane_nx, plane_ny;
};

Problem prob;

C s_z(double z)
{
   double d = -1.0, t = prob.pml_thick;
   double n_med = 1.0;
   if (z < prob.z_min)
   {
      d = (prob.z_min - z) / t;
      for (const Region &r : prob.regions)
         if (r.is_pml && !r.pml_top) { n_med = std::abs(std::sqrt(r.eps)); }
   }
   else if (z > prob.z_max)
   {
      d = (z - prob.z_max) / t;
      for (const Region &r : prob.regions)
         if (r.is_pml && r.pml_top) { n_med = std::abs(std::sqrt(r.eps)); }
   }
   if (d < 0.0) { return {1.0, 0.0}; }
   if (d > 1.0) { d = 1.0; }
   const double smax = -std::log(prob.pml_target) * (prob.pml_order + 1) /
                       (2.0 * prob.k0 * n_med * prob.pml_thick);
   return {1.0, smax * std::pow(d, prob.pml_order)};
}

void lam_inv_diag(double z, C m[3])
{
   const C s = s_z(z);
   m[0] = 1.0 / s;
   m[1] = 1.0 / s;
   m[2] = s;
}

// ---- matrix coefficients ----------------------------------------------

class LamInvPart : public MatrixCoefficient
{
public:
   LamInvPart(bool imag) : MatrixCoefficient(3), imag_(imag) {}
   void Eval(DenseMatrix &m, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      double x[3];
      Vector tx(x, 3);
      T.Transform(ip, tx);
      C d[3];
      lam_inv_diag(x[2], d);
      m.SetSize(3);
      m = 0.0;
      for (int i = 0; i < 3; ++i) { m(i, i) = imag_ ? d[i].imag() : d[i].real(); }
   }

private:
   bool imag_;
};

class MassPart : public MatrixCoefficient
{
public:
   MassPart(bool imag) : MatrixCoefficient(3), imag_(imag) {}
   void Eval(DenseMatrix &m, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      double x[3];
      Vector tx(x, 3);
      T.Transform(ip, tx);
      const C s = s_z(x[2]);
      const C eps = prob.regions[T.Attribute - 1].eps;
      const double k2 = prob.k0 * prob.k0;
      const C lam[3] = {s, s, 1.0 / s};
      m.SetSize(3);
      m = 0.0;
      for (int i = 0; i < 3; ++i)
      {
         const C v = -k2 * eps * lam[i];
         m(i, i) = imag_ ? v.imag() : v.real();
      }
   }

private:
   bool imag_;
};

void fill_K(DenseMatrix &K)
{
   K.SetSize(3);
   K = 0.0;
   K(0, 2) = prob.ky;
   K(1, 2) = -prob.kx;
   K(2, 0) = -prob.ky;
   K(2, 1) = prob.kx;
}

class CrossPart : public MatrixCoefficient
{
public:
   CrossPart(int side, bool imag) : MatrixCoefficient(3), side_(side), imag_(imag) {}
   void Eval(DenseMatrix &m, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      double x[3];
      Vector tx(x, 3);
      T.Transform(ip, tx);
      C li[3];
      lam_inv_diag(x[2], li);
      DenseMatrix K;
      fill_K(K);
      m.SetSize(3);
      for (int r = 0; r < 3; ++r)
         for (int c = 0; c < 3; ++c)
         {
            const C v = C(0.0, 1.0) *
                        (side_ == 0 ? K(r, c) * li[c] : li[r] * K(r, c));
            m(r, c) = imag_ ? v.imag() : v.real();
         }
   }

private:
   int side_;
   bool imag_;
};

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
      C li[3];
      lam_inv_diag(x[2], li);
      DenseMatrix K;
      fill_K(K);
      m.SetSize(3);
      for (int r = 0; r < 3; ++r)
         for (int c = 0; c < 3; ++c)
         {
            C v(0.0, 0.0);
            for (int j = 0; j < 3; ++j) { v += K(r, j) * li[j] * K(j, c); }
            v = -v;
            m(r, c) = imag_ ? v.imag() : v.real();
         }
   }

private:
   bool imag_;
};

// incident envelope Ehat(z) for a slab index, summed over parts
void incident_at(int slab, double z, C e[3])
{
   e[0] = e[1] = e[2] = C(0, 0);
   for (size_t pi = 0; pi < prob.incident.size(); ++pi)
   {
      if (!prob.active_parts.empty() &&
          std::find(prob.active_parts.begin(), prob.active_parts.end(),
                    (int)pi) == prob.active_parts.end())
      {
         continue;
      }
      const auto &part = prob.incident[pi];
      const SlabWave &sw = part[slab];
      const C up = std::exp(C(0, 1) * sw.qA * (z - sw.zA));
      const C dn = std::exp(C(0, 1) * sw.qB * (z - sw.zB));
      for (int i = 0; i < 3; ++i) { e[i] += sw.A[i] * up + sw.B[i] * dn; }
   }
}

// RHS: f_env = k0^2 (eps - eps_bg) Ehat(z) in frustum regions
class ScatterSource : public VectorCoefficient
{
public:
   using VectorCoefficient::Eval;
   ScatterSource(bool imag) : VectorCoefficient(3), imag_(imag) {}
   void Eval(Vector &f, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      f.SetSize(3);
      f = 0.0;
      const Region &reg = prob.regions[T.Attribute - 1];
      if (!reg.is_frustum) { return; }
      double x[3];
      Vector tx(x, 3);
      T.Transform(ip, tx);
      C e[3];
      incident_at(reg.slab, x[2], e);
      const C fac = prob.k0 * prob.k0 * reg.delta_eps;
      for (int i = 0; i < 3; ++i)
      {
         const C v = fac * e[i];
         f(i) = imag_ ? v.imag() : v.real();
      }
   }

private:
   bool imag_;
};

// Core solve.json parsing: domain/fem/pml scalars, region table, and the
// group's kpar. Sources, incident tables, solver options and output planes
// stay with the production main() (asm_test does not need them).
void parse_problem_core(const json &doc, int group)
{
   prob.k0 = doc["k0"].get<double>();
   prob.lx = doc["domain"]["Lx"].get<double>();
   prob.ly = doc["domain"]["Ly"].get<double>();
   prob.z_min = doc["domain"]["z_min"].get<double>();
   prob.z_max = doc["domain"]["z_max"].get<double>();
   prob.order = doc["fem"]["order"].get<int>();
   prob.pml_order = doc["pml"]["order"].get<double>();
   prob.pml_target = doc["pml"]["target_reflection"].get<double>();
   prob.pml_thick =
      doc["pml"]["thickness_wavelengths"].get<double>() * doc["wavelength"].get<double>();

   const auto &slabs_j = doc["slabs"];
   const int n_slab = (int)slabs_j.size() - 1;

   // background eps per slab (for delta_eps of frustum regions)
   std::vector<C> eps_bg(n_slab);
   for (const auto &rj : doc["regions"])
   {
      if (rj["kind"] == "background")
      {
         eps_bg[rj["slab"].get<int>()] = jc(rj["epsilon"]);
      }
   }
   for (const auto &rj : doc["regions"])
   {
      Region r;
      r.eps = jc(rj["epsilon"]);
      const std::string kind = rj["kind"].get<std::string>();
      if (kind == "pml_bottom" || kind == "pml_top")
      {
         r.is_pml = true;
         r.pml_top = kind == "pml_top";
         r.z0 = rj["z"][0].get<double>();
         r.z1 = rj["z"][1].get<double>();
         r.slab = r.pml_top ? n_slab - 1 : 0;
      }
      else
      {
         r.slab = rj["slab"].get<int>();
         if (kind == "frustum")
         {
            r.is_frustum = true;
            r.delta_eps = r.eps - eps_bg[r.slab];
         }
      }
      prob.regions.push_back(r);
   }

   const auto &gj = doc["groups"][group];
   prob.kx = gj["kpar"][0].get<double>();
   prob.ky = gj["kpar"][1].get<double>();
}
