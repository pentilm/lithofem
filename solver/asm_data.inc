// v2.5 GPU assembly: host-side data extraction + CPU reference reassembler
// (docs/gpu.md, milestone M1). Included inside an anonymous namespace
// after problem_common.inc (needs Problem/prob, s_z, lam_inv_diag, fill_K).
//
// The extracted arrays use the exact layout the GPU kernels will consume
// (flat, element-major, fixed strides), and the CPU reference reassembler
// performs the same operations in the same order as the future device code.
// Correctness anchor: reassemble() must match MFEM's SesquilinearForm
// ::Assemble(0) element-wise (rel < 1e-13, unit test U7) — this pins down
// every extracted quantity (dofs, signs, orientations, geometry, coefficient
// tables, reference basis tables) before any GPU code is written.

// ---- ND tetrahedron face-dof 2x2 transformation tables ------------------
// Copied from mfem::ND_DofTransformation (fem/doftrans.cpp); each block is
// column-major (m00, m10, m01, m11), indexed by the face orientation 0..5.
// TransformDual applies TInv(Fo)^T to each face-dof pair.
const double asm_nd_T[24] =
{
   1.0,  0.0,  0.0,  1.0,
   -1.0, -1.0,  0.0,  1.0,
   0.0,  1.0, -1.0, -1.0,
   1.0,  0.0, -1.0, -1.0,
   -1.0, -1.0,  1.0,  0.0,
   0.0,  1.0,  1.0,  0.0
};
const double asm_nd_TInv[24] =
{
   1.0,  0.0,  0.0,  1.0,
   -1.0, -1.0,  0.0,  1.0,
   -1.0, -1.0,  1.0,  0.0,
   1.0,  0.0, -1.0, -1.0,
   0.0,  1.0, -1.0, -1.0,
   0.0,  1.0,  1.0,  0.0
};

// ---- reference tables (per order, per integration rule) -----------------

struct AsmRefRule
{
   int nq = 0;
   std::vector<double> ip;     // nq*3 reference coordinates
   std::vector<double> w;      // nq weights
   std::vector<double> vshape; // nq*nd*3, [q*nd*3 + k*3 + d] CalcVShape(ip)
   std::vector<double> curl;   // nq*nd*3, CalcCurlShape(ip)
};

void extract_ref_rule(const FiniteElement &fe, const IntegrationRule &ir,
                      AsmRefRule &r)
{
   const int nd = fe.GetDof();
   r.nq = ir.GetNPoints();
   r.ip.resize((size_t)r.nq * 3);
   r.w.resize(r.nq);
   r.vshape.assign((size_t)r.nq * nd * 3, 0.0);
   r.curl.assign((size_t)r.nq * nd * 3, 0.0);
   DenseMatrix vs(nd, 3), cs(nd, 3);
   for (int q = 0; q < r.nq; ++q)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      r.ip[q * 3 + 0] = ip.x;
      r.ip[q * 3 + 1] = ip.y;
      r.ip[q * 3 + 2] = ip.z;
      r.w[q] = ip.weight;
      fe.CalcVShape(ip, vs);
      fe.CalcCurlShape(ip, cs);
      for (int k = 0; k < nd; ++k)
         for (int d = 0; d < 3; ++d)
         {
            r.vshape[((size_t)q * nd + k) * 3 + d] = vs(k, d);
            r.curl[((size_t)q * nd + k) * 3 + d] = cs(k, d);
         }
   }
}

// ---- extracted per-mesh assembly data -----------------------------------

struct AsmData
{
   int ne = 0;              // elements
   int nd = 0;              // dofs per element (ND tet: p(p+2)(p+3)/2)
   int p = 0;               // ND order
   int height = 0;          // global dof count (== fes.GetVSize, tdof==dof)
   bool has_kpar = false;
   // per-element, element-major flat arrays (GPU upload layout)
   std::vector<int> dof;     // ne*nd decoded dof index
   std::vector<double> sign; // ne*nd  +/-1 (MFEM negative-dof encoding)
   std::vector<int> fo;      // ne*4 face orientations 0..5 (empty for p=1)
   std::vector<int> attr;    // ne 1-based mesh attribute -> prob.regions[a-1]
   std::vector<double> J;    // ne*9 row-major Jacobian (constant, affine)
   std::vector<double> Jinv; // ne*9 row-major inverse Jacobian
   std::vector<double> detJ; // ne
   std::vector<double> v0;   // ne*3 element origin x(ref (0,0,0))
   // reference tables: CurlCurl rule (2p-2) and Mass/MixedVector rule (2p)
   AsmRefRule rule_cc, rule_mass;
   // CSR symbolic structure (sorted column indices)
   std::vector<int> I, Jcol;
};

void extract_asm_data(Mesh &mesh, FiniteElementSpace &fes, AsmData &ad)
{
   ad.ne = mesh.GetNE();
   ad.p = prob.order;
   ad.height = fes.GetVSize();
   ad.has_kpar = prob.kx != 0.0 || prob.ky != 0.0;
   const FiniteElement &fe = *fes.GetFE(0);
   ad.nd = fe.GetDof();

   // reference tables: exactly the default rules the production integrators
   // pick for affine ND tets (FunctionSpace::Pk, OrderW()==0):
   //   CurlCurlIntegrator      -> order 2p-2
   //   VectorFEMassIntegrator  -> order 2p
   //   MixedVector(Weak)Curl   -> order p + p = 2p (same rule object)
   extract_ref_rule(fe, IntRules.Get(Geometry::TETRAHEDRON,
                                     std::max(0, 2 * ad.p - 2)), ad.rule_cc);
   extract_ref_rule(fe, IntRules.Get(Geometry::TETRAHEDRON, 2 * ad.p),
                    ad.rule_mass);

   ad.dof.resize((size_t)ad.ne * ad.nd);
   ad.sign.resize((size_t)ad.ne * ad.nd);
   ad.fo.assign(ad.p >= 2 ? (size_t)ad.ne * 4 : 0, 0);
   ad.attr.resize(ad.ne);
   ad.J.resize((size_t)ad.ne * 9);
   ad.Jinv.resize((size_t)ad.ne * 9);
   ad.detJ.resize(ad.ne);
   ad.v0.resize((size_t)ad.ne * 3);

   Array<int> dofs;
   DofTransformation doftrans;
   IntegrationPoint origin;
   origin.Set3(0.0, 0.0, 0.0);
   for (int e = 0; e < ad.ne; ++e)
   {
      fes.GetElementDofs(e, dofs, doftrans);
      MFEM_VERIFY(dofs.Size() == ad.nd, "unexpected element dof count");
      for (int k = 0; k < ad.nd; ++k)
      {
         const int d = dofs[k];
         ad.dof[(size_t)e * ad.nd + k] = d >= 0 ? d : -1 - d;
         ad.sign[(size_t)e * ad.nd + k] = d >= 0 ? 1.0 : -1.0;
      }
      if (ad.p >= 2)
      {
         const Array<int> &Fo = doftrans.GetFaceOrientations();
         MFEM_VERIFY(Fo.Size() == 4, "expected 4 face orientations");
         for (int f = 0; f < 4; ++f) { ad.fo[(size_t)e * 4 + f] = Fo[f]; }
      }
      ad.attr[e] = mesh.GetAttribute(e);

      // affine geometry: constant Jacobian; periodic meshes carry L2 nodes,
      // so everything comes from the element transformation (vertex
      // coordinates may be wrapped copies for boundary-crossing elements)
      ElementTransformation *tr = mesh.GetElementTransformation(e);
      tr->SetIntPoint(&origin);
      const DenseMatrix &Jm = tr->Jacobian();
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
         {
            ad.J[(size_t)e * 9 + i * 3 + j] = Jm(i, j);
         }
      ad.detJ[e] = tr->Weight();
      MFEM_VERIFY(ad.detJ[e] > 0.0, "expected positively oriented tet");
      DenseMatrix Jinv(3, 3);
      CalcInverse(Jm, Jinv);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
         {
            ad.Jinv[(size_t)e * 9 + i * 3 + j] = Jinv(i, j);
         }
      Vector x0(3);
      tr->Transform(origin, x0);
      for (int d = 0; d < 3; ++d) { ad.v0[(size_t)e * 3 + d] = x0(d); }
   }
}

// CSR symbolic structure: per-row unique sorted columns from the element
// dof graph (union over elements of dof x dof). Built on host once per
// mesh; the value scatter (CPU reference below, GPU in M3) targets it.
void build_csr_symbolic(AsmData &ad)
{
   const int n = ad.height;
   std::vector<std::vector<int>> cols(n);
   {
      // rough per-row reserve: interior dof of p=3 sees ~nd*avg_valence
      for (auto &c : cols) { c.reserve(16); }
   }
   for (int e = 0; e < ad.ne; ++e)
   {
      const int *ed = ad.dof.data() + (size_t)e * ad.nd;
      for (int a = 0; a < ad.nd; ++a)
      {
         std::vector<int> &row = cols[ed[a]];
         for (int b = 0; b < ad.nd; ++b) { row.push_back(ed[b]); }
      }
   }
   ad.I.assign(n + 1, 0);
   size_t nnz = 0;
   for (int r = 0; r < n; ++r)
   {
      std::vector<int> &row = cols[r];
      std::sort(row.begin(), row.end());
      row.erase(std::unique(row.begin(), row.end()), row.end());
      nnz += row.size();
      ad.I[r + 1] = (int)nnz;
   }
   ad.Jcol.resize(nnz);
   for (int r = 0; r < n; ++r)
   {
      std::copy(cols[r].begin(), cols[r].end(), ad.Jcol.begin() + ad.I[r]);
   }
}

// ---- CPU reference reassembler ------------------------------------------
// Computes the complex local matrix of one element with the same data and
// operation order the GPU kernel will use, applies the dual DofTransformation
// (TInv(Fo)^T on face-dof pairs, columns then rows, as MFEM does), and
// scatters with dof signs into the symbolic CSR.

void reassemble_element(const AsmData &ad, int e, std::vector<C> &Ae)
{
   const int nd = ad.nd;
   const double *J = ad.J.data() + (size_t)e * 9;
   const double *Ji = ad.Jinv.data() + (size_t)e * 9;
   const double detJ = ad.detJ[e];
   const double *v0 = ad.v0.data() + (size_t)e * 3;
   const Region &reg = prob.regions[ad.attr[e] - 1];
   const double k2 = prob.k0 * prob.k0;

   Ae.assign((size_t)nd * nd, C(0.0, 0.0));
   std::vector<double> pc((size_t)nd * 3), pv((size_t)nd * 3);

   // curl-curl term on the 2p-2 rule: sum_i c_k[i] * lam_inv[i] * c_l[i]
   const AsmRefRule &rc = ad.rule_cc;
   for (int q = 0; q < rc.nq; ++q)
   {
      const double *xi = rc.ip.data() + (size_t)q * 3;
      const double z = v0[2] + J[6] * xi[0] + J[7] * xi[1] + J[8] * xi[2];
      C li[3];
      lam_inv_diag(z, li);
      const double w = rc.w[q] * detJ;
      // physical curl: c_k[i] = sum_j refcurl[k][j] * J[i][j] / detJ
      const double *cs = rc.curl.data() + (size_t)q * nd * 3;
      for (int k = 0; k < nd; ++k)
         for (int i = 0; i < 3; ++i)
         {
            pc[(size_t)k * 3 + i] =
               (cs[k * 3 + 0] * J[i * 3 + 0] + cs[k * 3 + 1] * J[i * 3 + 1] +
                cs[k * 3 + 2] * J[i * 3 + 2]) / detJ;
         }
      for (int k = 0; k < nd; ++k)
         for (int l = 0; l < nd; ++l)
         {
            C s(0.0, 0.0);
            for (int i = 0; i < 3; ++i)
            {
               s += pc[(size_t)k * 3 + i] * li[i] * pc[(size_t)l * 3 + i];
            }
            Ae[(size_t)k * nd + l] += w * s;
         }
   }

   // mass + (kpar) cross/klk terms share the 2p rule
   const AsmRefRule &rm = ad.rule_mass;
   double K[9];
   K[0] = 0.0;      K[1] = 0.0;      K[2] = prob.ky;
   K[3] = 0.0;      K[4] = 0.0;      K[5] = -prob.kx;
   K[6] = -prob.ky; K[7] = prob.kx;  K[8] = 0.0;
   for (int q = 0; q < rm.nq; ++q)
   {
      const double *xi = rm.ip.data() + (size_t)q * 3;
      const double z = v0[2] + J[6] * xi[0] + J[7] * xi[1] + J[8] * xi[2];
      const C s = s_z(z);
      const C lam[3] = {s, s, 1.0 / s};
      C li[3];
      lam_inv_diag(z, li);
      const double w = rm.w[q] * detJ;
      const double *vs = rm.vshape.data() + (size_t)q * nd * 3;
      const double *cs = rm.curl.data() + (size_t)q * nd * 3;
      // physical vshape: v_k[i] = sum_j refv[k][j] * Jinv[j][i]
      for (int k = 0; k < nd; ++k)
         for (int i = 0; i < 3; ++i)
         {
            pv[(size_t)k * 3 + i] =
               vs[k * 3 + 0] * Ji[0 * 3 + i] + vs[k * 3 + 1] * Ji[1 * 3 + i] +
               vs[k * 3 + 2] * Ji[2 * 3 + i];
            if (ad.has_kpar)
            {
               pc[(size_t)k * 3 + i] =
                  (cs[k * 3 + 0] * J[i * 3 + 0] + cs[k * 3 + 1] * J[i * 3 + 1] +
                   cs[k * 3 + 2] * J[i * 3 + 2]) / detJ;
            }
         }
      const C mdiag[3] = {-k2 * reg.eps * lam[0], -k2 * reg.eps * lam[1],
                          -k2 * reg.eps * lam[2]};
      for (int k = 0; k < nd; ++k)
         for (int l = 0; l < nd; ++l)
         {
            C s2(0.0, 0.0);
            for (int i = 0; i < 3; ++i)
            {
               s2 += pv[(size_t)k * 3 + i] * mdiag[i] * pv[(size_t)l * 3 + i];
            }
            Ae[(size_t)k * nd + l] += w * s2;
         }
      if (ad.has_kpar)
      {
         const C iu(0.0, 1.0);
         for (int k = 0; k < nd; ++k)
            for (int l = 0; l < nd; ++l)
            {
               // MixedVectorCurl (M0 = i K diag(li)): v_k . M0 . curl_l
               // MixedVectorWeakCurl (M1 = i diag(li) K): curl_k . M1 . v_l
               // KLK mass (M2 = -K diag(li) K): v_k . M2 . v_l
               C s0(0.0, 0.0), s1(0.0, 0.0), s2b(0.0, 0.0);
               for (int r = 0; r < 3; ++r)
                  for (int c = 0; c < 3; ++c)
                  {
                     const double krc = K[r * 3 + c];
                     if (krc != 0.0)
                     {
                        s0 += pv[(size_t)k * 3 + r] * (iu * krc * li[c]) *
                              pc[(size_t)l * 3 + c];
                        s1 += pc[(size_t)k * 3 + r] * (iu * li[r] * krc) *
                              pv[(size_t)l * 3 + c];
                     }
                     C m2(0.0, 0.0);
                     for (int t = 0; t < 3; ++t)
                     {
                        m2 += K[r * 3 + t] * li[t] * K[t * 3 + c];
                     }
                     s2b -= pv[(size_t)k * 3 + r] * m2 * pv[(size_t)l * 3 + c];
                  }
               Ae[(size_t)k * nd + l] += w * (s0 + s1 + s2b);
            }
      }
   }

   // dual DofTransformation: TInv(Fo)^T on face-dof pairs, cols then rows
   // (identical to mfem::DofTransformation::TransformDual on the summed
   // element matrix; p=1 has no face dofs)
   if (ad.p >= 2)
   {
      const int p = ad.p;
      const int foff = 6 * p;          // edge dofs first: 6 edges x p
      const int ntd = p * (p - 1);     // triangular-face dofs per face
      const int *fo = ad.fo.data() + (size_t)e * 4;
      for (int axis = 0; axis < 2; ++axis) // 0: columns, 1: rows
      {
         for (int f = 0; f < 4; ++f)
         {
            const double *Ti = asm_nd_TInv + 4 * fo[f];
            for (int i = 0; i < ntd / 2; ++i)
            {
               const int d0 = foff + f * ntd + 2 * i;
               for (int r = 0; r < nd; ++r)
               {
                  C &a = axis == 0 ? Ae[(size_t)r * nd + d0]
                                   : Ae[(size_t)d0 * nd + r];
                  C &b = axis == 0 ? Ae[(size_t)r * nd + d0 + 1]
                                   : Ae[(size_t)(d0 + 1) * nd + r];
                  // MultTranspose of the column-major 2x2 block
                  const C na = Ti[0] * a + Ti[1] * b;
                  const C nb = Ti[2] * a + Ti[3] * b;
                  a = na;
                  b = nb;
               }
            }
         }
      }
   }
}

// Full reassembly into the symbolic CSR; returns interleaved complex values
// aligned with (I, Jcol). Requires build_csr_symbolic() to have run.
void reassemble_all(const AsmData &ad, std::vector<double> &vre,
                    std::vector<double> &vim)
{
   const size_t nnz = ad.Jcol.size();
   vre.assign(nnz, 0.0);
   vim.assign(nnz, 0.0);
   std::vector<C> Ae;
   for (int e = 0; e < ad.ne; ++e)
   {
      reassemble_element(ad, e, Ae);
      const int *ed = ad.dof.data() + (size_t)e * ad.nd;
      const double *es = ad.sign.data() + (size_t)e * ad.nd;
      for (int a = 0; a < ad.nd; ++a)
      {
         const int row = ed[a];
         const int *jb = ad.Jcol.data() + ad.I[row];
         const int *je = ad.Jcol.data() + ad.I[row + 1];
         for (int b = 0; b < ad.nd; ++b)
         {
            const int *pos = std::lower_bound(jb, je, ed[b]);
            MFEM_VERIFY(pos != je && *pos == ed[b], "CSR position missing");
            const size_t k = (size_t)(pos - ad.Jcol.data());
            const C v = es[a] * es[b] * Ae[(size_t)a * ad.nd + b];
            vre[k] += v.real();
            vim[k] += v.imag();
         }
      }
   }
}
