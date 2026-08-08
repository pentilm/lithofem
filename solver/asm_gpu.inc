// v2.5 GPU assembly kernels (docs/gpu.md, milestones M2/M3).
// Included inside an anonymous namespace after asm_data.inc (needs AsmData,
// asm_nd_TInv, Problem/prob). The whole TU is compiled by nvcc (-x=cu).
//
// Portability contract (docs/gpu.md): standard CUDA C++ only — fp64
// arithmetic, plain shared memory, per-element blocks, double atomicAdd
// (sm_60+). No tensor cores, no TMA/clusters, no per-architecture branches.
//
// The kernels reproduce, in the same operation order, the CPU reference
// reassembler of asm_data.inc (which is pinned to MFEM by U7); U3/U4 pin
// the kernels directly to MFEM element matrices.

// ---- complex double ----------------------------------------------------

struct DC { double re, im; };

__host__ __device__ inline DC dc(double r, double i) { DC z; z.re = r; z.im = i; return z; }
__host__ __device__ inline DC dc_add(DC a, DC b) { return dc(a.re + b.re, a.im + b.im); }
__host__ __device__ inline DC dc_sub(DC a, DC b) { return dc(a.re - b.re, a.im - b.im); }
__host__ __device__ inline DC dc_mul(DC a, DC b)
{ return dc(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
__host__ __device__ inline DC dc_scale(double s, DC a) { return dc(s * a.re, s * a.im); }
__host__ __device__ inline DC dc_inv(DC a)
{
   const double d = a.re * a.re + a.im * a.im;
   return dc(a.re / d, -a.im / d);
}

// ---- device-side problem coefficients ----------------------------------

struct AsmPmlParams
{
   double z_min, z_max, thick, order;
   double smax_bot, smax_top;   // -log(target)*(order+1)/(2 k0 n_med thick)
   double k0, kx, ky;
};

// PML stretch s(z): same expression as the host s_z() with the n_med-
// dependent smax precomputed per side (identical double value).
__device__ inline DC dev_s_z(const AsmPmlParams &p, double z)
{
   double d = -1.0, smax = 0.0;
   if (z < p.z_min) { d = (p.z_min - z) / p.thick; smax = p.smax_bot; }
   else if (z > p.z_max) { d = (z - p.z_max) / p.thick; smax = p.smax_top; }
   if (d < 0.0) { return dc(1.0, 0.0); }
   if (d > 1.0) { d = 1.0; }
   return dc(1.0, smax * pow(d, p.order));
}

// integrator selection mask (per launch)
enum : int
{
   ASM_CURLCURL = 1,   // rule_cc
   ASM_MASS     = 2,   // rule_mass
   ASM_CROSS0   = 4,   // rule_mass, MixedVectorCurl  (i K Lam^-1)
   ASM_CROSS1   = 8,   // rule_mass, MixedVectorWeakCurl (i Lam^-1 K)
   ASM_KLK      = 16,  // rule_mass, -K Lam^-1 K mass
};

// Max local-matrix entries a thread accumulates: nd(p=4)=84 -> 84^2/256.
constexpr int ASM_BLOCK = 256;
constexpr int ASM_MAX_ACC = 32;

// ---- local matrix kernel -----------------------------------------------
// One block per element; per q-point the physical shapes are staged in
// shared memory cooperatively, then each thread accumulates its strided
// (k,l) entries in registers/local across all q; one global += at the end
// (two launches, one per integration rule, sum into Ae).

__global__ void asm_local_kernel(
   int nd, int nq,
   const double *__restrict__ ref_ip, const double *__restrict__ ref_w,
   const double *__restrict__ ref_v, const double *__restrict__ ref_c,
   const double *__restrict__ J9, const double *__restrict__ Ji9,
   const double *__restrict__ detJ, const double *__restrict__ v0,
   const int *__restrict__ attr, const DC *__restrict__ eps_tab,
   AsmPmlParams pml, int mask, DC *__restrict__ Ae)
{
   const int e = blockIdx.x;
   const int tid = threadIdx.x;
   extern __shared__ double sh[];
   double *pv = sh;            // nd*3 physical vshape at current q
   double *pc = sh + nd * 3;   // nd*3 physical curl at current q

   __shared__ double sJ[9], sJi[9], sdet, sv0z;
   __shared__ DC seps;
   if (tid == 0)
   {
      for (int i = 0; i < 9; ++i)
      {
         sJ[i] = J9[(size_t)e * 9 + i];
         sJi[i] = Ji9[(size_t)e * 9 + i];
      }
      sdet = detJ[e];
      sv0z = v0[(size_t)e * 3 + 2];
      seps = eps_tab[attr[e] - 1];
   }
   __syncthreads();

   const int n2 = nd * nd;
   const bool need_v = mask & (ASM_MASS | ASM_CROSS0 | ASM_CROSS1 | ASM_KLK);
   const bool need_c = mask & (ASM_CURLCURL | ASM_CROSS0 | ASM_CROSS1);
   const double k2 = pml.k0 * pml.k0;
   // K matrix (fill_K): only 4 nonzeros
   const double K02 = pml.ky, K12 = -pml.kx, K20 = -pml.ky, K21 = pml.kx;

   DC acc[ASM_MAX_ACC];
   const int nacc = (n2 - tid + ASM_BLOCK - 1) / ASM_BLOCK;
   for (int i = 0; i < nacc; ++i) { acc[i] = dc(0.0, 0.0); }

   for (int q = 0; q < nq; ++q)
   {
      // stage physical shapes for all basis functions
      for (int i = tid; i < nd * 3; i += ASM_BLOCK)
      {
         const int k = i / 3, d = i % 3;
         if (need_v)
         {
            pv[i] = ref_v[((size_t)q * nd + k) * 3 + 0] * sJi[0 * 3 + d] +
                    ref_v[((size_t)q * nd + k) * 3 + 1] * sJi[1 * 3 + d] +
                    ref_v[((size_t)q * nd + k) * 3 + 2] * sJi[2 * 3 + d];
         }
         if (need_c)
         {
            pc[i] = (ref_c[((size_t)q * nd + k) * 3 + 0] * sJ[d * 3 + 0] +
                     ref_c[((size_t)q * nd + k) * 3 + 1] * sJ[d * 3 + 1] +
                     ref_c[((size_t)q * nd + k) * 3 + 2] * sJ[d * 3 + 2]) / sdet;
         }
      }
      __syncthreads();

      // coefficients at this q (all threads compute the same scalars)
      const double z = sv0z + sJ[6] * ref_ip[q * 3 + 0] +
                       sJ[7] * ref_ip[q * 3 + 1] + sJ[8] * ref_ip[q * 3 + 2];
      const DC s = dev_s_z(pml, z);
      const DC si = dc_inv(s);
      const DC li[3] = {si, si, s};                    // Lam^-1 diagonal
      const double w = ref_w[q] * sdet;
      DC mdiag[3];
      if (mask & ASM_MASS)
      {
         const DC me = dc_scale(-k2, seps);
         mdiag[0] = dc_mul(me, s);
         mdiag[1] = mdiag[0];
         mdiag[2] = dc_mul(me, si);
      }

      for (int idx = tid, a = 0; idx < n2; idx += ASM_BLOCK, ++a)
      {
         const int k = idx / nd, l = idx % nd;
         const double *vk = pv + k * 3, *vl = pv + l * 3;
         const double *ck = pc + k * 3, *cl = pc + l * 3;
         DC sum = dc(0.0, 0.0);
         if (mask & ASM_CURLCURL)
         {
            for (int i = 0; i < 3; ++i)
            {
               sum = dc_add(sum, dc_scale(ck[i] * cl[i], li[i]));
            }
         }
         if (mask & ASM_MASS)
         {
            for (int i = 0; i < 3; ++i)
            {
               sum = dc_add(sum, dc_scale(vk[i] * vl[i], mdiag[i]));
            }
         }
         if (mask & ASM_CROSS0)   // v_k . (i K Lam^-1) . curl_l
         {
            DC t = dc_add(dc_add(
                      dc_scale(vk[0] * K02 * cl[2], li[2]),
                      dc_scale(vk[1] * K12 * cl[2], li[2])),
                   dc_add(
                      dc_scale(vk[2] * K20 * cl[0], li[0]),
                      dc_scale(vk[2] * K21 * cl[1], li[1])));
            sum = dc_add(sum, dc_mul(dc(0.0, 1.0), t));
         }
         if (mask & ASM_CROSS1)   // curl_k . (i Lam^-1 K) . v_l
         {
            DC t = dc_add(dc_add(
                      dc_scale(ck[0] * K02 * vl[2], li[0]),
                      dc_scale(ck[1] * K12 * vl[2], li[1])),
                   dc_add(
                      dc_scale(ck[2] * K20 * vl[0], li[2]),
                      dc_scale(ck[2] * K21 * vl[1], li[2])));
            sum = dc_add(sum, dc_mul(dc(0.0, 1.0), t));
         }
         if (mask & ASM_KLK)      // v_k . (-K Lam^-1 K) . v_l
         {
            // M2 = -K diag(li) K; K columns/rows only touch (0,1)<->2
            // M2(0,0) = -K02 li2 K20; M2(0,1) = -K02 li2 K21
            // M2(1,0) = -K12 li2 K20; M2(1,1) = -K12 li2 K21
            // M2(2,2) = -(K20 li0 K02 + K21 li1 K12)
            DC t = dc_add(dc_add(
                      dc_scale(-vk[0] * K02 * K20 * vl[0], li[2]),
                      dc_scale(-vk[0] * K02 * K21 * vl[1], li[2])),
                   dc_add(
                      dc_scale(-vk[1] * K12 * K20 * vl[0], li[2]),
                      dc_scale(-vk[1] * K12 * K21 * vl[1], li[2])));
            t = dc_add(t, dc_scale(-vk[2] * (K20 * K02) * vl[2], li[0]));
            t = dc_add(t, dc_scale(-vk[2] * (K21 * K12) * vl[2], li[1]));
            sum = dc_add(sum, t);
         }
         acc[a] = dc_add(acc[a], dc_scale(w, sum));
      }
      __syncthreads();
   }

   for (int idx = tid, a = 0; idx < n2; idx += ASM_BLOCK, ++a)
   {
      DC *dst = Ae + (size_t)e * n2 + idx;
      *dst = dc_add(*dst, acc[a]);
   }
}

// ---- dual DofTransformation kernel (p >= 2) ----------------------------
// TInv(Fo)^T on face-dof pairs, columns then rows (asm_data.inc layout).

__device__ __constant__ double d_asm_nd_TInv[24];

__global__ void asm_te_kernel(int nd, int p, const int *__restrict__ fo,
                              DC *__restrict__ Ae)
{
   const int e = blockIdx.x;
   DC *A = Ae + (size_t)e * nd * nd;
   const int foff = 6 * p;
   const int ntd = p * (p - 1);
   for (int axis = 0; axis < 2; ++axis)
   {
      for (int f = 0; f < 4; ++f)
      {
         const double *Ti = d_asm_nd_TInv + 4 * fo[(size_t)e * 4 + f];
         for (int i = 0; i < ntd / 2; ++i)
         {
            const int d0 = foff + f * ntd + 2 * i;
            for (int r = threadIdx.x; r < nd; r += blockDim.x)
            {
               DC &a = axis == 0 ? A[(size_t)r * nd + d0]
                                 : A[(size_t)d0 * nd + r];
               DC &b = axis == 0 ? A[(size_t)r * nd + d0 + 1]
                                 : A[(size_t)(d0 + 1) * nd + r];
               const DC na = dc_add(dc_scale(Ti[0], a), dc_scale(Ti[1], b));
               const DC nb = dc_add(dc_scale(Ti[2], a), dc_scale(Ti[3], b));
               a = na;
               b = nb;
            }
         }
      }
      __syncthreads();
   }
}

// ---- host wrapper -------------------------------------------------------

#define LFEM_ASM_CUDA_OK(call)                                               \
   do {                                                                      \
      cudaError_t e_ = (call);                                               \
      if (e_ != cudaSuccess) {                                               \
         std::printf("asm_gpu: CUDA error '%s' at %s:%d\n",                  \
                     cudaGetErrorString(e_), __FILE__, __LINE__);            \
         return false;                                                       \
      }                                                                      \
   } while (0)

struct AsmGpuRule
{
   int nq = 0;
   double *ip = nullptr, *w = nullptr, *v = nullptr, *c = nullptr;
};

class AsmGpu
{
public:
   ~AsmGpu() { Release(); }

   bool Init(const AsmData &ad, int gpu_id)
   {
      ne_ = ad.ne;
      nd_ = ad.nd;
      p_ = ad.p;
      MFEM_VERIFY(nd_ * nd_ <= ASM_BLOCK * ASM_MAX_ACC,
                  "element too large for the accumulator layout");
      LFEM_ASM_CUDA_OK(cudaSetDevice(gpu_id));
      LFEM_ASM_CUDA_OK(cudaMemcpyToSymbol(d_asm_nd_TInv, asm_nd_TInv,
                                          sizeof(asm_nd_TInv)));
      // geometry / topology
      LFEM_ASM_CUDA_OK(upload(d_J_, ad.J));
      LFEM_ASM_CUDA_OK(upload(d_Ji_, ad.Jinv));
      LFEM_ASM_CUDA_OK(upload(d_det_, ad.detJ));
      LFEM_ASM_CUDA_OK(upload(d_v0_, ad.v0));
      LFEM_ASM_CUDA_OK(upload(d_attr_, ad.attr));
      if (p_ >= 2) { LFEM_ASM_CUDA_OK(upload(d_fo_, ad.fo)); }
      LFEM_ASM_CUDA_OK(upload_rule(rule_cc_, ad.rule_cc, nd_));
      LFEM_ASM_CUDA_OK(upload_rule(rule_mass_, ad.rule_mass, nd_));
      // region coefficient table
      std::vector<DC> eps(prob.regions.size());
      for (size_t r = 0; r < prob.regions.size(); ++r)
      {
         eps[r] = dc(prob.regions[r].eps.real(), prob.regions[r].eps.imag());
      }
      LFEM_ASM_CUDA_OK(upload(d_eps_, eps));
      // PML params with per-side smax precomputed exactly as host s_z does
      pml_.z_min = prob.z_min;
      pml_.z_max = prob.z_max;
      pml_.thick = prob.pml_thick;
      pml_.order = prob.pml_order;
      pml_.k0 = prob.k0;
      pml_.kx = prob.kx;
      pml_.ky = prob.ky;
      double n_bot = 1.0, n_top = 1.0;
      for (const Region &r : prob.regions)
      {
         if (r.is_pml && !r.pml_top) { n_bot = std::abs(std::sqrt(r.eps)); }
         if (r.is_pml && r.pml_top) { n_top = std::abs(std::sqrt(r.eps)); }
      }
      pml_.smax_bot = -std::log(prob.pml_target) * (prob.pml_order + 1) /
                      (2.0 * prob.k0 * n_bot * prob.pml_thick);
      pml_.smax_top = -std::log(prob.pml_target) * (prob.pml_order + 1) /
                      (2.0 * prob.k0 * n_top * prob.pml_thick);
      LFEM_ASM_CUDA_OK(cudaMalloc(&d_Ae_, (size_t)ne_ * nd_ * nd_ * sizeof(DC)));
      return true;
   }

   // compute local matrices: zero Ae, launch the cc rule then the mass-rule
   // integrators (masks let U3 test each integrator in isolation)
   bool ComputeLocal(int mask_cc, int mask_mass, bool apply_te)
   {
      LFEM_ASM_CUDA_OK(cudaMemset(d_Ae_, 0,
                                  (size_t)ne_ * nd_ * nd_ * sizeof(DC)));
      const size_t shmem = (size_t)nd_ * 3 * 2 * sizeof(double);
      if (mask_cc)
      {
         asm_local_kernel<<<ne_, ASM_BLOCK, shmem>>>(
            nd_, rule_cc_.nq, rule_cc_.ip, rule_cc_.w, rule_cc_.v, rule_cc_.c,
            d_J_, d_Ji_, d_det_, d_v0_, d_attr_, d_eps_, pml_, mask_cc, d_Ae_);
      }
      if (mask_mass)
      {
         asm_local_kernel<<<ne_, ASM_BLOCK, shmem>>>(
            nd_, rule_mass_.nq, rule_mass_.ip, rule_mass_.w, rule_mass_.v,
            rule_mass_.c, d_J_, d_Ji_, d_det_, d_v0_, d_attr_, d_eps_, pml_,
            mask_mass, d_Ae_);
      }
      if (apply_te && p_ >= 2)
      {
         asm_te_kernel<<<ne_, 128>>>(nd_, p_, d_fo_, d_Ae_);
      }
      LFEM_ASM_CUDA_OK(cudaGetLastError());
      LFEM_ASM_CUDA_OK(cudaDeviceSynchronize());
      return true;
   }

   // test-only download (never called on the production path; the marker
   // line lets the e2e tests assert exactly that)
   bool DownloadLocal(std::vector<DC> &out) const
   {
      std::printf("asm_gpu_test_download (test-only D2H)\n");
      out.resize((size_t)ne_ * nd_ * nd_);
      LFEM_ASM_CUDA_OK(cudaMemcpy(out.data(), d_Ae_, out.size() * sizeof(DC),
                                  cudaMemcpyDeviceToHost));
      return true;
   }

   DC *DeviceLocal() const { return d_Ae_; }
   int ne() const { return ne_; }
   int nd() const { return nd_; }

   void Release()
   {
      auto fre = [](auto *&p)
      { if (p) { cudaFree(p); p = nullptr; } };
      fre(d_J_); fre(d_Ji_); fre(d_det_); fre(d_v0_);
      fre(d_attr_); fre(d_fo_); fre(d_eps_); fre(d_Ae_);
      release_rule(rule_cc_);
      release_rule(rule_mass_);
   }

protected:
   template <typename T>
   static cudaError_t upload(T *&dst, const std::vector<T> &src)
   {
      cudaError_t e = cudaMalloc(&dst, src.size() * sizeof(T));
      if (e != cudaSuccess) { return e; }
      return cudaMemcpy(dst, src.data(), src.size() * sizeof(T),
                        cudaMemcpyHostToDevice);
   }
   static cudaError_t upload(DC *&dst, const std::vector<DC> &src)
   {
      cudaError_t e = cudaMalloc(&dst, src.size() * sizeof(DC));
      if (e != cudaSuccess) { return e; }
      return cudaMemcpy(dst, src.data(), src.size() * sizeof(DC),
                        cudaMemcpyHostToDevice);
   }
   static cudaError_t upload_rule(AsmGpuRule &r, const AsmRefRule &src, int)
   {
      r.nq = src.nq;
      cudaError_t e;
      auto up = [&](double *&d, const std::vector<double> &h) -> cudaError_t
      {
         cudaError_t ee = cudaMalloc(&d, h.size() * sizeof(double));
         if (ee != cudaSuccess) { return ee; }
         return cudaMemcpy(d, h.data(), h.size() * sizeof(double),
                           cudaMemcpyHostToDevice);
      };
      if ((e = up(r.ip, src.ip)) != cudaSuccess) { return e; }
      if ((e = up(r.w, src.w)) != cudaSuccess) { return e; }
      if ((e = up(r.v, src.vshape)) != cudaSuccess) { return e; }
      return up(r.c, src.curl);
   }
   static void release_rule(AsmGpuRule &r)
   {
      auto fre = [](double *&p)
      { if (p) { cudaFree(p); p = nullptr; } };
      fre(r.ip); fre(r.w); fre(r.v); fre(r.c);
   }

   int ne_ = 0, nd_ = 0, p_ = 0;
   double *d_J_ = nullptr, *d_Ji_ = nullptr, *d_det_ = nullptr,
          *d_v0_ = nullptr;
   int *d_attr_ = nullptr, *d_fo_ = nullptr;
   DC *d_eps_ = nullptr;
   DC *d_Ae_ = nullptr;
   AsmGpuRule rule_cc_, rule_mass_;
   AsmPmlParams pml_{};
};

// ---- M3: global CSR scatter, essential elimination, residual ------------

// Scatter with dof signs into the symbolic CSR: one block per element, each
// (a,b) entry finds its nnz slot by binary search in the sorted row and
// atomicAdd-accumulates (double atomics: standard CUDA, sm_60+). design note:
// the direct-search scatter is ~1% of the assembly budget, so neither a
// precomputed position map nor a cub sort-reduce pipeline is warranted.
__global__ void asm_scatter_kernel(
   int nd, const int *__restrict__ dof, const double *__restrict__ sign,
   const int *__restrict__ I, const int *__restrict__ Jc,
   const DC *__restrict__ Ae, DC *__restrict__ vals)
{
   const int e = blockIdx.x;
   const int *ed = dof + (size_t)e * nd;
   const double *es = sign + (size_t)e * nd;
   const int n2 = nd * nd;
   for (int idx = threadIdx.x; idx < n2; idx += blockDim.x)
   {
      const int a = idx / nd, b = idx % nd;
      const int r = ed[a], c = ed[b];
      int lo = I[r], hi = I[r + 1] - 1;
      while (lo < hi)
      {
         const int mid = (lo + hi) >> 1;
         if (Jc[mid] < c) { lo = mid + 1; }
         else { hi = mid; }
      }
      const double s = es[a] * es[b];
      const DC v = Ae[(size_t)e * n2 + idx];
      atomicAdd(&vals[lo].re, s * v.re);
      atomicAdd(&vals[lo].im, s * v.im);
   }
}

// Essential elimination, one thread per row: zero ess rows and columns,
// unit diagonal (A_r DIAG_ONE + A_i DIAG_ZERO of the CPU path => 1 + 0i).
__global__ void asm_eliminate_kernel(
   int n, const int *__restrict__ I, const int *__restrict__ Jc,
   const unsigned char *__restrict__ is_ess, DC *__restrict__ vals)
{
   const int r = blockIdx.x * blockDim.x + threadIdx.x;
   if (r >= n) { return; }
   const bool row_ess = is_ess[r];
   for (int k = I[r]; k < I[r + 1]; ++k)
   {
      const int c = Jc[k];
      if (row_ess || is_ess[c])
      {
         vals[k] = (row_ess && c == r) ? dc(1.0, 0.0) : dc(0.0, 0.0);
      }
   }
}

// complex CSR SpMV: y = A x (self-residual check without a host matrix)
__global__ void asm_spmv_kernel(
   int n, const int *__restrict__ I, const int *__restrict__ Jc,
   const DC *__restrict__ vals, const DC *__restrict__ x, DC *__restrict__ y)
{
   const int r = blockIdx.x * blockDim.x + threadIdx.x;
   if (r >= n) { return; }
   DC acc = dc(0.0, 0.0);
   for (int k = I[r]; k < I[r + 1]; ++k)
   {
      acc = dc_add(acc, dc_mul(vals[k], x[Jc[k]]));
   }
   y[r] = acc;
}

// Global-assembly extension: owns the device CSR the cuDSS solver consumes
// zero-copy. The matrix values NEVER travel to the host on the production
// path (DownloadValues below exists for the U5/U6 tests only).
class AsmGpuGlobal : public AsmGpu
{
public:
   bool BuildGlobal(const AsmData &ad, const Array<int> &ess_list)
   {
      n_ = ad.height;
      nnz_ = (int)ad.Jcol.size();
      LFEM_ASM_CUDA_OK(upload(d_I_, ad.I));
      LFEM_ASM_CUDA_OK(upload(d_Jc_, ad.Jcol));
      LFEM_ASM_CUDA_OK(upload(d_dof_, ad.dof));
      LFEM_ASM_CUDA_OK(upload(d_sign_, ad.sign));
      std::vector<unsigned char> ess(n_, 0);
      for (int i = 0; i < ess_list.Size(); ++i) { ess[ess_list[i]] = 1; }
      LFEM_ASM_CUDA_OK(upload(d_ess_, ess));
      LFEM_ASM_CUDA_OK(cudaMalloc(&d_vals_, (size_t)nnz_ * sizeof(DC)));
      return true;
   }

   // local matrices (already computed+transformed in d_Ae_) -> CSR values
   bool ScatterEliminate(bool eliminate)
   {
      LFEM_ASM_CUDA_OK(cudaMemset(d_vals_, 0, (size_t)nnz_ * sizeof(DC)));
      asm_scatter_kernel<<<ne_, ASM_BLOCK>>>(nd_, d_dof_, d_sign_, d_I_,
                                             d_Jc_, d_Ae_, d_vals_);
      if (eliminate)
      {
         const int nb = (n_ + 255) / 256;
         asm_eliminate_kernel<<<nb, 256>>>(n_, d_I_, d_Jc_, d_ess_, d_vals_);
      }
      LFEM_ASM_CUDA_OK(cudaGetLastError());
      LFEM_ASM_CUDA_OK(cudaDeviceSynchronize());
      return true;
   }

   // ||A X - B|| / ||B|| with X, B as production 2n vectors [re; im]
   bool Residual(const Vector &X, const Vector &B, double &rel) const
   {
      std::vector<DC> hx(n_), hb(n_);
      const double *xp = X.GetData(), *bp = B.GetData();
      for (int i = 0; i < n_; ++i)
      {
         hx[i] = dc(xp[i], xp[n_ + i]);
         hb[i] = dc(bp[i], bp[n_ + i]);
      }
      DC *d_x = nullptr, *d_y = nullptr;
      LFEM_ASM_CUDA_OK(cudaMalloc(&d_x, (size_t)n_ * sizeof(DC)));
      LFEM_ASM_CUDA_OK(cudaMalloc(&d_y, (size_t)n_ * sizeof(DC)));
      cudaMemcpy(d_x, hx.data(), (size_t)n_ * sizeof(DC),
                 cudaMemcpyHostToDevice);
      const int nb = (n_ + 255) / 256;
      asm_spmv_kernel<<<nb, 256>>>(n_, d_I_, d_Jc_, d_vals_, d_x, d_y);
      cudaError_t err = cudaDeviceSynchronize();
      std::vector<DC> hy(n_);
      cudaMemcpy(hy.data(), d_y, (size_t)n_ * sizeof(DC),
                 cudaMemcpyDeviceToHost);
      cudaFree(d_x);
      cudaFree(d_y);
      if (err != cudaSuccess) { return false; }
      double rn = 0.0, bn = 0.0;
      for (int i = 0; i < n_; ++i)
      {
         const double dr = hy[i].re - hb[i].re, di = hy[i].im - hb[i].im;
         rn += dr * dr + di * di;
         bn += hb[i].re * hb[i].re + hb[i].im * hb[i].im;
      }
      rel = bn > 0.0 ? std::sqrt(rn / bn) : 0.0;
      return true;
   }

   // test-only (U5/U6); never called on the production path (marker line
   // asserted absent from production stdout by the U8 tests)
   bool DownloadValues(std::vector<DC> &out) const
   {
      std::printf("asm_gpu_test_download (test-only D2H)\n");
      out.resize(nnz_);
      LFEM_ASM_CUDA_OK(cudaMemcpy(out.data(), d_vals_,
                                  (size_t)nnz_ * sizeof(DC),
                                  cudaMemcpyDeviceToHost));
      return true;
   }

   int n() const { return n_; }
   int nnz() const { return nnz_; }
   int *DeviceI() const { return d_I_; }
   int *DeviceJ() const { return d_Jc_; }
   void *DeviceValues() const { return (void *)d_vals_; }

   ~AsmGpuGlobal()
   {
      auto fre = [](auto *&p)
      { if (p) { cudaFree(p); p = nullptr; } };
      fre(d_I_); fre(d_Jc_); fre(d_dof_); fre(d_sign_); fre(d_ess_);
      fre(d_vals_);
   }

private:
   int n_ = 0, nnz_ = 0;
   int *d_I_ = nullptr, *d_Jc_ = nullptr, *d_dof_ = nullptr;
   double *d_sign_ = nullptr;
   unsigned char *d_ess_ = nullptr;
   DC *d_vals_ = nullptr;
};
