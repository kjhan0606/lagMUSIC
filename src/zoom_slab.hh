// Phase G.0: zoom-region z-slab decomposition primitive.
//
// Each refinement cluster's [min_z, max_z] z-extent is split evenly across
// a per-cluster MPI sub-communicator. Each rank holds its z-interior plus
// halo cells at the low/high z faces for cross-rank stencil access.
//
// This file declares only the layout descriptor and the smoke-test entry.
// Wiring into refinement_hierarchy / GridHierarchy / mg_solver happens in
// Phases G.1+; this header is callable standalone.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#else
typedef int MPI_Comm;
#define MPI_COMM_NULL 0
#define MPI_COMM_WORLD 0
#endif

namespace MUSIC {
namespace zoom_slab {

// ---------------------------------------------------------------------------
// Layout descriptor: one ZoomSlabLayout per (cluster, level, sub_comm).
// All fields are replicated across sub_comm except my_* which are rank-local.
// ---------------------------------------------------------------------------
struct ZoomSlabLayout {
    // identity
    std::size_t cluster_id;
    unsigned    level;
    MPI_Comm    sub_comm;
    int         sub_rank;
    int         sub_size;

    // cluster geometry: fine-cell indices in the parent (level) grid.
    // The cluster occupies parent[cluster_oax .. cluster_oax+cluster_nx)
    // in each axis, in fine cells at this level.
    int cluster_oax, cluster_oay, cluster_oaz;
    int cluster_nx,  cluster_ny,  cluster_nz;

    // per-rank z-slab assignment (cluster-local indices in [0, cluster_nz)).
    int my_z0, my_z1;       // half-open interior range
    int halo_w;             // halo width in fine cells

    // gather/scatter metadata, replicated across sub_comm.
    std::vector<int> all_z0;  // size = sub_size
    std::vector<int> all_z1;  // size = sub_size
};

inline int local_nz(const ZoomSlabLayout& L) { return L.my_z1 - L.my_z0; }

// Storage size (cells) for the local-with-halo z-slab buffer.
// Layout is (i, j, k) with i in [0, cluster_nx), j in [0, cluster_ny),
// k in [0, local_nz + 2*halo_w) where k in [0, halo_w) is the low-z halo
// and k in [local_nz + halo_w, local_nz + 2*halo_w) is the high-z halo.
inline std::size_t local_with_halo_size(const ZoomSlabLayout& L) {
    return (std::size_t)L.cluster_nx * (std::size_t)L.cluster_ny *
           (std::size_t)(local_nz(L) + 2 * L.halo_w);
}

// Storage size (cells) for the interior-only local buffer.
inline std::size_t local_interior_size(const ZoomSlabLayout& L) {
    return (std::size_t)L.cluster_nx * (std::size_t)L.cluster_ny *
           (std::size_t)local_nz(L);
}

// Storage size (cells) for the full cluster (sub-comm-root buffer for scatter/gather).
inline std::size_t cluster_full_size(const ZoomSlabLayout& L) {
    return (std::size_t)L.cluster_nx * (std::size_t)L.cluster_ny *
           (std::size_t)L.cluster_nz;
}

// ---------------------------------------------------------------------------
// Construct a layout by splitting [0, cluster_nz) evenly across sub_comm.
// Even-split policy: each rank gets cluster_nz / sub_size, with remainder
// distributed one extra cell to the first (cluster_nz % sub_size) ranks.
// halo_w must be <= local_nz on every rank for halo_exchange_z correctness.
// ---------------------------------------------------------------------------
ZoomSlabLayout make_layout(
    MPI_Comm    sub_comm,
    std::size_t cluster_id,
    unsigned    level,
    int cluster_oax, int cluster_oay, int cluster_oaz,
    int cluster_nx,  int cluster_ny,  int cluster_nz,
    int halo_w);

// ---------------------------------------------------------------------------
// scatter: sub_comm rank 0 holds full_buf (cluster_nx * cluster_ny * cluster_nz,
// row-major (i, j, k) with k fastest). Distributes interior to each rank's
// my_interior buffer of size local_interior_size(L). Halo cells are NOT
// populated here; caller copies interior into the halo-padded layout and
// then calls halo_exchange_z.
// ---------------------------------------------------------------------------
void scatter_cluster_to_zslab_double(const ZoomSlabLayout& L,
                                     const double* full_buf,
                                     double*       my_interior);

void scatter_cluster_to_zslab_float (const ZoomSlabLayout& L,
                                     const float*  full_buf,
                                     float*        my_interior);

// Phase G.1: scatter from an arbitrary source rank in sub_comm (e.g., the
// per-box owner in MUSIC's owner=b%nproc round-robin). full_buf is meaningful
// only on rank == src_rank (within sub_comm). Default-source (src_rank=0)
// variants above are equivalent to passing src_rank=0 here.
void scatter_cluster_to_zslab_double_from(const ZoomSlabLayout& L, int src_rank,
                                          const double* full_buf,
                                          double*       my_interior);

void scatter_cluster_to_zslab_float_from (const ZoomSlabLayout& L, int src_rank,
                                          const float*  full_buf,
                                          float*        my_interior);

// ---------------------------------------------------------------------------
// gather: inverse of scatter. sub_comm rank 0 receives the assembled
// cluster_full_size(L) buffer.
// ---------------------------------------------------------------------------
void gather_zslab_to_cluster_double(const ZoomSlabLayout& L,
                                    const double* my_interior,
                                    double*       full_buf);

void gather_zslab_to_cluster_float (const ZoomSlabLayout& L,
                                    const float*  my_interior,
                                    float*        full_buf);

// Phase G.1: gather to an arbitrary destination rank in sub_comm.
void gather_zslab_to_cluster_double_to(const ZoomSlabLayout& L, int dst_rank,
                                       const double* my_interior,
                                       double*       full_buf);

void gather_zslab_to_cluster_float_to (const ZoomSlabLayout& L, int dst_rank,
                                       const float*  my_interior,
                                       float*        full_buf);

// ---------------------------------------------------------------------------
// halo exchange along z within sub_comm. Input is local_with_halo buffer
// of size local_with_halo_size(L); on entry, the interior at
// k in [halo_w, halo_w + local_nz) must already be filled. On exit, the
// halo cells at k in [0, halo_w) and [halo_w+local_nz, 2*halo_w+local_nz)
// hold neighbour-rank interior data.
//
// Z-edge policy: NON-PERIODIC. Sub_comm rank 0's low-z halo and the last
// rank's high-z halo are zero-filled (cluster boundary; caller fills them
// from a coarser level if needed).
// ---------------------------------------------------------------------------
void halo_exchange_z_double(const ZoomSlabLayout& L, double* local_with_halo);
void halo_exchange_z_float (const ZoomSlabLayout& L, float*  local_with_halo);

// ---------------------------------------------------------------------------
// Phase G.1 convenience overloads: let templated callers (e.g.
// GridHierarchy<T>) dispatch by T without if-constexpr (codebase is C++11).
// ---------------------------------------------------------------------------
inline void scatter_from(const ZoomSlabLayout& L, int src, const double* f, double* my)
{ scatter_cluster_to_zslab_double_from(L, src, f, my); }
inline void scatter_from(const ZoomSlabLayout& L, int src, const float*  f, float*  my)
{ scatter_cluster_to_zslab_float_from (L, src, f, my); }
inline void gather_to  (const ZoomSlabLayout& L, int dst, const double* my, double* f)
{ gather_zslab_to_cluster_double_to(L, dst, my, f); }
inline void gather_to  (const ZoomSlabLayout& L, int dst, const float*  my, float*  f)
{ gather_zslab_to_cluster_float_to (L, dst, my, f); }
inline void halo_exchange_z(const ZoomSlabLayout& L, double* b)
{ halo_exchange_z_double(L, b); }
inline void halo_exchange_z(const ZoomSlabLayout& L, float*  b)
{ halo_exchange_z_float (L, b); }

// ---------------------------------------------------------------------------
// Phase G.2: slab residual primitive.
// Compute r = h^2 * f - L u, using the 7-point O2 Laplacian
//   L u(i,j,k) = u(i-1,j,k) + u(i+1,j,k) + u(i,j-1,k) + u(i,j+1,k)
//                + u(i,j,k-1) + u(i,j,k+1) - 6 u(i,j,k)
// over the interior of the z-slab (i in [1, cnx-1), j in [1, cny-1),
// k_local in [halo_w, halo_w + local_nz)). The user must have called
// halo_exchange_z(L, u_buf) before this so the z-halo cells are populated.
// x/y boundary cells of the cluster (i==0, i==cnx-1, j==0, j==cny-1) are
// NOT updated — caller fills r at those cells from a separate source if
// needed. Returns the per-rank L2 norm of r (squared, NOT global-reduced).
//
// u_buf and r_buf use the local_with_halo_size(L) layout (i,j,k) row-major
// with k fastest.
double residual_z_slab_double(const ZoomSlabLayout& L,
                              const double* u_with_halo,
                              const double* f_with_halo,
                              double h,
                              double* r_with_halo);
float  residual_z_slab_float (const ZoomSlabLayout& L,
                              const float*  u_with_halo,
                              const float*  f_with_halo,
                              float  h,
                              float*  r_with_halo);

inline double residual_z(const ZoomSlabLayout& L,
                         const double* u, const double* f,
                         double h, double* r)
{ return residual_z_slab_double(L, u, f, h, r); }
inline float  residual_z(const ZoomSlabLayout& L,
                         const float* u, const float* f,
                         float h, float* r)
{ return residual_z_slab_float(L, u, f, h, r); }

// Phase G.2: smoke test for residual_z. Builds a single cluster with a
// known smooth u(i,j,k) (e.g., sin/cos product), computes f = L u / h^2
// analytically, then verifies residual_z returns ~0 to machine precision
// in the interior. Returns true on success.
bool smoke_test_residual_single_cluster(int halo_w = 2);

// ---------------------------------------------------------------------------
// Phase G.2b: red-black Gauss-Seidel smoother on the z-slab.
//
// One full GS sweep (color=0 then color=1) of the 7-point O2 update
//   u(i,j,k) = ( u(i-1,j,k) + u(i+1,j,k) + u(i,j-1,k) + u(i,j+1,k)
//              + u(i,j,k-1) + u(i,j,k+1) + h^2 f(i,j,k) ) / 6
// applied at cells (ix+iy+iz_global) % 2 == color, where iz_global is the
// cluster-global z-index (so the coloring matches across sub_comm ranks).
//
// Update domain: i in [1, cnx-1), j in [1, cny-1), kl in [halo_w, halo_w+local_nz).
// Cluster x/y boundary cells (i==0, i==cnx-1, j==0, j==cny-1) are NOT updated —
// caller fills them from a coarser level if needed (mirrors residual_z_slab).
// Cluster z-edge cells on sub_rank==0 / sub_rank==sub_size-1 are updated by the
// stencil using the (zero-filled) edge halo — the caller may either pre-fill
// those halo cells with parent-prolonged values before calling, or accept the
// zero-edge boundary condition.
//
// u_with_halo is updated in place. f_with_halo is read-only. Both use
// local_with_halo_size(L) layout. halo_exchange_z(L, u_with_halo) is performed
// internally between the two color sweeps AND at the end of the smoother, so
// after this returns u_with_halo's z-halo cells hold neighbours' fresh values.
//
// Returns 0.0 (reserved — future variants may return the per-rank L2 of u
// updates if needed for convergence monitoring; for now, residual_z handles
// monitoring).
//
// Caller must do halo_exchange_z(L, u_with_halo) BEFORE the first call in a
// smoothing sequence (so color=0 reads fresh halo values). Subsequent calls
// can skip that pre-exchange since gs_z's tail halo_exchange leaves the buffer
// halo-fresh.
void gs_z_slab_double(const ZoomSlabLayout& L,
                      double* u_with_halo,
                      const double* f_with_halo,
                      double h);
void gs_z_slab_float (const ZoomSlabLayout& L,
                      float*  u_with_halo,
                      const float*  f_with_halo,
                      float  h);

inline void gs_z(const ZoomSlabLayout& L,
                 double* u, const double* f, double h)
{ gs_z_slab_double(L, u, f, h); }
inline void gs_z(const ZoomSlabLayout& L,
                 float* u, const float* f, float h)
{ gs_z_slab_float(L, u, f, h); }

// Phase G.2b smoke test for gs_z. Builds a single cluster, applies enough
// GS sweeps to reduce the residual on a sin-product analytic source by a
// known factor (relative to the initial residual), and verifies bit-identical
// behavior across np=1/2/4/8 (sub_comm rank decomposition does not change the
// red-black coloring because color uses iz_global). Returns true on success.
bool smoke_test_gs_single_cluster(int halo_w = 1, int n_sweeps = 8);

// ---------------------------------------------------------------------------
// Phase G.2b (production-wire step 1): red-black GS smoother on z-slab using
// production's L u = -f sign convention.
//
//   u(i,j,k) = ( sum_neighbors + h^2 * f(i,j,k) ) / 6      <-- THIS variant
//   u(i,j,k) = ( sum_neighbors - h^2 * f(i,j,k) ) / 6      <-- gs_z above
//
// Production multigrid::solver::GaussSeidel (mg_solver.hh:198-217) computes
//   u = (rhs(u) + h^2 f) * c0   with c0 = -1/m_scheme.ccoeff()
// For stencil_7P, ccoeff = -6, rhs = sum of 6 neighbours, so the update is
//   u = (sum + h^2 f) / 6  --> matches L u = -f (apply = sum - 6u = -h^2 f).
//
// Use this variant when wiring gs_z directly into twoGrid_multibox's GS sites
// without negating f at the call site. All other semantics (halo exchange
// between colors, cluster-global iz coloring, x/y boundary skip) match gs_z.
void gs_z_neg_slab_double(const ZoomSlabLayout& L,
                          double* u_with_halo,
                          const double* f_with_halo,
                          double h);
void gs_z_neg_slab_float (const ZoomSlabLayout& L,
                          float*  u_with_halo,
                          const float*  f_with_halo,
                          float  h);

inline void gs_z_neg(const ZoomSlabLayout& L,
                     double* u, const double* f, double h)
{ gs_z_neg_slab_double(L, u, f, h); }
inline void gs_z_neg(const ZoomSlabLayout& L,
                     float* u, const float* f, float h)
{ gs_z_neg_slab_float(L, u, f, h); }

// Phase G.2b production-wire step 1 smoke test. Builds a single cluster with
// deterministic smooth u_init, f. Runs N sweeps of gs_z_neg via the slab path,
// AND independently runs N sweeps of an inlined production-formula serial loop
// (using the SAME formula production GaussSeidel applies for stencil_7P) on
// rank 0's full cluster. Verifies bit-identical match against BOTH:
//   (a) the single-rank slab reference (same impl on MPI_COMM_SELF), and
//   (b) the inlined production-formula serial reference.
// (a) verifies MPI decomposition correctness; (b) verifies the slab formula
// matches what twoGrid_multibox will see at the production GS site.
// Returns true iff both comparisons are bit-identical (max|err|==0).
bool smoke_test_gs_neg_single_cluster(int halo_w = 1, int n_sweeps = 4);

// ---------------------------------------------------------------------------
// Phase G.2b production-wire step 2: gs_z_neg variant that also skips cluster
// z-edge cells (kl=halo_w on sub_rank==0 and kl=halo_w+nz_r-1 on the last
// sub_rank). For wiring into a MeshvarBnd-aware path that uses cluster_nz =
// MeshvarBnd.size(2) + 2 — i.e. the cluster z-perimeter holds frozen BC cells.
//
// All other semantics match gs_z_neg.
void gs_z_neg_skip_z_slab_double(const ZoomSlabLayout& L,
                                 double* u_with_halo,
                                 const double* f_with_halo,
                                 double h);
void gs_z_neg_skip_z_slab_float (const ZoomSlabLayout& L,
                                 float*  u_with_halo,
                                 const float*  f_with_halo,
                                 float  h);

inline void gs_z_neg_skip_z(const ZoomSlabLayout& L,
                            double* u, const double* f, double h)
{ gs_z_neg_skip_z_slab_double(L, u, f, h); }
inline void gs_z_neg_skip_z(const ZoomSlabLayout& L,
                            float* u, const float* f, float h)
{ gs_z_neg_skip_z_slab_float(L, u, f, h); }

// ---------------------------------------------------------------------------
// Phase G.2b B.5.0: 7-point Laplacian apply on the z-slab.
//
// Computes
//   out(i,j,k) = u(i-1,j,k) + u(i+1,j,k) + u(i,j-1,k) + u(i,j+1,k)
//              + u(i,j,k-1) + u(i,j,k+1) - 6 * u(i,j,k)
// over the cluster interior, EXCLUDING the cluster perimeter cells that
// hold parent BC values (i in [1,cnx-1), j in [1,cny-1), cluster-global
// k in [1,cnz-1)). Mirrors stencil_7P::apply in schemes.hh.
//
// Does NOT divide by h^2 — caller scales for FAS source assembly. Pure
// local kernel: no MPI inside. Caller must populate u_with_halo's z-halo
// via halo_exchange_z before invoking. Output 'out_with_halo' uses the
// same local_with_halo_size layout; only update-domain cells are written
// (others untouched, so caller may pre-zero or pre-fill the buffer).
//
// Foundation for B.5.1 wiring into twoGrid_multibox_spmd's apply+restrict
// path (the B.2b.6 site where the apply currently runs rank-0-local on
// the box owner before restrict_meshvarbnd dispatches the 8-cell average).
// ---------------------------------------------------------------------------
void apply_z_slab_double(const ZoomSlabLayout& L,
                         const double* u_with_halo,
                         double*       out_with_halo);
void apply_z_slab_float (const ZoomSlabLayout& L,
                         const float*  u_with_halo,
                         float*        out_with_halo);

inline void apply_z_slab(const ZoomSlabLayout& L,
                         const double* u, double* o)
{ apply_z_slab_double(L, u, o); }
inline void apply_z_slab(const ZoomSlabLayout& L,
                         const float* u, float* o)
{ apply_z_slab_float(L, u, o); }

// Smoke test: builds a single cluster, fills u with a smooth pattern,
// runs apply via the slab path, compares against an inlined serial
// reference. Bit-identical at np=1 (kernel and serial-ref share buffer
// layout); ULP-tolerant (<= 16*eps*max|ref|) at np>=2 because clang
// FMA-contracts `sum - 6*u` differently between the halo-padded slab
// buffer and the contiguous full-cluster reference buffer. Each cell's
// arithmetic is independent so the diff does not cascade.
bool smoke_test_apply_z_slab_single_cluster(int halo_w = 1);

// ---------------------------------------------------------------------------
// Phase G.2b B.5.2 smoke: restrict_meshvarbnd_from_slab vs restrict_meshvarbnd.
// Builds a fine MeshvarBnd vf with a deterministic smooth pattern, scatters it
// to a per-rank slab using existing scatter_from primitive, then drives the
// new from-slab bridge. Compares the resulting coarse MeshvarBnd against the
// reference produced by the existing scatter-internally restrict_meshvarbnd.
// Bit-identical contract (max|err| == 0).
bool smoke_test_restrict_meshvarbnd_from_slab_single_cluster(int halo_w = 1);

// Phase G.2b B.5.2-prod smoke: fused apply_meshvarbnd_to_slab +
// restrict_meshvarbnd_from_slab vs the existing apply_meshvarbnd +
// restrict_meshvarbnd path. Both paths consume the same vf MeshvarBnd and run
// the same apply_z_slab on the same scatter; Path B redistributes the apply
// result across sub_comm into Layout B (no owner gather/scatter round-trip).
// Bit-identical contract (max|err| == 0).
bool smoke_test_apply_meshvarbnd_to_slab_fused_single_cluster(int halo_w = 1);
} // namespace zoom_slab
} // namespace MUSIC

// Forward decl so the bridge prototype can name MeshvarBnd<T> without forcing
// every caller of zoom_slab.hh to include mesh.hh.
template<typename T> class MeshvarBnd;

namespace MUSIC {
namespace zoom_slab {

// ---------------------------------------------------------------------------
// Phase G.2b production-wire step 2: end-to-end GS on a per-box MeshvarBnd<T>.
//
// Bridge primitive: packs MeshvarBnd's interior + 1-cell BC perimeter into a
// cluster_full buffer (cluster shape = MeshvarBnd shape + 2 on each axis),
// scatters across sub_comm, exchanges z-halo, runs N sweeps of
// gs_z_neg_skip_z_slab (skips cluster perimeter cells, which hold frozen BC),
// gathers, and writes the updated interior back into the MeshvarBnd.
//
// Preconditions:
//   - u and f are valid pointers ONLY on sub_comm rank == box_owner; on other
//     ranks they may be NULL.
//   - u and f have identical interior dimensions (size(0,1,2)) and m_nbnd >= 1.
//   - Box-owner's u and f hold the per-box state from twoGrid_multibox: u is
//     the current guess (interior + parent-prolonged BC at the perimeter), f
//     is the per-box source. The BC perimeter is NOT modified.
//
// Postcondition: u's interior cells (i,j,k in [0, size(d))) are updated; BC
// cells (at indices -1 and size(d)) are unchanged.
//
// Matches the per-box GaussSeidel call sites in mg_solver.hh:554-555 (pre-
// smooth) and :689-690 (post-smooth) for the stencil_7P / O2 case. h is the
// physical cell spacing (same as production's `h = 1.0/(1<<ilevel)`).
void gs_z_neg_meshvarbnd_double(int box_owner,
                                MeshvarBnd<double>* u,
                                const MeshvarBnd<double>* f,
                                double h, int n_sweeps,
                                MPI_Comm sub_comm, int halo_w = 1);
void gs_z_neg_meshvarbnd_float (int box_owner,
                                MeshvarBnd<float>*  u,
                                const MeshvarBnd<float>*  f,
                                float  h, int n_sweeps,
                                MPI_Comm sub_comm, int halo_w = 1);

inline void gs_z_neg_meshvarbnd(int box_owner,
                                MeshvarBnd<double>* u,
                                const MeshvarBnd<double>* f,
                                double h, int n_sweeps,
                                MPI_Comm sub_comm, int halo_w = 1)
{ gs_z_neg_meshvarbnd_double(box_owner, u, f, h, n_sweeps, sub_comm, halo_w); }
inline void gs_z_neg_meshvarbnd(int box_owner,
                                MeshvarBnd<float>* u,
                                const MeshvarBnd<float>* f,
                                float h, int n_sweeps,
                                MPI_Comm sub_comm, int halo_w = 1)
{ gs_z_neg_meshvarbnd_float (box_owner, u, f, h, n_sweeps, sub_comm, halo_w); }

// ---------------------------------------------------------------------------
// Phase G.2b B.5.1: collective apply Laplacian on sub_comm.
//
// Computes (apply(u)) / h2 cell-by-cell using the 7-point stencil and writes
// the result into out's interior. Mirrors gs_z_neg_meshvarbnd_impl plumbing:
// box_owner packs u interior + 1-cell BC perimeter into a cluster_full buffer,
// scatters across sub_comm, exchanges z-halo, runs apply_z_slab, gathers, and
// writes back to out's interior (BC perimeter untouched).
//
// Replaces the rank-0-local apply loop at the B.2b.6 site in
// mg_solver.hh::twoGrid_multibox_spmd (where apply_uf scratch is built before
// restrict_meshvarbnd dispatches the 8-cell average). Result is bit-identical
// to the rank-0 loop at np==1 and ULP-tolerant at np>=2 (same FMA-contract
// behavior documented for apply_z_slab; see B.5.0 docstring).
//
// Preconditions:
//   - u and out are valid ONLY on sub_comm rank == box_owner; may be NULL elsewhere.
//   - u and out have identical interior dimensions; u->m_nbnd >= 1.
//
// Returns true iff the collective path executed; returns false (with no writes)
// on sub_size==1 or halo_w too large. In the false case the caller must run a
// local apply loop on box_owner.
bool apply_meshvarbnd_double(int box_owner,
                             const MeshvarBnd<double>* u,
                             MeshvarBnd<double>* out,
                             double h2,
                             MPI_Comm sub_comm, int halo_w = 1);
bool apply_meshvarbnd_float (int box_owner,
                             const MeshvarBnd<float>*  u,
                             MeshvarBnd<float>*  out,
                             float  h2,
                             MPI_Comm sub_comm, int halo_w = 1);

inline bool apply_meshvarbnd(int box_owner,
                             const MeshvarBnd<double>* u,
                             MeshvarBnd<double>* out,
                             double h2,
                             MPI_Comm sub_comm, int halo_w = 1)
{ return apply_meshvarbnd_double(box_owner, u, out, h2, sub_comm, halo_w); }
inline bool apply_meshvarbnd(int box_owner,
                             const MeshvarBnd<float>* u,
                             MeshvarBnd<float>* out,
                             float h2,
                             MPI_Comm sub_comm, int halo_w = 1)
{ return apply_meshvarbnd_float (box_owner, u, out, h2, sub_comm, halo_w); }

// ---------------------------------------------------------------------------
// Phase G.2b B.2b.4: collective restrict on sub_comm. Bridges a per-box fine
// MeshvarBnd<T> into a coarse sub-region of its parent MeshvarBnd<T>, mirroring
// mg_straight::restrict in mg_operators.hh:
//
//   Vc(i+ox, j+oy, k+oz) = (1/8) * sum_{di,dj,dk in {0,1}}
//                              vf(2*i+di, 2*j+dj, 2*k+dk)
//
// where (ox,oy,oz) = vf->offset(d) — i.e. the offset of this fine box within
// its parent. The bridge:
//   (1) box_owner broadcasts fine dims (nx,ny,nz) + parent offsets (ox,oy,oz)
//   (2) all ranks build Lf (nx,ny,nz) and Lc (nx/2,ny/2,nz/2) layouts; alignment
//       is verified (fine z-split must be 2x coarse z-split)
//   (3) box_owner packs vf interior into cluster_full buffer; scatter to slabs
//   (4) all ranks run restrict_z_slab locally
//   (5) gather coarse slabs back to box_owner
//   (6) box_owner writes into Vc at parent offset (i+ox, j+oy, k+oz)
//
// Returns true iff the collective path executed; returns false on
//   - sub_size == 1, or
//   - alignment failure (nz / 2 not divisible by sub_size, etc.)
// In the false-return case, no writes were made and the caller must perform a
// local restrict instead (on box_owner only — non-owners' Vc is NULL/inaccessible).
//
// Preconditions:
//   - vf and Vc are valid ONLY on sub_comm rank == box_owner; may be NULL elsewhere.
//   - vf->size(d) must be even on all 3 axes (production restrict reads in 2x blocks).
//   - Vc must be large enough to hold (ox..ox+nx/2-1) in each dim.
bool restrict_meshvarbnd_double(int box_owner,
                                const MeshvarBnd<double>* vf,
                                MeshvarBnd<double>* Vc,
                                MPI_Comm sub_comm);
bool restrict_meshvarbnd_float (int box_owner,
                                const MeshvarBnd<float>*  vf,
                                MeshvarBnd<float>*  Vc,
                                MPI_Comm sub_comm);

inline bool restrict_meshvarbnd(int box_owner,
                                const MeshvarBnd<double>* vf,
                                MeshvarBnd<double>* Vc,
                                MPI_Comm sub_comm)
{ return restrict_meshvarbnd_double(box_owner, vf, Vc, sub_comm); }
inline bool restrict_meshvarbnd(int box_owner,
                                const MeshvarBnd<float>* vf,
                                MeshvarBnd<float>* Vc,
                                MPI_Comm sub_comm)
{ return restrict_meshvarbnd_float (box_owner, vf, Vc, sub_comm); }

// ---------------------------------------------------------------------------
// Phase G.2b B.5.2: restrict bridge that consumes a fine z-slab already in
// scattered form (mirrors restrict_meshvarbnd_impl but skips the fine-side
// scatter). Designed for B.5.4 keep-in-slab wiring, where the apply step
// leaves apply_uf as a per-rank fine slab and we want to feed it directly
// into restriction without a gather→scatter round-trip through the owner.
//
// Semantics:
//   (1) all sub_comm ranks already share a fine layout Lf describing the
//       z-slab decomposition of the fine cluster
//   (2) each rank provides its local fine slab (interior only, no halo,
//       size local_interior_size(Lf)) in my_vf_int
//   (3) the bridge builds the matching coarse layout Lc, pads the fine slab
//       with halo, runs restrict_impl locally, extracts coarse interior,
//       gathers to box_owner, and writes into Vc at parent offset
//       (i+Vc_off_x, j+Vc_off_y, k+Vc_off_z)
//
// Returns true iff the collective path executed; returns false on the same
// conditions as restrict_meshvarbnd_impl (sub_size==1, axis parity, alignment,
// halo_w==0). In the false-return case no writes were made.
//
// Preconditions:
//   - Lf is identical across all sub_comm ranks (typically constructed via
//     make_layout(sub_comm, ...) collectively)
//   - my_vf_int is non-null on every rank (size local_interior_size(Lf))
//   - Vc is valid ONLY on sub_comm rank == box_owner; may be NULL elsewhere
//   - Vc must contain the sub-region [Vc_off_x..Vc_off_x+Lf.cluster_nx/2-1]
//     × ... fully in its interior
//   - Lf.halo_w >= 1 (required by restrict_impl's halo-stride buffer layout)
bool restrict_meshvarbnd_from_slab_double(int box_owner,
                                          const ZoomSlabLayout& Lf,
                                          const double* my_vf_int,
                                          MeshvarBnd<double>* Vc,
                                          int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                          MPI_Comm sub_comm);
bool restrict_meshvarbnd_from_slab_float (int box_owner,
                                          const ZoomSlabLayout& Lf,
                                          const float*  my_vf_int,
                                          MeshvarBnd<float>*  Vc,
                                          int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                          MPI_Comm sub_comm);

inline bool restrict_meshvarbnd_from_slab(int box_owner,
                                          const ZoomSlabLayout& Lf,
                                          const double* my_vf_int,
                                          MeshvarBnd<double>* Vc,
                                          int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                          MPI_Comm sub_comm)
{ return restrict_meshvarbnd_from_slab_double(box_owner, Lf, my_vf_int, Vc,
                                              Vc_off_x, Vc_off_y, Vc_off_z, sub_comm); }
inline bool restrict_meshvarbnd_from_slab(int box_owner,
                                          const ZoomSlabLayout& Lf,
                                          const float* my_vf_int,
                                          MeshvarBnd<float>* Vc,
                                          int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                          MPI_Comm sub_comm)
{ return restrict_meshvarbnd_from_slab_float (box_owner, Lf, my_vf_int, Vc,
                                              Vc_off_x, Vc_off_y, Vc_off_z, sub_comm); }

// ---------------------------------------------------------------------------
// Phase G.2b B.5.4.b: collective restrict on sub_comm from a *padded* z-slab
// layout (BC perimeter of `perimeter` cells on every side). Composes the
// perimeter-strip redistribute with the existing from_slab restrict so the
// caller can keep uf in the smoother's padded layout (cluster_n = nxf+2p,
// halo_w >= 1) across pre-smooth → u-restrict without an interior round-trip.
//
// Preconditions:
//   - Lf_padded.cluster_nx/ny/nz == 2p + (un-padded fine dim) on every axis
//   - my_padded_int is non-null on every rank (size local_interior_size(Lf_padded))
//   - perimeter >= 1
//   - Un-padded dims must be even on every axis and the un-padded nz must be a
//     multiple of 2 * sub_size (so the implied un-padded layout exists)
//   - Vc / Vc_off_* contract identical to restrict_meshvarbnd_from_slab
//
// Returns true iff the bridge executed. False on sub_size<2, perimeter<1,
// non-even dims, nz alignment, or downstream restrict failure → caller falls
// back to scatter-Vc-into-padded-slab + the existing bridge.
bool restrict_meshvarbnd_from_padded_slab_double(int box_owner,
                                                 const ZoomSlabLayout& Lf_padded,
                                                 const double* my_padded_int,
                                                 int perimeter,
                                                 MeshvarBnd<double>* Vc,
                                                 int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                                 MPI_Comm sub_comm);
bool restrict_meshvarbnd_from_padded_slab_float (int box_owner,
                                                 const ZoomSlabLayout& Lf_padded,
                                                 const float*  my_padded_int,
                                                 int perimeter,
                                                 MeshvarBnd<float>*  Vc,
                                                 int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                                 MPI_Comm sub_comm);

inline bool restrict_meshvarbnd_from_padded_slab(int box_owner,
                                                 const ZoomSlabLayout& Lf_padded,
                                                 const double* my_padded_int,
                                                 int perimeter,
                                                 MeshvarBnd<double>* Vc,
                                                 int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                                 MPI_Comm sub_comm)
{ return restrict_meshvarbnd_from_padded_slab_double(box_owner, Lf_padded, my_padded_int,
                                                     perimeter, Vc,
                                                     Vc_off_x, Vc_off_y, Vc_off_z, sub_comm); }
inline bool restrict_meshvarbnd_from_padded_slab(int box_owner,
                                                 const ZoomSlabLayout& Lf_padded,
                                                 const float* my_padded_int,
                                                 int perimeter,
                                                 MeshvarBnd<float>* Vc,
                                                 int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                                 MPI_Comm sub_comm)
{ return restrict_meshvarbnd_from_padded_slab_float (box_owner, Lf_padded, my_padded_int,
                                                     perimeter, Vc,
                                                     Vc_off_x, Vc_off_y, Vc_off_z, sub_comm); }

// ---------------------------------------------------------------------------
// Phase G.2b B.5.2-prod: combined apply + redistribute into the restrict-fine
// z-slab layout. Mirrors apply_meshvarbnd's owner-side scatter + apply path,
// but instead of gathering the result back to box_owner it MPI_Alltoallv-
// redistributes the per-rank Layout-A interior into the caller's pre-built
// Layout-B (= Lf) slab — with the 1/h2 scaling folded into the pack pass.
//
// Designed to chain directly into restrict_meshvarbnd_from_slab: the two
// together replace apply_meshvarbnd + restrict_meshvarbnd at the B.2b.6 site
// in mg_solver.hh::twoGrid_multibox_spmd and eliminate the owner gather +
// re-scatter round trip between them.
//
// Layout precondition: caller has built `Lf` via make_layout(sub_comm, ...,
// nx, ny, nz, halo_w>=1) where (nx,ny,nz) = u->size(d). Cluster offset is
// irrelevant; pass 0,0,0. The internal Layout A is built with cluster_nx =
// nx+2, cluster_ny = ny+2, cluster_nz = nz+2 (1-cell BC perimeter) using the
// same halo_w.
//
// Preconditions:
//   - u is valid ONLY on sub_comm rank == box_owner; may be NULL elsewhere.
//   - u->size(d) matches Lf.cluster_n* on every axis; u->m_nbnd >= 1.
//   - my_apply_int is non-null on every rank, sized local_interior_size(Lf).
//
// Returns true iff the bridge executed. Returns false on sub_size<2, dim
// mismatch, halo_w too large, or Lf geometry mismatch. The caller must fall
// back to apply_meshvarbnd + restrict_meshvarbnd when false is returned.
bool apply_meshvarbnd_to_slab_double(int box_owner,
                                     const MeshvarBnd<double>* u,
                                     const ZoomSlabLayout& Lf,
                                     double* my_apply_int,
                                     double h2,
                                     MPI_Comm sub_comm);
bool apply_meshvarbnd_to_slab_float (int box_owner,
                                     const MeshvarBnd<float>*  u,
                                     const ZoomSlabLayout& Lf,
                                     float*  my_apply_int,
                                     float  h2,
                                     MPI_Comm sub_comm);

inline bool apply_meshvarbnd_to_slab(int box_owner,
                                     const MeshvarBnd<double>* u,
                                     const ZoomSlabLayout& Lf,
                                     double* my_apply_int,
                                     double h2,
                                     MPI_Comm sub_comm)
{ return apply_meshvarbnd_to_slab_double(box_owner, u, Lf, my_apply_int, h2, sub_comm); }
inline bool apply_meshvarbnd_to_slab(int box_owner,
                                     const MeshvarBnd<float>* u,
                                     const ZoomSlabLayout& Lf,
                                     float* my_apply_int,
                                     float h2,
                                     MPI_Comm sub_comm)
{ return apply_meshvarbnd_to_slab_float (box_owner, u, Lf, my_apply_int, h2, sub_comm); }

// ---------------------------------------------------------------------------
// Phase G.2b B.2b.5: collective prolong_add on sub_comm. Bridges a coarse
// parent sub-region into a fine per-box MeshvarBnd<T> via += semantics,
// mirroring mg_straight::prolong_add in mg_operators.hh:
//
//   vf(2i+di, 2j+dj, 2k+dk) += Vc(i+ox, j+oy, k+oz)   for di,dj,dk in {0,1}
//
// where (nx,ny,nz) = (vf->size(d)/2) and (ox,oy,oz) = vf->offset(d) — i.e.
// the fine box absorbs the coarse correction defined on its parent sub-region.
//
// The bridge:
//   (1) box_owner broadcasts fine dims (nxf,nyf,nzf) + offsets (ox,oy,oz)
//   (2) all ranks build Lf (nxf,...) and Lc (nxf/2,...) layouts; alignment verified
//   (3) box_owner packs vf interior + the (nxc,nyc,nzc) sub-region of Vc starting
//       at (ox,oy,oz) into cluster_full buffers
//   (4) scatter both fine and coarse to z-slabs
//   (5) all ranks run prolong_add_z_slab locally (vf_slab += Vc_slab semantics)
//   (6) gather fine z-slabs back to box_owner
//   (7) box_owner writes resulting vf back into (*vf)(i,j,k) interior cells
//
// Returns true iff the collective path executed; returns false on sub_size==1
// or alignment failure (same conditions as restrict_meshvarbnd). The caller
// must perform a local prolong_add when false is returned.
//
// Preconditions:
//   - vf and Vc are valid ONLY on sub_comm rank == box_owner; may be NULL elsewhere.
//   - vf->size(d) must be even on all 3 axes.
//   - Vc must contain the sub-region [ox..ox+nxc-1] x ... fully in its interior.
bool prolong_add_meshvarbnd_double(int box_owner,
                                   const MeshvarBnd<double>* Vc,
                                   MeshvarBnd<double>* vf,
                                   MPI_Comm sub_comm);
bool prolong_add_meshvarbnd_float (int box_owner,
                                   const MeshvarBnd<float>*  Vc,
                                   MeshvarBnd<float>*  vf,
                                   MPI_Comm sub_comm);

inline bool prolong_add_meshvarbnd(int box_owner,
                                   const MeshvarBnd<double>* Vc,
                                   MeshvarBnd<double>* vf,
                                   MPI_Comm sub_comm)
{ return prolong_add_meshvarbnd_double(box_owner, Vc, vf, sub_comm); }
inline bool prolong_add_meshvarbnd(int box_owner,
                                   const MeshvarBnd<float>* Vc,
                                   MeshvarBnd<float>* vf,
                                   MPI_Comm sub_comm)
{ return prolong_add_meshvarbnd_float (box_owner, Vc, vf, sub_comm); }

// Phase G.2b production-wire step 2 smoke test. Constructs a small MeshvarBnd
// pair on box-owner (rank 0), fills interior + 1-cell BC perimeter with a
// deterministic smooth pattern. Saves initial u, then:
//   (1) runs gs_z_neg_meshvarbnd for n_sweeps via the slab path
//   (2) runs inlined production-formula GaussSeidel on a copy of the initial
//       u (using MeshvarBnd's operator() which reads through to BC) for
//       n_sweeps on box-owner
// Compares interior cells bit-identical. Returns true iff max|err|==0.
bool smoke_test_gs_neg_meshvarbnd_single_cluster(int halo_w = 1, int n_sweeps = 4);

// Phase G.2b A': opt-in toggle that gates the bridge call at the per-box GS
// sites in multigrid::solver::twoGrid_multibox. Set once per Poisson solve from
// poisson.cc reading `setup.zoom_slab_smoother`. Default false — bridge is
// only invoked when smoother_enabled() && S::order==2 && z-slabs populated.
void set_smoother_enabled(bool v);
bool smoother_enabled();

// Phase G.2b B.2b.2: opt-in toggle for the SPMD composite-MG path inside
// twoGrid_multibox. Default false → existing rank-0-only V-cycle wrapped by
// with_pbox_distributed. When true the call-site routes through
// with_pbox_distributed_spmd + twoGrid_multibox_spmd (owner-gated per-box
// loops with B.2b.1 parent broadcast/accumulate). Set once per solve from
// poisson.cc reading `setup.zoom_slab_spmd_multigrid`. Independent of
// smoother_enabled / subcomm_policy.
void set_spmd_multigrid_enabled(bool v);
bool spmd_multigrid_enabled();

// Phase G.2b B.5.4.a: opt-in toggle for keep-in-slab pre/post smoothing inside
// twoGrid_multibox_spmd. When on (and per-op bridge active), the N pre/post
// smoothing iterations share a single (size+4)^3 z-slab buffer so uf is
// scattered once at the top of each phase and gathered once at the bottom,
// instead of bouncing through MeshvarBnd on every iteration. Default false.
void set_keep_slab_smooth_enabled(bool v);
bool keep_slab_smooth_enabled();

// Phase G.2b B.5.4.b: opt-in toggle for keep-in-slab u-restrict. Builds on
// B.5.4.a — when on (and B.5.4.a is on), the pre-smooth phase uses
// smooth_pre_post_n_meshvarbnd_keep_slab to return the padded-cluster
// interior buffer; the immediately following u-restrict consumes it via
// restrict_meshvarbnd_from_padded_slab (perimeter=2), eliminating one
// gather+scatter round-trip per box per V-cycle. Default false; requires
// keep_slab_smooth_enabled() to fire.
void set_keep_slab_urestrict_enabled(bool v);
bool keep_slab_urestrict_enabled();

// ---------------------------------------------------------------------------
// Phase G.2b (B.1): per-cluster sub_comm registry for the bridge.
//
// The (A') wire hard-coded sub_comm = MPI_COMM_WORLD. (B.1) introduces a
// registry of per-cluster sub_comms so the bridge can dispatch per-cluster on
// any policy without each call-site re-deriving it. The registry is global
// (module-level static); the eventual D.6 path adds it to a more structured
// location, but for B.1 a single registry suffices because multibox topology
// is fixed for the duration of a run.
//
// Policy options:
//   "world"       (default) — every cluster's sub_comm == MPI_COMM_WORLD.
//                 Identical to A' behavior. No MPI_Comm_split. Bit-identical
//                 regression guaranteed.
//   "self"        — every cluster's sub_comm == MPI_COMM_SELF (singleton).
//                 At np=1 identical to "world"; at np>1 only the calling rank
//                 participates in the bridge (the np>1 gate is still on, so
//                 not exercised yet — sets the stage for B.2+).
//   "round_robin" — splits ranks into nb groups via MPI_Comm_split. Group g
//                 owns cluster g. Requires nb <= np. Created comms must be
//                 freed via free_subcomm_registry() before MPI_Finalize.
//
// Lifecycle:
//   set_subcomm_policy(...)  — local, just stores a string.
//   build_subcomm_registry(...) — collective on MPI_COMM_WORLD. Frees previous
//                 registry first, then builds per the current policy.
//   subcomm_for_box(level, box_id) — local lookup. Returns MPI_COMM_WORLD when
//                 registry is empty for that slot (graceful default).
//   free_subcomm_registry() — collective on MPI_COMM_WORLD. Safe to call when
//                 the registry is empty or built from "world"/"self" policy
//                 (those skip the MPI_Comm_free calls).
// ---------------------------------------------------------------------------
void set_subcomm_policy(const std::string& p);
const std::string& subcomm_policy();

void build_subcomm_registry(const std::vector<std::size_t>& nb_per_level);
MPI_Comm subcomm_for_box(unsigned level, std::size_t box_id);
void free_subcomm_registry();

// ---------------------------------------------------------------------------
// Phase G.2b: restrict (R) fine z-slab to coarse z-slab.
//
// 8-cell straight average (mirrors mg_straight::restrict in mg_operators.hh):
//   V_c(ic, jc, kc) = (1/8) * sum_{di,dj,dk in {0,1}} v_f(2*ic+di, 2*jc+dj, 2*kc+dk)
//
// Layout precondition (caller must ensure; restrict_impl checks at runtime):
//   Lc shares the same sub_comm as Lf
//   Lc.cluster_nx == Lf.cluster_nx / 2, ditto ny/nz
//   2 * Lc.my_z0 == Lf.my_z0  and  2 * Lc.my_z1 == Lf.my_z1
// (i.e. each rank's fine slab covers exactly the cells needed for its coarse
// slab; restrict is a pure-local op with no MPI communication.)
//
// vf and Vc both use local_with_halo storage. vf's interior must be fresh;
// vf's z-halo is NOT read (restrict reads only interior cells in z). Vc's
// interior is overwritten; its halo is NOT touched — caller does
// halo_exchange_z(Lc, Vc) if the next op needs Vc's halo.
void restrict_z_slab_double(const ZoomSlabLayout& Lf, const ZoomSlabLayout& Lc,
                            const double* vf_with_halo,
                            double*       Vc_with_halo);
void restrict_z_slab_float (const ZoomSlabLayout& Lf, const ZoomSlabLayout& Lc,
                            const float*  vf_with_halo,
                            float*        Vc_with_halo);

inline void restrict_z(const ZoomSlabLayout& Lf, const ZoomSlabLayout& Lc,
                       const double* vf, double* Vc)
{ restrict_z_slab_double(Lf, Lc, vf, Vc); }
inline void restrict_z(const ZoomSlabLayout& Lf, const ZoomSlabLayout& Lc,
                       const float* vf, float* Vc)
{ restrict_z_slab_float(Lf, Lc, vf, Vc); }

// Phase G.2b smoke test for restrict_z. Builds a single fine cluster, fills
// vf with a deterministic smooth pattern on sub_comm rank 0, scatters into
// fine z-slab, restricts to coarse z-slab, gathers, and compares cell-by-cell
// against a serial reference computed on rank 0. Returns true iff
// max|Vc_slab - Vc_serial| == 0 (double) or < tol (float).
//
// Cluster_nz internally chosen so 2 | (cluster_nz / sub_size) on all 1/2/4/8.
bool smoke_test_restrict_single_cluster(int halo_w = 1);

// ---------------------------------------------------------------------------
// Phase G.2b: prolong (P) coarse z-slab to fine z-slab — pure injection.
//
// Zero-order injection (mirrors mg_straight::prolong in mg_operators.hh):
//   v_f(2*ic+di, 2*jc+dj, 2*kc+dk) = V_c(ic, jc, kc)   for all di,dj,dk in {0,1}
//
// Layout precondition (same alignment as restrict_z_slab):
//   Lc shares the same sub_comm as Lf
//   Lc.cluster_nx == Lf.cluster_nx / 2, ditto ny/nz
//   2 * Lc.my_z0 == Lf.my_z0  and  2 * Lc.my_z1 == Lf.my_z1
//
// Vc and vf both use local_with_halo storage. Vc's interior is read; its halo
// is NOT consulted (injection writes to the same 2x2x2 fine block, all derived
// from one local coarse cell). vf's interior is overwritten; vf's halo is NOT
// touched — caller does halo_exchange_z(Lf, vf) if next op needs vf's halo.
void prolong_z_slab_double(const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                           const double* Vc_with_halo,
                           double*       vf_with_halo);
void prolong_z_slab_float (const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                           const float*  Vc_with_halo,
                           float*        vf_with_halo);

inline void prolong_z(const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                      const double* Vc, double* vf)
{ prolong_z_slab_double(Lc, Lf, Vc, vf); }
inline void prolong_z(const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                      const float* Vc, float* vf)
{ prolong_z_slab_float(Lc, Lf, Vc, vf); }

// Phase G.2b smoke test for prolong_z. Mirrors restrict smoke: build a smooth
// coarse pattern, scatter to coarse z-slabs, prolong to fine z-slabs, gather,
// compare bit-identical against a single-rank reference. Returns true on success.
bool smoke_test_prolong_single_cluster(int halo_w = 1);

// ---------------------------------------------------------------------------
// Phase G.2b: prolong_add (P_add) — coarse z-slab to fine z-slab, += semantics.
//
// Mirrors mg_straight::prolong_add in mg_operators.hh (which is what the
// production poisson solver uses — see src/poisson.cc:101-107). Each of the 8
// fine children of a coarse cell has the coarse cell's value ADDED to its
// current contents:
//   v_f(2*ic+di, 2*jc+dj, 2*kc+dk) += V_c(ic, jc, kc)   for di,dj,dk in {0,1}
//
// Layout precondition (same as prolong_z_slab):
//   Lc shares the same sub_comm as Lf
//   Lc.cluster_nx == Lf.cluster_nx / 2, ditto ny/nz
//   2 * Lc.my_z0 == Lf.my_z0  and  2 * Lc.my_z1 == Lf.my_z1
//
// Vc and vf use local_with_halo storage. Vc's halo is NOT consulted; vf's
// halo is NOT touched. Vf's interior is read-modify-write.
//
// Typical use: V-cycle uncoarsening — coarse correction is prolong-added to
// the fine-level guess after the recursive solve.
void prolong_add_z_slab_double(const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                               const double* Vc_with_halo,
                               double*       vf_with_halo);
void prolong_add_z_slab_float (const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                               const float*  Vc_with_halo,
                               float*        vf_with_halo);

inline void prolong_add_z(const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                          const double* Vc, double* vf)
{ prolong_add_z_slab_double(Lc, Lf, Vc, vf); }
inline void prolong_add_z(const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                          const float* Vc, float* vf)
{ prolong_add_z_slab_float(Lc, Lf, Vc, vf); }

// Phase G.2b smoke test for prolong_add_z. Same pattern as prolong smoke, but
// vf is initialized to a nonzero smooth pattern (different from Vc) so the
// += semantics is actually exercised, then compared bit-identical against the
// single-rank reference of the same impl.
bool smoke_test_prolong_add_single_cluster(int halo_w = 1);

// ---------------------------------------------------------------------------
// Phase G.2b B.5.3a: prolong_bnd_z_slab — tricubic BC perimeter write.
//
// Port of mg_cubic::prolong_bnd (mg_operators.hh:604). Writes the m_nbnd=2
// boundary layer on each side of the fine cluster by tricubic interpolation
// from the (replicated) coarse parent. Foundation kernel for B.5.3b's
// interp_O3_fluxcorr::interp_coarse_fine — that adds 6 flux-correction faces
// using interp2/interp2left/interp2right on top of this BC write.
//
// Buffer layouts:
//   * coarse_buf: REPLICATED on every sub_comm rank, in cluster frame
//                 (cnxc, cnyc, cnzc). cz fastest, row-major. Caller passes
//                 cluster-frame coarse offset (ox_c, oy_c, oz_c) such that the
//                 fine interior maps to coarse cluster x ∈ [ox_c, ox_c+nx).
//                 ox_c,oy_c,oz_c >= 3 so cubic reads at cluster x ∈
//                 [ox_c-3, ox_c+nx+2] stay in bounds.
//   * fine_slab_buf: SLABBED across sub_comm via LfC.
//                    cluster_nx == 2*nx+4 (etc); halo_w >= 1.
//                    BC perimeter cells live in rank-0 / rank-(last) INTERIOR
//                    (since fine MVBND z=-2,-1 map to cluster z=0,1 which
//                    rank 0 owns when my_z0=0). Halo cells are NOT written.
//
// Returns false on cluster-shape mismatch, halo_w<1, ox_c<3, or cnxc too
// small for the cubic stencil.
bool prolong_bnd_z_slab_double(int cnxc, int cnyc, int cnzc,
                               const double* coarse_buf,
                               int ox_c, int oy_c, int oz_c,
                               int nx, int ny, int nz,
                               const ZoomSlabLayout& LfC,
                               double* fine_slab_buf);
bool prolong_bnd_z_slab_float (int cnxc, int cnyc, int cnzc,
                               const float*  coarse_buf,
                               int ox_c, int oy_c, int oz_c,
                               int nx, int ny, int nz,
                               const ZoomSlabLayout& LfC,
                               float*  fine_slab_buf);

inline bool prolong_bnd_z_slab(int cnxc, int cnyc, int cnzc,
                               const double* coarse_buf,
                               int ox_c, int oy_c, int oz_c,
                               int nx, int ny, int nz,
                               const ZoomSlabLayout& LfC, double* fine_slab_buf)
{ return prolong_bnd_z_slab_double(cnxc, cnyc, cnzc, coarse_buf,
                                    ox_c, oy_c, oz_c, nx, ny, nz,
                                    LfC, fine_slab_buf); }
inline bool prolong_bnd_z_slab(int cnxc, int cnyc, int cnzc,
                               const float* coarse_buf,
                               int ox_c, int oy_c, int oz_c,
                               int nx, int ny, int nz,
                               const ZoomSlabLayout& LfC, float* fine_slab_buf)
{ return prolong_bnd_z_slab_float (cnxc, cnyc, cnzc, coarse_buf,
                                    ox_c, oy_c, oz_c, nx, ny, nz,
                                    LfC, fine_slab_buf); }

// Phase G.2b B.5.3a smoke test for prolong_bnd_z_slab. Builds a deterministic
// coarse sinusoid pattern replicated on all ranks, allocates per-rank fine
// slab, runs the kernel, gathers fine cluster to rank 0, runs the same kernel
// with sub_comm = MPI_COMM_SELF as the reference, compares cell-by-cell.
// Expect max|err| == 0 (both paths execute identical interp_cubic arithmetic).
bool smoke_test_prolong_bnd_single_cluster(int halo_w = 1);

// ---------------------------------------------------------------------------
// Phase G.2b B.5.3b: interp_cf_flux_z_slab — 6 per-face flux corrections on
// top of the cubic BC perimeter produced by B.5.3a's prolong_bnd_z_slab.
// Composite (B.5.3a + halo_exchange_z + B.5.3b) is the slab equivalent of
// interp_O3_fluxcorr::interp_coarse_fine (mg_interp.hh:434-636); B.5.3c will
// wire that composite at the twoGrid_multibox_spmd interp_cf site.
//
// Buffer layouts (same as B.5.3a):
//   * coarse_buf:     REPLICATED across sub_comm, cluster frame (cnxc,cnyc,cnzc).
//   * fine_slab_buf:  SLABBED via LfC; halo-padded. PRECONDITION: caller has
//                     populated the interior, run B.5.3a (BC cubic), and
//                     halo_exchange_z'd. Kernel OVERWRITES BC perimeter with
//                     flux-corrected values.
//
// Preconditions (returns false):
//   * cluster shape mismatch (LfC.cluster_n* != 2*n+4)
//   * halo_w < 1
//   * ox_c | oy_c | oz_c < 1
//   * cnxc < ox_c + nx + 2  (and same for y,z) — coarse +1 stencil for right/top/back
//   * nx | ny | nz odd
//   * LfC.my_z0 odd OR local_nz odd  (even-pair preservation across rank boundary;
//     equivalent to (nz+2) % sub_size == 0 in the standard slab layout)
//
// FP-bit-identical to interp_O3_fluxcorr::interp_coarse_fine: left face uses
// the 4-arg interp2 form (mg_interp.hh:500); the other 5 faces use the 7-arg
// form with explicit (-1, 0, 1) ordinates.
bool interp_cf_flux_z_slab_double(int cnxc, int cnyc, int cnzc,
                                  const double* coarse_buf,
                                  int ox_c, int oy_c, int oz_c,
                                  int nx, int ny, int nz,
                                  const ZoomSlabLayout& LfC,
                                  double* fine_slab_buf);
bool interp_cf_flux_z_slab_float (int cnxc, int cnyc, int cnzc,
                                  const float*  coarse_buf,
                                  int ox_c, int oy_c, int oz_c,
                                  int nx, int ny, int nz,
                                  const ZoomSlabLayout& LfC,
                                  float*  fine_slab_buf);

inline bool interp_cf_flux_z_slab(int cnxc, int cnyc, int cnzc,
                                  const double* coarse_buf,
                                  int ox_c, int oy_c, int oz_c,
                                  int nx, int ny, int nz,
                                  const ZoomSlabLayout& LfC, double* fine_slab_buf)
{ return interp_cf_flux_z_slab_double(cnxc, cnyc, cnzc, coarse_buf,
                                       ox_c, oy_c, oz_c, nx, ny, nz,
                                       LfC, fine_slab_buf); }
inline bool interp_cf_flux_z_slab(int cnxc, int cnyc, int cnzc,
                                  const float*  coarse_buf,
                                  int ox_c, int oy_c, int oz_c,
                                  int nx, int ny, int nz,
                                  const ZoomSlabLayout& LfC, float*  fine_slab_buf)
{ return interp_cf_flux_z_slab_float (cnxc, cnyc, cnzc, coarse_buf,
                                       ox_c, oy_c, oz_c, nx, ny, nz,
                                       LfC, fine_slab_buf); }

// Phase G.2b B.5.3b smoke test: tests composite (prolong_bnd + halo_exchange
// _z + interp_cf_flux) against single-rank reference. Expects max|err| == 0.
bool smoke_test_interp_cf_flux_single_cluster(int halo_w = 1);

// ---------------------------------------------------------------------------
// Phase G.2b B.5.3c: collective interp_coarse_fine on sub_comm. Bridges a
// parent MeshvarBnd<T> (uc) into a fine per-box MeshvarBnd<T> (uf) via the
// composite (prolong_bnd_z_slab + halo_exchange_z + interp_cf_flux_z_slab),
// mirroring interp_O3_fluxcorr::interp_coarse_fine (mg_interp.hh:434-636).
//
// Writes uf's 2-cell BC perimeter (m_nbnd=2 layer); reads uf's interior and
// uc's interior + 1 BC cell on each side (cluster reach
// [oxf-3, oxf+nxc_int+2] on uc-local indexing). The protocol Bcasts the
// coarse cluster sub-region (REPLICATED on every sub_comm rank), scatters
// the fine cluster to z-slabs, runs the composite, gathers fine back.
//
// Returns true iff the collective path executed; returns false on:
//   - sub_size == 1 (caller should run local interp_coarse_fine)
//   - any kernel precondition failure (parity, alignment, geometry,
//     uc coarse-reach OOB)
// Caller MUST run local interp_coarse_fine when this returns false.
//
// Preconditions (when valid uc/uf on owner):
//   - uf->size(d) even on all 3 axes and >= 2
//   - uf->offset(d) >= 1 (low-side coarse stencil reach)
//   - uc->size(d) >= uf->offset(d) + uf->size(d)/2 + 1 (hi-side reach)
//   - (nzf + 4) % (2 * sub_size) == 0  AND  local_nz / my_z0 even on each rank
//     (pair preservation for z-slab decomposition)
bool interp_coarse_fine_meshvarbnd_double(int box_owner,
                                          const MeshvarBnd<double>* uc,
                                          MeshvarBnd<double>* uf,
                                          MPI_Comm sub_comm);
bool interp_coarse_fine_meshvarbnd_float (int box_owner,
                                          const MeshvarBnd<float>*  uc,
                                          MeshvarBnd<float>*  uf,
                                          MPI_Comm sub_comm);

inline bool interp_coarse_fine_meshvarbnd(int box_owner,
                                          const MeshvarBnd<double>* uc,
                                          MeshvarBnd<double>* uf,
                                          MPI_Comm sub_comm)
{ return interp_coarse_fine_meshvarbnd_double(box_owner, uc, uf, sub_comm); }
inline bool interp_coarse_fine_meshvarbnd(int box_owner,
                                          const MeshvarBnd<float>* uc,
                                          MeshvarBnd<float>* uf,
                                          MPI_Comm sub_comm)
{ return interp_coarse_fine_meshvarbnd_float (box_owner, uc, uf, sub_comm); }

// ---------------------------------------------------------------------------
// Phase G.2b B.5.4.a: keep-in-slab N-iteration smoothing.
//
// Equivalent to N iterations of (interp_coarse_fine_meshvarbnd then
// gs_z_neg_meshvarbnd with n_sweeps=1), but using one shared (size+4)^3
// z-slab buffer so uf+ff are scattered once at entry and uf gathered once
// at exit. The slab includes the 2-cell flux-correction perimeter (required
// by interp_cf) so the 1-cell BC ring read by GS is naturally inside it.
//
// Returns true iff the collective path executed; returns false on:
//   - sub_size == 1
//   - uf dim parity / size failure (size(d) must be even, >= 2)
//   - uf->offset(d) < 1 (low-side coarse stencil reach)
//   - uc->size(d) < uf->offset(d) + uf->size(d)/2 + 1 (hi-side reach)
//   - (nzf + 4) % (2 * sub_size) != 0 (pair preservation)
//   - local_nz odd or my_z0 odd on any rank
//
// Preconditions (when valid uc/uf/ff on owner):
//   - uf->m_nbnd >= 2 (interp_cf flux corr writes ix=-2)
//   - ff->m_nbnd >= 1 (cluster pack only reads interior, so 0 also works but
//     the existing MeshvarBnd factory always gives m_nbnd >= 1)
//   - uf and ff have identical interior dims
//
// On false, caller MUST fall back to the existing per-op loop (N iterations
// of interp_coarse_fine_meshvarbnd + gs_z_neg_meshvarbnd).
bool smooth_pre_post_n_meshvarbnd_double(int box_owner,
                                         const MeshvarBnd<double>* uc,
                                         MeshvarBnd<double>* uf,
                                         const MeshvarBnd<double>* ff,
                                         double h, int n_sweeps,
                                         MPI_Comm sub_comm);
bool smooth_pre_post_n_meshvarbnd_float (int box_owner,
                                         const MeshvarBnd<float>*  uc,
                                         MeshvarBnd<float>*  uf,
                                         const MeshvarBnd<float>*  ff,
                                         float  h, int n_sweeps,
                                         MPI_Comm sub_comm);

inline bool smooth_pre_post_n_meshvarbnd(int box_owner,
                                         const MeshvarBnd<double>* uc,
                                         MeshvarBnd<double>* uf,
                                         const MeshvarBnd<double>* ff,
                                         double h, int n_sweeps,
                                         MPI_Comm sub_comm)
{ return smooth_pre_post_n_meshvarbnd_double(box_owner, uc, uf, ff, h, n_sweeps, sub_comm); }
inline bool smooth_pre_post_n_meshvarbnd(int box_owner,
                                         const MeshvarBnd<float>* uc,
                                         MeshvarBnd<float>* uf,
                                         const MeshvarBnd<float>* ff,
                                         float h, int n_sweeps,
                                         MPI_Comm sub_comm)
{ return smooth_pre_post_n_meshvarbnd_float (box_owner, uc, uf, ff, h, n_sweeps, sub_comm); }

// ---------------------------------------------------------------------------
// Phase G.2b B.5.4.b: keep-in-slab N-iteration smoothing.
//
// Identical observable effect to smooth_pre_post_n_meshvarbnd (uf is
// scattered/smoothed/gathered + written back on owner), but ADDITIONALLY
// hands the post-smooth padded-cluster interior buffer back to the caller
// via the output parameters:
//   - Lf_out            = the padded-cluster ZoomSlabLayout (cluster_n =
//                         {nxf,nyf,nzf}+4, halo_w=1) on every sub_comm rank
//   - my_uf_int_out     = per-rank interior buffer of the padded cluster
//                         (size local_interior_size(Lf_out)), with the
//                         2-cell BC perimeter populated by the smoother.
//
// Designed so the caller can immediately consume the padded slab via
// restrict_meshvarbnd_from_padded_slab(perimeter=2) at the u-restrict site,
// skipping the otherwise-needed scatter of uf for restrict. uf stays current
// on owner (necessary for downstream apply+restrict, which reads MeshvarBnd
// uf), so the savings is 1 fine scatter per box per V-cycle.
//
// Returns false on the same fallback conditions as the gathering variant
// (sub_size<=1, parity/reach failures, ...), plus n_sweeps<=0 (because the
// outputs would be unpopulated and the caller would silently consume stale
// scratch).
bool smooth_pre_post_n_meshvarbnd_keep_slab_double(
        int box_owner,
        const MeshvarBnd<double>* uc,
        MeshvarBnd<double>* uf,
        const MeshvarBnd<double>* ff,
        double h, int n_sweeps,
        MPI_Comm sub_comm,
        ZoomSlabLayout& Lf_out,
        std::vector<double>& my_uf_int_out);
bool smooth_pre_post_n_meshvarbnd_keep_slab_float (
        int box_owner,
        const MeshvarBnd<float>*  uc,
        MeshvarBnd<float>*  uf,
        const MeshvarBnd<float>*  ff,
        float h, int n_sweeps,
        MPI_Comm sub_comm,
        ZoomSlabLayout& Lf_out,
        std::vector<float>&  my_uf_int_out);

inline bool smooth_pre_post_n_meshvarbnd_keep_slab(
        int box_owner,
        const MeshvarBnd<double>* uc,
        MeshvarBnd<double>* uf,
        const MeshvarBnd<double>* ff,
        double h, int n_sweeps,
        MPI_Comm sub_comm,
        ZoomSlabLayout& Lf_out,
        std::vector<double>& my_uf_int_out)
{ return smooth_pre_post_n_meshvarbnd_keep_slab_double(
        box_owner, uc, uf, ff, h, n_sweeps, sub_comm, Lf_out, my_uf_int_out); }
inline bool smooth_pre_post_n_meshvarbnd_keep_slab(
        int box_owner,
        const MeshvarBnd<float>* uc,
        MeshvarBnd<float>* uf,
        const MeshvarBnd<float>* ff,
        float h, int n_sweeps,
        MPI_Comm sub_comm,
        ZoomSlabLayout& Lf_out,
        std::vector<float>&  my_uf_int_out)
{ return smooth_pre_post_n_meshvarbnd_keep_slab_float (
        box_owner, uc, uf, ff, h, n_sweeps, sub_comm, Lf_out, my_uf_int_out); }

// ---------------------------------------------------------------------------
// Phase G.4: 2LPT FD source primitive on the z-slab.
//
// Implements the same operator as cosmology.cc::compute_2LPT_source over
// a single cluster z-slab. Stencil radii:
//   order=2 : radius 2 (halo_w must be >= 2)
//   order=4 : radius 4 (halo_w must be >= 4); order=4 and 6 share this code
//             (per cosmology.cc — order=4 is actually 8th-order accurate).
//
// f(i,j,k) = ( D00 D11 - D01^2 ) + ( D00 D22 - D02^2 ) + ( D11 D22 - D12^2 )
// where D_ab are 2nd derivatives of u with cell spacing h_grid.
//
// u_with_halo and f_with_halo use the local_with_halo layout. The z-halo
// must already be populated (caller exchanges via halo_exchange_z).
// f is written at i in [r, cnx-r), j in [r, cny-r), kl in [halo_w, halo_w+nz_r),
// where r = stencil radius (2 or 4). x/y cluster boundary cells are NOT
// written; nor are z-edge cells whose stencil reads into a zero-filled halo
// (those are written as if u extends with zeros outside, matching the
// non-periodic z-edge policy).
//
// Returns the per-rank L1 sum of f over the interior cells written (caller
// can MPI_Allreduce + divide by cluster volume to subtract the cluster mean
// for the multigrid Poisson source — that step is left to the caller so the
// primitive is composable and bit-identical-testable without reductions).
double lpt2_fd_z_slab_double(const ZoomSlabLayout& L,
                             const double* u_with_halo,
                             double h_grid,
                             unsigned order,
                             double* f_with_halo);
float  lpt2_fd_z_slab_float (const ZoomSlabLayout& L,
                             const float*  u_with_halo,
                             float  h_grid,
                             unsigned order,
                             float*  f_with_halo);

inline double lpt2_fd_z(const ZoomSlabLayout& L,
                        const double* u, double h, unsigned order, double* f)
{ return lpt2_fd_z_slab_double(L, u, h, order, f); }
inline float  lpt2_fd_z(const ZoomSlabLayout& L,
                        const float*  u, float  h, unsigned order, float*  f)
{ return lpt2_fd_z_slab_float (L, u, h, order, f); }

// Phase G.4 smoke test. Builds a single cluster with a deterministic u
// pattern, computes f_slab on each rank's z-slab, gathers, and compares
// cell-by-cell against a serial reference of the same FD operator (with
// out-of-bounds-as-zero policy mirroring the slab's edge halo). Returns
// true iff max|f_slab - f_serial| == 0 (or < tol for floats).
//
// Opt-in via setup.test_zoom_slab_lpt2=yes; order selected by
// setup.test_zoom_slab_lpt2_order (2 or 4, default 2);
// halo_w via setup.test_zoom_slab_halo (must be >= stencil radius).
bool smoke_test_lpt2_single_cluster(unsigned order = 2, int halo_w = 2);

// ---------------------------------------------------------------------------
// Standalone smoke test. Builds a single-cluster layout over MPI_COMM_WORLD
// with synthetic cluster geometry (cluster_nx = cluster_ny = 16,
// cluster_nz = chosen so even split + halo works), fills a deterministic
// pattern on sub-comm rank 0, scatters, exchanges halo, gathers, and
// verifies bit-identical roundtrip + correct halo cell content.
//
// Returns true on success, false on mismatch.
// Use opt-in setup flag "test_zoom_slab = yes" to invoke from main.
// ---------------------------------------------------------------------------
bool smoke_test_single_cluster(int halo_w = 2);

} // namespace zoom_slab
} // namespace MUSIC
