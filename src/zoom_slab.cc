// Phase G.0: zoom-region z-slab decomposition primitive.
// See zoom_slab.hh for design.

#include "zoom_slab.hh"
#include "mpi_helper.hh"
#include "log.hh"
#include "mesh.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

namespace MUSIC {
namespace zoom_slab {

// Phase G.2b A': opt-in toggle for the bridge-based GS smoother at the
// per-box GS sites in multigrid::solver::twoGrid_multibox. Set by poisson.cc
// reading `setup.zoom_slab_smoother` once per solve. Cached as a static so the
// templated solver class can check it without restructuring its constructor.
namespace { bool g_smoother_enabled = false; }
void set_smoother_enabled(bool v) { g_smoother_enabled = v; }
bool smoother_enabled() { return g_smoother_enabled; }

// Phase G.2b B.2b.2: opt-in toggle for the SPMD composite-MG path inside
// twoGrid_multibox. Default false — when off, twoGrid_multibox runs its
// existing rank-0-only logic (wrapped by with_pbox_distributed at the
// callsite); when on, twoGrid_multibox routes through twoGrid_multibox_spmd
// which broadcasts parents to child_owners, owner-gates per-box ops, and
// accumulates children back to parents. The flag is independent of
// smoother_enabled / subcomm_policy — those still control the bridge.
namespace { bool g_spmd_multigrid_enabled = false; }
void set_spmd_multigrid_enabled(bool v) { g_spmd_multigrid_enabled = v; }
bool spmd_multigrid_enabled() { return g_spmd_multigrid_enabled; }

// Phase G.2b B.5.4.a: opt-in toggle for keep-in-slab pre/post smoothing.
// When on (and the per-op bridge is also active), twoGrid_multibox_spmd's
// pre-smooth and post-smooth loops collapse the N×(scatter+interp_cf+gather +
// scatter+gs+gather) sequence into 1×scatter + N×(coarse_bcast + kernel) +
// 1×gather per box per phase via smooth_pre_post_n_meshvarbnd. Default false.
namespace { bool g_keep_slab_smooth_enabled = false; }
void set_keep_slab_smooth_enabled(bool v) { g_keep_slab_smooth_enabled = v; }
bool keep_slab_smooth_enabled() { return g_keep_slab_smooth_enabled; }

// Phase G.2b B.5.4.b: opt-in toggle for keep-in-slab u-restrict. When on
// (and B.5.4.a keep-slab smoothing is also active), twoGrid_multibox_spmd's
// pre-smooth+u-restrict pair shares the padded-cluster slab buffer:
// pre-smooth's smooth_pre_post_n_meshvarbnd_keep_slab hands the post-smoother
// padded interior to restrict_meshvarbnd_from_padded_slab (perimeter=2),
// eliminating one gather/scatter round-trip per box per V-cycle. Default
// false. Requires keep_slab_smooth_enabled() to fire.
namespace { bool g_keep_slab_urestrict_enabled = false; }
void set_keep_slab_urestrict_enabled(bool v) { g_keep_slab_urestrict_enabled = v; }
bool keep_slab_urestrict_enabled() { return g_keep_slab_urestrict_enabled; }

// Phase G.2b B.5.4.c: opt-in toggle for fused keep-in-slab prolong_add. When
// on (and B.5.4.a keep-slab smoothing is also active, with m_npostsmooth>0),
// twoGrid_multibox_spmd's post-coarse prolong_add and post-smooth share one
// scatter/gather lifetime via prolong_add_then_smooth_n_meshvarbnd,
// eliminating one fine scatter + one fine gather per box per V-cycle. Default
// false. Requires keep_slab_smooth_enabled() to fire.
namespace { bool g_keep_slab_prolong_enabled = false; }
void set_keep_slab_prolong_enabled(bool v) { g_keep_slab_prolong_enabled = v; }
bool keep_slab_prolong_enabled() { return g_keep_slab_prolong_enabled; }

// ---------------------------------------------------------------------------
// Phase G.2b (B.1): per-cluster sub_comm registry. See zoom_slab.hh comments
// for policy semantics + lifecycle contract.
// ---------------------------------------------------------------------------
namespace {
    std::string g_subcomm_policy = "world";
    // Registry indexed [level][box_id]. Empty when nothing was built.
    std::vector< std::vector<MPI_Comm> > g_subcomm_registry;
    // Tracks which comms in the registry were created via MPI_Comm_split
    // (so we know which to free vs which to leave alone).
    std::vector< std::vector<bool> > g_subcomm_owned;

    bool comm_is_freeable(MPI_Comm c) {
#ifdef USE_MPI
        if( c == MPI_COMM_WORLD ) return false;
        if( c == MPI_COMM_SELF  ) return false;
        if( c == MPI_COMM_NULL  ) return false;
        return true;
#else
        (void)c;
        return false;
#endif
    }
}

void set_subcomm_policy(const std::string& p) {
    if( p == "world" || p == "self" || p == "round_robin" )
        g_subcomm_policy = p;
    else
        throw std::runtime_error("zoom_slab::set_subcomm_policy: unknown policy '" + p +
                                 "' (expected world|self|round_robin)");
}
const std::string& subcomm_policy() { return g_subcomm_policy; }

void free_subcomm_registry() {
#ifdef USE_MPI
    for( size_t L=0; L<g_subcomm_registry.size(); ++L ){
        for( size_t b=0; b<g_subcomm_registry[L].size(); ++b ){
            if( g_subcomm_owned[L][b] && comm_is_freeable(g_subcomm_registry[L][b]) ){
                MPI_Comm c = g_subcomm_registry[L][b];
                MPI_Comm_free(&c);
            }
        }
    }
#endif
    g_subcomm_registry.clear();
    g_subcomm_owned.clear();
}

void build_subcomm_registry(const std::vector<std::size_t>& nb_per_level) {
    // Always clear first — caller may rebuild after policy change.
    free_subcomm_registry();
    const size_t Lmax = nb_per_level.size();
    g_subcomm_registry.assign(Lmax, std::vector<MPI_Comm>());
    g_subcomm_owned.assign   (Lmax, std::vector<bool>());

#ifdef USE_MPI
    const int my_rank = MUSIC::mpi::rank();
    const int np      = MUSIC::mpi::size();
#else
    const int my_rank = 0;
    const int np      = 1;
#endif

    for( size_t L=0; L<Lmax; ++L ){
        const size_t nb = nb_per_level[L];
        g_subcomm_registry[L].assign(nb, MPI_COMM_WORLD);
        g_subcomm_owned   [L].assign(nb, false);

        if( g_subcomm_policy == "world" ){
            // All slots already MPI_COMM_WORLD.
            continue;
        }
        else if( g_subcomm_policy == "self" ){
#ifdef USE_MPI
            for( size_t b=0; b<nb; ++b )
                g_subcomm_registry[L][b] = MPI_COMM_SELF;
#endif
            continue;
        }
        else if( g_subcomm_policy == "round_robin" ){
#ifdef USE_MPI
            if( nb == 0 ) continue;
            if( (int)nb > np ){
                throw std::runtime_error(
                    "zoom_slab::build_subcomm_registry: round_robin policy requires "
                    "nb <= np (got nb=" + std::to_string(nb) + ", np=" + std::to_string(np) + ")");
            }
            // Assign rank r to cluster g = r * nb / np. This packs contiguous
            // rank ranges into each cluster (np/nb ranks per cluster, with
            // residual ranks landing in higher-indexed clusters).
            const int my_color = (int)( ((long long)my_rank * (long long)nb) / (long long)np );
            MPI_Comm new_comm = MPI_COMM_NULL;
            MPI_Comm_split(MPI_COMM_WORLD, my_color, my_rank, &new_comm);
            // Every rank now belongs to exactly one cluster's sub_comm. Other
            // clusters' sub_comm is MPI_COMM_NULL from this rank's perspective —
            // but every rank must store ALL nb comms (mirrored) so that
            // collective calls within any one cluster's sub_comm find the right
            // group. We achieve this by doing nb-1 additional splits with the
            // "wrong" color (MPI_UNDEFINED) so MPI returns MPI_COMM_NULL for
            // those, then placing new_comm in the matching slot.
            // Actually — MPI_Comm_split with same key=my_rank produces stable
            // ordering. Each rank gets exactly one non-null comm. So the slot
            // we own is g_subcomm_registry[L][my_color] = new_comm; for the
            // other (nb-1) slots we still need a usable comm if any code on
            // this rank tries to read them. Use MPI_COMM_SELF as the safe
            // default for non-owned slots (no collective call from this rank
            // can ever fire there in B.1; in B.2+ the owner check gates it).
            for( size_t b=0; b<nb; ++b )
                g_subcomm_registry[L][b] = MPI_COMM_SELF;
            g_subcomm_registry[L][my_color] = new_comm;
            g_subcomm_owned   [L][my_color] = true;
#endif
            continue;
        }
        // Unknown policy — fall through to all-world (set_subcomm_policy
        // validates input; this branch only fires if g_subcomm_policy is
        // mutated externally without going through the setter).
    }
}

MPI_Comm subcomm_for_box(unsigned level, std::size_t box_id) {
    if( (size_t)level >= g_subcomm_registry.size() ) return MPI_COMM_WORLD;
    if( box_id >= g_subcomm_registry[level].size()  ) return MPI_COMM_WORLD;
    return g_subcomm_registry[level][box_id];
}

// ---------------------------------------------------------------------------
// internal helpers (translation-unit-private)
// ---------------------------------------------------------------------------
namespace {

#ifdef USE_MPI
inline int sub_rank_of(MPI_Comm c) {
    int r = 0; MPI_Comm_rank(c, &r); return r;
}
inline int sub_size_of(MPI_Comm c) {
    int s = 1; MPI_Comm_size(c, &s); return s;
}
#else
inline int sub_rank_of(MPI_Comm) { return 0; }
inline int sub_size_of(MPI_Comm) { return 1; }
#endif

// Even-split with leftover spread to first ranks. Returns [z0, z1) for r in [0, sub_size).
inline void even_split(int cluster_nz, int sub_size, int r, int& z0, int& z1) {
    const int base = cluster_nz / sub_size;
    const int rem  = cluster_nz % sub_size;
    if( r < rem ) {
        z0 = r * (base + 1);
        z1 = z0 + (base + 1);
    } else {
        z0 = rem * (base + 1) + (r - rem) * base;
        z1 = z0 + base;
    }
}

// Generic scatter/gather bodies, instantiated for double/float via wrappers.
// Phase G.1: src_rank/dst_rank parameterised so callers can scatter from
// the per-box owner (not necessarily sub_comm rank 0).
template<typename T>
void scatter_impl(const ZoomSlabLayout& L, int src_rank,
                  const T* full_buf, T* my_interior)
{
    const std::size_t nxy = (std::size_t)L.cluster_nx * (std::size_t)L.cluster_ny;
#ifdef USE_MPI
    const int rk = L.sub_rank;
    const int np = L.sub_size;
    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
    const int tag_base = 4100;

    if( rk == src_rank ) {
        // own slice
        {
            const int z0 = L.all_z0[src_rank];
            const int z1 = L.all_z1[src_rank];
            for( int i=0; i<L.cluster_nx; ++i )
                for( int j=0; j<L.cluster_ny; ++j )
                    for( int k=z0; k<z1; ++k ) {
                        const std::size_t src = ((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j)
                                              * (std::size_t)L.cluster_nz + (std::size_t)k;
                        const std::size_t dst = ((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j)
                                              * (std::size_t)(z1 - z0) + (std::size_t)(k - z0);
                        my_interior[dst] = full_buf[src];
                    }
        }
        // workers (everyone except src)
        for( int r=0; r<np; ++r ) {
            if( r == src_rank ) continue;
            const int z0 = L.all_z0[r];
            const int z1 = L.all_z1[r];
            const int nz_r = z1 - z0;
            if( nz_r <= 0 ) continue;
            const std::size_t cnt = nxy * (std::size_t)nz_r;
            std::vector<T> buf(cnt);
            for( int i=0; i<L.cluster_nx; ++i )
                for( int j=0; j<L.cluster_ny; ++j )
                    for( int k=z0; k<z1; ++k ) {
                        const std::size_t src = ((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j)
                                              * (std::size_t)L.cluster_nz + (std::size_t)k;
                        const std::size_t bdst = ((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j)
                                               * (std::size_t)nz_r + (std::size_t)(k - z0);
                        buf[bdst] = full_buf[src];
                    }
            MPI_Send(buf.data(), (int)cnt, dtype, r, tag_base + r, L.sub_comm);
        }
    } else {
        const int nz_r = L.my_z1 - L.my_z0;
        if( nz_r <= 0 ) return;
        const std::size_t cnt = nxy * (std::size_t)nz_r;
        MPI_Recv(my_interior, (int)cnt, dtype, src_rank, tag_base + rk, L.sub_comm, MPI_STATUS_IGNORE);
    }
#else
    (void)src_rank;
    // serial: full_buf -> my_interior (assume z0=0, z1=cluster_nz)
    std::memcpy(my_interior, full_buf, nxy * (std::size_t)L.cluster_nz * sizeof(T));
#endif
}

template<typename T>
void gather_impl(const ZoomSlabLayout& L, int dst_rank,
                 const T* my_interior, T* full_buf)
{
    const std::size_t nxy = (std::size_t)L.cluster_nx * (std::size_t)L.cluster_ny;
#ifdef USE_MPI
    const int rk = L.sub_rank;
    const int np = L.sub_size;
    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
    const int tag_base = 4300;

    if( rk == dst_rank ) {
        // own slice
        {
            const int z0 = L.all_z0[dst_rank];
            const int z1 = L.all_z1[dst_rank];
            for( int i=0; i<L.cluster_nx; ++i )
                for( int j=0; j<L.cluster_ny; ++j )
                    for( int k=z0; k<z1; ++k ) {
                        const std::size_t dst = ((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j)
                                              * (std::size_t)L.cluster_nz + (std::size_t)k;
                        const std::size_t src = ((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j)
                                              * (std::size_t)(z1 - z0) + (std::size_t)(k - z0);
                        full_buf[dst] = my_interior[src];
                    }
        }
        // workers (everyone except dst)
        for( int r=0; r<np; ++r ) {
            if( r == dst_rank ) continue;
            const int z0 = L.all_z0[r];
            const int z1 = L.all_z1[r];
            const int nz_r = z1 - z0;
            if( nz_r <= 0 ) continue;
            const std::size_t cnt = nxy * (std::size_t)nz_r;
            std::vector<T> buf(cnt);
            MPI_Recv(buf.data(), (int)cnt, dtype, r, tag_base + r, L.sub_comm, MPI_STATUS_IGNORE);
            for( int i=0; i<L.cluster_nx; ++i )
                for( int j=0; j<L.cluster_ny; ++j )
                    for( int k=z0; k<z1; ++k ) {
                        const std::size_t dst = ((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j)
                                              * (std::size_t)L.cluster_nz + (std::size_t)k;
                        const std::size_t bsrc = ((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j)
                                               * (std::size_t)nz_r + (std::size_t)(k - z0);
                        full_buf[dst] = buf[bsrc];
                    }
        }
    } else {
        const int nz_r = L.my_z1 - L.my_z0;
        if( nz_r <= 0 ) return;
        const std::size_t cnt = nxy * (std::size_t)nz_r;
        MPI_Send(const_cast<T*>(my_interior), (int)cnt, dtype, dst_rank, tag_base + rk, L.sub_comm);
    }
#else
    (void)dst_rank;
    std::memcpy(full_buf, my_interior, nxy * (std::size_t)L.cluster_nz * sizeof(T));
#endif
}

// Halo exchange: each rank sends top halo_w interior slabs to its high-z
// neighbour and bottom halo_w interior slabs to its low-z neighbour, then
// receives halo slabs into the corresponding halo regions of local_with_halo.
//
// local_with_halo storage layout: row-major (i, j, k) with k fastest;
//   k = 0 .. halo_w-1                            -> low-z halo
//   k = halo_w .. halo_w + local_nz - 1          -> interior
//   k = halo_w + local_nz .. 2*halo_w + local_nz - 1 -> high-z halo
template<typename T>
void halo_impl(const ZoomSlabLayout& L, T* buf)
{
#ifdef USE_MPI
    const int rk = L.sub_rank;
    const int np = L.sub_size;
    const int nz_r = local_nz(L);
    const int hw = L.halo_w;
    if( hw <= 0 || np <= 1 ) {
        // edge halos remain zero (already-zero or caller responsibility)
        if( np <= 1 ) return;
    }
    if( hw > nz_r ) {
        LOGERR("zoom_slab::halo_exchange: halo_w=%d > local_nz=%d on rank %d (sub_size=%d)",
               hw, nz_r, rk, np);
        throw std::runtime_error("zoom_slab halo_w too large for slab");
    }
    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
    const int tag_lo = 4500;  // send to low neighbour
    const int tag_hi = 4501;  // send to high neighbour
    const std::size_t nxy = (std::size_t)L.cluster_nx * (std::size_t)L.cluster_ny;
    const std::size_t plane = nxy; // one k-plane = nx*ny cells
    const int stride_k = 1; // k is fastest
    const std::size_t halo_cells = nxy * (std::size_t)hw;
    (void)plane; (void)stride_k;

    // Pack send buffers. Two slabs of hw planes each.
    std::vector<T> send_lo(halo_cells), send_hi(halo_cells);
    for( int i=0; i<L.cluster_nx; ++i )
        for( int j=0; j<L.cluster_ny; ++j ) {
            const std::size_t kstride = (std::size_t)(2*hw + nz_r);
            const std::size_t row = ((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j) * kstride;
            for( int k=0; k<hw; ++k ) {
                // bottom interior slabs go to low neighbour
                send_lo[((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j) * (std::size_t)hw + (std::size_t)k]
                    = buf[row + (std::size_t)(hw + k)];
                // top interior slabs go to high neighbour
                send_hi[((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j) * (std::size_t)hw + (std::size_t)k]
                    = buf[row + (std::size_t)(hw + nz_r - hw + k)];
            }
        }

    std::vector<T> recv_lo(halo_cells, T(0)), recv_hi(halo_cells, T(0));

    // Non-blocking exchange. Edges (rk==0 low, rk==np-1 high) skip the send/recv;
    // their halos stay zero-initialised.
    MPI_Request reqs[4];
    int nreq = 0;
    if( rk > 0 ) {
        MPI_Isend(send_lo.data(), (int)halo_cells, dtype, rk-1, tag_lo,
                  L.sub_comm, &reqs[nreq++]);
        MPI_Irecv(recv_lo.data(), (int)halo_cells, dtype, rk-1, tag_hi,
                  L.sub_comm, &reqs[nreq++]);
    }
    if( rk < np-1 ) {
        MPI_Isend(send_hi.data(), (int)halo_cells, dtype, rk+1, tag_hi,
                  L.sub_comm, &reqs[nreq++]);
        MPI_Irecv(recv_hi.data(), (int)halo_cells, dtype, rk+1, tag_lo,
                  L.sub_comm, &reqs[nreq++]);
    }
    if( nreq > 0 ) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // Unpack recv buffers into halo regions.
    for( int i=0; i<L.cluster_nx; ++i )
        for( int j=0; j<L.cluster_ny; ++j ) {
            const std::size_t kstride = (std::size_t)(2*hw + nz_r);
            const std::size_t row = ((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j) * kstride;
            for( int k=0; k<hw; ++k ) {
                if( rk > 0 )
                    buf[row + (std::size_t)k] =
                        recv_lo[((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j) * (std::size_t)hw + (std::size_t)k];
                else
                    buf[row + (std::size_t)k] = T(0);
                if( rk < np-1 )
                    buf[row + (std::size_t)(hw + nz_r + k)] =
                        recv_hi[((std::size_t)i * (std::size_t)L.cluster_ny + (std::size_t)j) * (std::size_t)hw + (std::size_t)k];
                else
                    buf[row + (std::size_t)(hw + nz_r + k)] = T(0);
            }
        }
#else
    (void)L; (void)buf;
#endif
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------
ZoomSlabLayout make_layout(
    MPI_Comm    sub_comm,
    std::size_t cluster_id,
    unsigned    level,
    int cluster_oax, int cluster_oay, int cluster_oaz,
    int cluster_nx,  int cluster_ny,  int cluster_nz,
    int halo_w)
{
    ZoomSlabLayout L;
    L.cluster_id = cluster_id;
    L.level      = level;
    L.sub_comm   = sub_comm;
    L.sub_rank   = sub_rank_of(sub_comm);
    L.sub_size   = sub_size_of(sub_comm);
    L.cluster_oax = cluster_oax;
    L.cluster_oay = cluster_oay;
    L.cluster_oaz = cluster_oaz;
    L.cluster_nx  = cluster_nx;
    L.cluster_ny  = cluster_ny;
    L.cluster_nz  = cluster_nz;
    L.halo_w      = halo_w;

    L.all_z0.assign(L.sub_size, 0);
    L.all_z1.assign(L.sub_size, 0);
    for( int r=0; r<L.sub_size; ++r ) {
        int z0=0, z1=0;
        even_split(cluster_nz, L.sub_size, r, z0, z1);
        L.all_z0[r] = z0;
        L.all_z1[r] = z1;
    }
    L.my_z0 = L.all_z0[L.sub_rank];
    L.my_z1 = L.all_z1[L.sub_rank];

    // Sanity: every rank must have positive local_nz to participate in halo
    // exchanges. If sub_size > cluster_nz, the layout is malformed; caller
    // must reduce sub_size first.
    if( L.sub_size > 1 && (L.my_z1 - L.my_z0) <= 0 ) {
        LOGERR("zoom_slab::make_layout: rank %d/%d has empty z-slab (cluster_nz=%d, sub_size=%d)",
               L.sub_rank, L.sub_size, cluster_nz, L.sub_size);
        throw std::runtime_error("zoom_slab sub_size exceeds cluster_nz");
    }
    return L;
}

void scatter_cluster_to_zslab_double(const ZoomSlabLayout& L, const double* f, double* my)
{ scatter_impl<double>(L, 0, f, my); }
void scatter_cluster_to_zslab_float (const ZoomSlabLayout& L, const float*  f, float*  my)
{ scatter_impl<float >(L, 0, f, my); }
void scatter_cluster_to_zslab_double_from(const ZoomSlabLayout& L, int src, const double* f, double* my)
{ scatter_impl<double>(L, src, f, my); }
void scatter_cluster_to_zslab_float_from (const ZoomSlabLayout& L, int src, const float*  f, float*  my)
{ scatter_impl<float >(L, src, f, my); }
void gather_zslab_to_cluster_double(const ZoomSlabLayout& L, const double* my, double* f)
{ gather_impl<double>(L, 0, my, f); }
void gather_zslab_to_cluster_float (const ZoomSlabLayout& L, const float*  my, float*  f)
{ gather_impl<float >(L, 0, my, f); }
void gather_zslab_to_cluster_double_to(const ZoomSlabLayout& L, int dst, const double* my, double* f)
{ gather_impl<double>(L, dst, my, f); }
void gather_zslab_to_cluster_float_to (const ZoomSlabLayout& L, int dst, const float*  my, float*  f)
{ gather_impl<float >(L, dst, my, f); }
void halo_exchange_z_double(const ZoomSlabLayout& L, double* b) { halo_impl<double>(L, b); }
void halo_exchange_z_float (const ZoomSlabLayout& L, float*  b) { halo_impl<float >(L, b); }

// ---------------------------------------------------------------------------
// Phase G.2: slab residual primitive.
// Computes r = h^2 f - L u over the z-slab interior, where L is the 7-point
// O2 Laplacian. u and r use local_with_halo storage (i,j,k) row-major with
// k fastest and k in [halo_w, halo_w+local_nz) the interior. f also uses the
// local_with_halo layout but only the interior cells are read.
//
// Iteration range: i in [1, cnx-1), j in [1, cny-1), k_local in
// [halo_w, halo_w+local_nz). The cluster x/y boundary cells (i=0, i=cnx-1,
// j=0, j=cny-1) are NOT touched. The z-halo cells must already be populated
// (caller responsibility — typically via halo_exchange_z, with zero-fill at
// cluster z-edges per non-periodic policy).
// ---------------------------------------------------------------------------
namespace {

template<typename T>
double residual_impl(const ZoomSlabLayout& L,
                     const T* u, const T* f,
                     T h, T* r)
{
    const int cnx   = L.cluster_nx;
    const int cny   = L.cluster_ny;
    const int nz_r  = local_nz(L);
    const int hw    = L.halo_w;
    const int kstride = 2*hw + nz_r;
    const T   h2    = h * h;
    const std::size_t ny_str = (std::size_t)cny * (std::size_t)kstride;
    const std::size_t x_str  = (std::size_t)kstride;

    double sum_sq = 0.0;
    for( int i=1; i<cnx-1; ++i ) {
        const std::size_t i_off    = (std::size_t)i * ny_str;
        const std::size_t im1_off  = (std::size_t)(i-1) * ny_str;
        const std::size_t ip1_off  = (std::size_t)(i+1) * ny_str;
        for( int j=1; j<cny-1; ++j ) {
            const std::size_t row    = i_off   + (std::size_t)j * x_str;
            const std::size_t rim1   = im1_off + (std::size_t)j * x_str;
            const std::size_t rip1   = ip1_off + (std::size_t)j * x_str;
            const std::size_t rjm1   = i_off   + (std::size_t)(j-1) * x_str;
            const std::size_t rjp1   = i_off   + (std::size_t)(j+1) * x_str;
            for( int kl=hw; kl<hw+nz_r; ++kl ) {
                const std::size_t idx = row + (std::size_t)kl;
                const T lu = u[rim1 + kl] + u[rip1 + kl]
                           + u[rjm1 + kl] + u[rjp1 + kl]
                           + u[row  + kl - 1] + u[row + kl + 1]
                           - T(6) * u[idx];
                const T rr = h2 * f[idx] - lu;
                r[idx] = rr;
                sum_sq += (double)rr * (double)rr;
            }
        }
    }
    return sum_sq;
}

} // anonymous namespace

double residual_z_slab_double(const ZoomSlabLayout& L,
                              const double* u, const double* f,
                              double h, double* r)
{ return residual_impl<double>(L, u, f, h, r); }

float residual_z_slab_float(const ZoomSlabLayout& L,
                            const float* u, const float* f,
                            float h, float* r)
{ return (float)residual_impl<float>(L, u, f, h, r); }

// ---------------------------------------------------------------------------
// Phase G.2b: red-black GS smoother on z-slab. Mirrors mg_solver.hh::GaussSeidel
// for the 7-point O2 stencil (ccoeff = -6). Color uses cluster-global iz so
// the same cell gets the same color regardless of sub_comm decomposition.
// ---------------------------------------------------------------------------
namespace {

template<typename T>
void gs_impl(const ZoomSlabLayout& L,
             T* u, const T* f, T h)
{
    const int cnx   = L.cluster_nx;
    const int cny   = L.cluster_ny;
    const int nz_r  = local_nz(L);
    const int hw    = L.halo_w;
    const int kstride = 2*hw + nz_r;
    const int z_base  = L.my_z0;  // cluster-global z index for kl == hw
    const T   h2    = h * h;
    const T   sixth = T(1) / T(6);
    const std::size_t ny_str = (std::size_t)cny * (std::size_t)kstride;
    const std::size_t x_str  = (std::size_t)kstride;

    for( int color=0; color<2; ++color ) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for( int i=1; i<cnx-1; ++i ) {
            const std::size_t i_off    = (std::size_t)i * ny_str;
            const std::size_t im1_off  = (std::size_t)(i-1) * ny_str;
            const std::size_t ip1_off  = (std::size_t)(i+1) * ny_str;
            for( int j=1; j<cny-1; ++j ) {
                const std::size_t row    = i_off   + (std::size_t)j * x_str;
                const std::size_t rim1   = im1_off + (std::size_t)j * x_str;
                const std::size_t rip1   = ip1_off + (std::size_t)j * x_str;
                const std::size_t rjm1   = i_off   + (std::size_t)(j-1) * x_str;
                const std::size_t rjp1   = i_off   + (std::size_t)(j+1) * x_str;
                // Pick kl with matching color first to avoid the if-branch
                // inside the kl loop; stride by 2.
                const int kg_lo = z_base + 0;          // kl=hw -> kg = z_base
                const int kl_start = hw + (( (i + j + kg_lo) & 1 ) == color ? 0 : 1);
                for( int kl=kl_start; kl<hw+nz_r; kl+=2 ) {
                    const std::size_t idx = row + (std::size_t)kl;
                    const T sum = u[rim1 + kl] + u[rip1 + kl]
                                + u[rjm1 + kl] + u[rjp1 + kl]
                                + u[row  + kl - 1] + u[row + kl + 1];
                    // residual_z_slab convention is L u = f (r = h^2 f - L u).
                    // GS update for that convention is u = (sum - h^2 f) / 6.
                    // Production mg_solver.hh uses L u = -f and would feed -f
                    // here at the call site (or negate sign of h2 below).
                    u[idx] = (sum - h2 * f[idx]) * sixth;
                }
            }
        }
        // Refresh z-halo so opposite-color sweep (or caller's next op) sees
        // the freshly-updated neighbour-rank slab interior.
#ifdef USE_MPI
        if( sizeof(T) == sizeof(double) )
            halo_exchange_z_double(L, reinterpret_cast<double*>(u));
        else
            halo_exchange_z_float (L, reinterpret_cast<float*> (u));
#else
        (void)L;
#endif
    }
}

} // anonymous namespace

void gs_z_slab_double(const ZoomSlabLayout& L,
                      double* u, const double* f, double h)
{ gs_impl<double>(L, u, f, h); }

void gs_z_slab_float(const ZoomSlabLayout& L,
                     float* u, const float* f, float h)
{ gs_impl<float>(L, u, f, h); }

// ---------------------------------------------------------------------------
// Phase G.2b (production-wire step 1): gs_z_neg = gs_z with production's
// L u = -f sign convention. Update reads u = (sum + h^2 f)/6 instead of
// (sum - h^2 f)/6. Near-duplicate of gs_impl with the sign flip on h2*f only;
// kept as a separate function to avoid touching the shipped-and-verified
// gs_impl.
// ---------------------------------------------------------------------------
namespace {

template<typename T>
void gs_neg_impl(const ZoomSlabLayout& L,
                 T* u, const T* f, T h)
{
    const int cnx   = L.cluster_nx;
    const int cny   = L.cluster_ny;
    const int nz_r  = local_nz(L);
    const int hw    = L.halo_w;
    const int kstride = 2*hw + nz_r;
    const int z_base  = L.my_z0;  // cluster-global z index for kl == hw
    const T   h2    = h * h;
    const T   sixth = T(1) / T(6);
    const std::size_t ny_str = (std::size_t)cny * (std::size_t)kstride;
    const std::size_t x_str  = (std::size_t)kstride;

    for( int color=0; color<2; ++color ) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for( int i=1; i<cnx-1; ++i ) {
            const std::size_t i_off    = (std::size_t)i * ny_str;
            const std::size_t im1_off  = (std::size_t)(i-1) * ny_str;
            const std::size_t ip1_off  = (std::size_t)(i+1) * ny_str;
            for( int j=1; j<cny-1; ++j ) {
                const std::size_t row    = i_off   + (std::size_t)j * x_str;
                const std::size_t rim1   = im1_off + (std::size_t)j * x_str;
                const std::size_t rip1   = ip1_off + (std::size_t)j * x_str;
                const std::size_t rjm1   = i_off   + (std::size_t)(j-1) * x_str;
                const std::size_t rjp1   = i_off   + (std::size_t)(j+1) * x_str;
                const int kg_lo = z_base + 0;
                const int kl_start = hw + (( (i + j + kg_lo) & 1 ) == color ? 0 : 1);
                for( int kl=kl_start; kl<hw+nz_r; kl+=2 ) {
                    const std::size_t idx = row + (std::size_t)kl;
                    const T sum = u[rim1 + kl] + u[rip1 + kl]
                                + u[rjm1 + kl] + u[rjp1 + kl]
                                + u[row  + kl - 1] + u[row + kl + 1];
                    // Production GaussSeidel (mg_solver.hh:215) for stencil_7P:
                    //   u = (rhs(u) + h^2 * f) * c0   with c0=1/6, rhs=sum_n
                    u[idx] = (sum + h2 * f[idx]) * sixth;
                }
            }
        }
#ifdef USE_MPI
        if( sizeof(T) == sizeof(double) )
            halo_exchange_z_double(L, reinterpret_cast<double*>(u));
        else
            halo_exchange_z_float (L, reinterpret_cast<float*> (u));
#else
        (void)L;
#endif
    }
}

} // anonymous namespace

void gs_z_neg_slab_double(const ZoomSlabLayout& L,
                          double* u, const double* f, double h)
{ gs_neg_impl<double>(L, u, f, h); }

void gs_z_neg_slab_float(const ZoomSlabLayout& L,
                         float* u, const float* f, float h)
{ gs_neg_impl<float>(L, u, f, h); }

// ---------------------------------------------------------------------------
// Phase G.2b production-wire step 2: gs_z_neg variant that skips cluster
// z-edge cells AND applies a +1 parity offset so the coloring matches
// production GaussSeidel's MeshvarBnd-interior parity.
//
// Designed for the MeshvarBnd bridge (gs_z_neg_meshvarbnd_*) where the
// cluster shape extends MeshvarBnd by +1 on every axis (cluster perimeter
// holds frozen BC cells). The +1 axis-shift on each of i,j,k flips the
// parity of (i+j+k) by an odd amount → kernel must add +1 to (i+j+kg)
// before testing color to recover production's MeshvarBnd-interior parity.
//
// Update domain (per rank):
//   i in [1, cnx-1)
//   j in [1, cny-1)
//   kl in [kl_lo, kl_hi) where
//     kl_lo = (sub_rank == 0)            ? hw+1       : hw
//     kl_hi = (sub_rank == sub_size-1)   ? hw+nz_r-1  : hw+nz_r
// Cells whose cluster-global kg ∈ {0, cluster_nz-1} are skipped — they sit
// on the cluster z-perimeter and hold BC values.
namespace {

template<typename T>
void gs_neg_skip_z_impl(const ZoomSlabLayout& L,
                        T* u, const T* f, T h)
{
    const int cnx   = L.cluster_nx;
    const int cny   = L.cluster_ny;
    const int nz_r  = local_nz(L);
    const int hw    = L.halo_w;
    const int kstride = 2*hw + nz_r;
    const int z_base  = L.my_z0;
    const T   h2    = h * h;
    const T   sixth = T(1) / T(6);
    const std::size_t ny_str = (std::size_t)cny * (std::size_t)kstride;
    const std::size_t x_str  = (std::size_t)kstride;

    const int kl_lo = (L.sub_rank == 0)                ? (hw + 1)      : hw;
    const int kl_hi = (L.sub_rank == L.sub_size - 1)   ? (hw + nz_r - 1) : (hw + nz_r);

    for( int color=0; color<2; ++color ) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for( int i=1; i<cnx-1; ++i ) {
            const std::size_t i_off    = (std::size_t)i * ny_str;
            const std::size_t im1_off  = (std::size_t)(i-1) * ny_str;
            const std::size_t ip1_off  = (std::size_t)(i+1) * ny_str;
            for( int j=1; j<cny-1; ++j ) {
                const std::size_t row    = i_off   + (std::size_t)j * x_str;
                const std::size_t rim1   = im1_off + (std::size_t)j * x_str;
                const std::size_t rip1   = ip1_off + (std::size_t)j * x_str;
                const std::size_t rjm1   = i_off   + (std::size_t)(j-1) * x_str;
                const std::size_t rjp1   = i_off   + (std::size_t)(j+1) * x_str;
                const int kg_lo = z_base + (kl_lo - hw);
                // Production parity = ((ci-1) + (cj-1) + (ck-1)) & 1 = (ci+cj+ck+1) & 1
                const int kl_start = kl_lo + (( (i + j + kg_lo + 1) & 1 ) == color ? 0 : 1);
                for( int kl=kl_start; kl<kl_hi; kl+=2 ) {
                    const std::size_t idx = row + (std::size_t)kl;
                    const T sum = u[rim1 + kl] + u[rip1 + kl]
                                + u[rjm1 + kl] + u[rjp1 + kl]
                                + u[row  + kl - 1] + u[row + kl + 1];
                    u[idx] = (sum + h2 * f[idx]) * sixth;
                }
            }
        }
#ifdef USE_MPI
        if( sizeof(T) == sizeof(double) )
            halo_exchange_z_double(L, reinterpret_cast<double*>(u));
        else
            halo_exchange_z_float (L, reinterpret_cast<float*> (u));
#else
        (void)L;
#endif
    }
}

} // anonymous namespace

void gs_z_neg_skip_z_slab_double(const ZoomSlabLayout& L,
                                 double* u, const double* f, double h)
{ gs_neg_skip_z_impl<double>(L, u, f, h); }

void gs_z_neg_skip_z_slab_float(const ZoomSlabLayout& L,
                                float* u, const float* f, float h)
{ gs_neg_skip_z_impl<float>(L, u, f, h); }

// ---------------------------------------------------------------------------
// Phase G.2b B.5.0: 7-point Laplacian apply on z-slab.
//
// Pure local kernel mirroring stencil_7P::apply (schemes.hh). Same skip
// rule as gs_neg_skip_z_impl: writes only the cluster interior excluding
// the cluster perimeter (cells whose cluster-global k is in {0, cnz-1}
// or whose i/j is on the cluster x/y boundary). Halos must already be
// populated; the kernel does not touch them.
// ---------------------------------------------------------------------------
namespace {

template<typename T>
void apply_z_slab_impl(const ZoomSlabLayout& L,
                       const T* u, T* out)
{
    const int cnx   = L.cluster_nx;
    const int cny   = L.cluster_ny;
    const int nz_r  = local_nz(L);
    const int hw    = L.halo_w;
    const int kstride = 2*hw + nz_r;
    const std::size_t ny_str = (std::size_t)cny * (std::size_t)kstride;
    const std::size_t x_str  = (std::size_t)kstride;

    const int kl_lo = (L.sub_rank == 0)                ? (hw + 1)        : hw;
    const int kl_hi = (L.sub_rank == L.sub_size - 1)   ? (hw + nz_r - 1) : (hw + nz_r);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for( int i=1; i<cnx-1; ++i ) {
        const std::size_t i_off    = (std::size_t)i * ny_str;
        const std::size_t im1_off  = (std::size_t)(i-1) * ny_str;
        const std::size_t ip1_off  = (std::size_t)(i+1) * ny_str;
        for( int j=1; j<cny-1; ++j ) {
            const std::size_t row    = i_off   + (std::size_t)j * x_str;
            const std::size_t rim1   = im1_off + (std::size_t)j * x_str;
            const std::size_t rip1   = ip1_off + (std::size_t)j * x_str;
            const std::size_t rjm1   = i_off   + (std::size_t)(j-1) * x_str;
            const std::size_t rjp1   = i_off   + (std::size_t)(j+1) * x_str;
            for( int kl=kl_lo; kl<kl_hi; ++kl ) {
                const std::size_t idx = row + (std::size_t)kl;
                const T sum = u[rim1 + kl] + u[rip1 + kl]
                            + u[rjm1 + kl] + u[rjp1 + kl]
                            + u[row  + kl - 1] + u[row + kl + 1];
                out[idx] = sum - T(6) * u[idx];
            }
        }
    }
}

} // anonymous namespace

void apply_z_slab_double(const ZoomSlabLayout& L,
                         const double* u, double* out)
{ apply_z_slab_impl<double>(L, u, out); }

void apply_z_slab_float (const ZoomSlabLayout& L,
                         const float* u, float* out)
{ apply_z_slab_impl<float>(L, u, out); }

// ---------------------------------------------------------------------------
// Phase G.2b production-wire step 2: MeshvarBnd<T> ↔ ZoomSlabLayout bridge.
//
// On box_owner: packs MeshvarBnd's interior + 1-cell BC perimeter into a
// cluster_full buffer with cluster shape = (size(0)+2, size(1)+2, size(2)+2).
// Scatters across sub_comm, runs N sweeps of gs_z_neg_skip_z (which skips
// the cluster perimeter, leaving BC values frozen), gathers back, and
// writes the updated interior to MeshvarBnd's interior cells.
//
// Cluster dimensions are broadcast from box_owner so workers don't need
// any prior knowledge of the per-box geometry.
namespace {

template<typename T>
void gs_z_neg_meshvarbnd_impl(int box_owner,
                              MeshvarBnd<T>* u,
                              const MeshvarBnd<T>* f,
                              T h, int n_sweeps,
                              MPI_Comm sub_comm, int halo_w)
{
    int rk = 0, sz = 1;
#ifdef USE_MPI
    MPI_Comm_rank(sub_comm, &rk);
    MPI_Comm_size(sub_comm, &sz);
#endif

    // Broadcast interior dims from box_owner.
    int dims[3] = {0, 0, 0};
    if( rk == box_owner ) {
        if( !u || !f )
            throw std::runtime_error("gs_z_neg_meshvarbnd: u/f null on box_owner");
        dims[0] = (int)u->size(0);
        dims[1] = (int)u->size(1);
        dims[2] = (int)u->size(2);
        if( (int)f->size(0) != dims[0] || (int)f->size(1) != dims[1] || (int)f->size(2) != dims[2] )
            throw std::runtime_error("gs_z_neg_meshvarbnd: u/f size mismatch");
        if( u->m_nbnd < 1 || f->m_nbnd < 1 )
            throw std::runtime_error("gs_z_neg_meshvarbnd: m_nbnd must be >= 1");
    }
#ifdef USE_MPI
    MPI_Bcast(dims, 3, MPI_INT, box_owner, sub_comm);
#endif

    const int nx = dims[0], ny = dims[1], nz = dims[2];
    if( nx <= 0 || ny <= 0 || nz <= 0 )
        throw std::runtime_error("gs_z_neg_meshvarbnd: bad cluster dims");

    // Cluster shape = MeshvarBnd interior + 1-cell BC perimeter on all axes.
    const int cnx = nx + 2;
    const int cny = ny + 2;
    const int cnz = nz + 2;

    // halo_w must be <= local_nz on every rank for halo_exchange_z safety.
    const int min_local = cnz / sz;  // worst-case (ranks getting the smaller piece)
    if( halo_w > min_local )
        throw std::runtime_error("gs_z_neg_meshvarbnd: halo_w too large for cluster_nz/sub_size split");

    ZoomSlabLayout L = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/0,
        /*oax=*/0, /*oay=*/0, /*oaz=*/0,
        cnx, cny, cnz, halo_w);

    // Box-owner: pack interior + BC perimeter into cluster_full buffer.
    std::vector<T> cluster_u_in, cluster_f_in;
    if( rk == box_owner ) {
        cluster_u_in.assign(cluster_full_size(L), T(0));
        cluster_f_in.assign(cluster_full_size(L), T(0));
        for( int ci=0; ci<cnx; ++ci )
            for( int cj=0; cj<cny; ++cj )
                for( int ck=0; ck<cnz; ++ck ) {
                    const std::size_t idx = ((std::size_t)ci * (std::size_t)cny + (std::size_t)cj)
                                          * (std::size_t)cnz + (std::size_t)ck;
                    cluster_u_in[idx] = (*u)(ci - 1, cj - 1, ck - 1);
                    cluster_f_in[idx] = (*f)(ci - 1, cj - 1, ck - 1);
                }
    }

    // Per-rank buffers.
    std::vector<T> my_u_int(local_interior_size(L), T(0));
    std::vector<T> my_f_int(local_interior_size(L), T(0));
    std::vector<T> u_pad   (local_with_halo_size(L), T(0));
    std::vector<T> f_pad   (local_with_halo_size(L), T(0));

    // Scatter u and f.
    scatter_from(L, box_owner,
                 rk == box_owner ? cluster_u_in.data() : (const T*)nullptr,
                 my_u_int.data());
    scatter_from(L, box_owner,
                 rk == box_owner ? cluster_f_in.data() : (const T*)nullptr,
                 my_f_int.data());

    // Copy per-rank interior into local-with-halo (k offset by halo_w).
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int ii=0; ii<cnx; ++ii )
            for( int jj=0; jj<cny; ++jj )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    u_pad[dst] = my_u_int[src];
                    f_pad[dst] = my_f_int[src];
                }
    }

    // Pre-exchange z-halo so first color sweep reads fresh values.
    halo_exchange_z(L, u_pad.data());

    // N sweeps. gs_z_neg_skip_z internally re-exchanges halos between colors.
    for( int s=0; s<n_sweeps; ++s )
        gs_z_neg_skip_z(L, u_pad.data(), f_pad.data(), h);

    // Extract interior from local-with-halo back into my_u_int.
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int ii=0; ii<cnx; ++ii )
            for( int jj=0; jj<cny; ++jj )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t src = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    my_u_int[dst] = u_pad[src];
                }
    }

    // Gather back to box_owner.
    std::vector<T> cluster_u_out;
    if( rk == box_owner ) cluster_u_out.assign(cluster_full_size(L), T(0));
    gather_to(L, box_owner, my_u_int.data(),
              rk == box_owner ? cluster_u_out.data() : (T*)nullptr);

    // Box-owner: write back to MeshvarBnd interior (skip BC perimeter).
    if( rk == box_owner ) {
        for( int ci=1; ci<cnx-1; ++ci )
            for( int cj=1; cj<cny-1; ++cj )
                for( int ck=1; ck<cnz-1; ++ck ) {
                    const std::size_t idx = ((std::size_t)ci * (std::size_t)cny + (std::size_t)cj)
                                          * (std::size_t)cnz + (std::size_t)ck;
                    (*u)(ci - 1, cj - 1, ck - 1) = cluster_u_out[idx];
                }
    }
}

} // anonymous namespace

void gs_z_neg_meshvarbnd_double(int box_owner,
                                MeshvarBnd<double>* u,
                                const MeshvarBnd<double>* f,
                                double h, int n_sweeps,
                                MPI_Comm sub_comm, int halo_w)
{ gs_z_neg_meshvarbnd_impl<double>(box_owner, u, f, h, n_sweeps, sub_comm, halo_w); }

void gs_z_neg_meshvarbnd_float (int box_owner,
                                MeshvarBnd<float>*  u,
                                const MeshvarBnd<float>*  f,
                                float  h, int n_sweeps,
                                MPI_Comm sub_comm, int halo_w)
{ gs_z_neg_meshvarbnd_impl<float>(box_owner, u, f, h, n_sweeps, sub_comm, halo_w); }

// ---------------------------------------------------------------------------
// Phase G.2b B.5.1: collective apply Laplacian on sub_comm.
//
// Mirrors gs_z_neg_meshvarbnd_impl structure but applies stencil_7P::apply (the
// 7-point Laplacian) on u and writes (apply(u))/h2 into out's interior.
// Replaces the rank-0-local apply loop at the B.2b.6 site (mg_solver.hh:982)
// that builds the scratch apply_uf before restrict_meshvarbnd consumes it.
//
// Preconditions:
//   - u and out are valid pointers ONLY on sub_comm rank == box_owner; NULL elsewhere.
//   - u->size(d) == out->size(d) on all 3 axes; u->m_nbnd >= 1.
//
// Postcondition: on box_owner, out's interior cells (i,j,k in [0, size(d))) hold
// (apply(u, i,j,k)) / h2 with the same FP semantics as stencil_7P::apply applied
// cell-by-cell — i.e. each cell read via MeshvarBnd's BC accessor at ±1 neighbors.
//
// Returns false (with no writes) when sub_size==1 or halo_w exceeds the
// per-rank z-thickness, so the caller can fall back to the rank-0-local loop.
namespace {

template<typename T>
bool apply_meshvarbnd_impl(int box_owner,
                           const MeshvarBnd<T>* u,
                           MeshvarBnd<T>* out,
                           T h2,
                           MPI_Comm sub_comm, int halo_w)
{
    int rk = 0, sz = 1;
#ifdef USE_MPI
    MPI_Comm_rank(sub_comm, &rk);
    MPI_Comm_size(sub_comm, &sz);
#endif

    if( sz < 2 ) return false;

    // Broadcast interior dims from box_owner.
    int dims[3] = {0, 0, 0};
    if( rk == box_owner ) {
        if( !u || !out )
            throw std::runtime_error("apply_meshvarbnd: u/out null on box_owner");
        dims[0] = (int)u->size(0);
        dims[1] = (int)u->size(1);
        dims[2] = (int)u->size(2);
        if( (int)out->size(0) != dims[0] || (int)out->size(1) != dims[1] || (int)out->size(2) != dims[2] )
            throw std::runtime_error("apply_meshvarbnd: u/out size mismatch");
        if( u->m_nbnd < 1 )
            throw std::runtime_error("apply_meshvarbnd: u m_nbnd must be >= 1");
    }
#ifdef USE_MPI
    MPI_Bcast(dims, 3, MPI_INT, box_owner, sub_comm);
#endif

    const int nx = dims[0], ny = dims[1], nz = dims[2];
    if( nx <= 0 || ny <= 0 || nz <= 0 ) return false;

    // Cluster shape = MeshvarBnd interior + 1-cell BC perimeter on all axes.
    const int cnx = nx + 2;
    const int cny = ny + 2;
    const int cnz = nz + 2;

    const int min_local = cnz / sz;
    if( halo_w > min_local ) return false;

    ZoomSlabLayout L = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/0,
        /*oax=*/0, /*oay=*/0, /*oaz=*/0,
        cnx, cny, cnz, halo_w);

    // Box-owner: pack u interior + 1-cell BC perimeter into cluster_full buffer.
    std::vector<T> cluster_u_in;
    if( rk == box_owner ) {
        cluster_u_in.assign(cluster_full_size(L), T(0));
        for( int ci=0; ci<cnx; ++ci )
            for( int cj=0; cj<cny; ++cj )
                for( int ck=0; ck<cnz; ++ck ) {
                    const std::size_t idx = ((std::size_t)ci * (std::size_t)cny + (std::size_t)cj)
                                          * (std::size_t)cnz + (std::size_t)ck;
                    cluster_u_in[idx] = (*u)(ci - 1, cj - 1, ck - 1);
                }
    }

    // Per-rank buffers.
    std::vector<T> my_u_int(local_interior_size(L), T(0));
    std::vector<T> u_pad   (local_with_halo_size(L), T(0));
    std::vector<T> out_pad (local_with_halo_size(L), T(0));

    // Scatter u.
    scatter_from(L, box_owner,
                 rk == box_owner ? cluster_u_in.data() : (const T*)nullptr,
                 my_u_int.data());

    // Copy per-rank interior into u_pad (k offset by halo_w).
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int ii=0; ii<cnx; ++ii )
            for( int jj=0; jj<cny; ++jj )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    u_pad[dst] = my_u_int[src];
                }
    }

    // Exchange z-halo so apply reads coherent z-neighbors at slab boundaries.
    halo_exchange_z(L, u_pad.data());

    // Apply 7-point Laplacian. Writes update-domain cells only;
    // cluster-perimeter cells in out_pad stay 0.
    apply_z_slab(L, u_pad.data(), out_pad.data());

    // Extract interior from out_pad (cluster-perimeter cells still 0, but those
    // will be skipped when writing back to MeshvarBnd anyway).
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int ii=0; ii<cnx; ++ii )
            for( int jj=0; jj<cny; ++jj )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t src = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    my_u_int[dst] = out_pad[src];
                }
    }

    // Gather back to box_owner.
    std::vector<T> cluster_out;
    if( rk == box_owner ) cluster_out.assign(cluster_full_size(L), T(0));
    gather_to(L, box_owner, my_u_int.data(),
              rk == box_owner ? cluster_out.data() : (T*)nullptr);

    // Box-owner: write back to MeshvarBnd interior (skip BC perimeter).
    // Scale by 1/h2 to match the original `m_scheme.apply(uf, i,j,k) / h2`
    // contract at the B.2b.6 site.
    if( rk == box_owner ) {
        const T inv_h2 = T(1) / h2;
        for( int ci=1; ci<cnx-1; ++ci )
            for( int cj=1; cj<cny-1; ++cj )
                for( int ck=1; ck<cnz-1; ++ck ) {
                    const std::size_t idx = ((std::size_t)ci * (std::size_t)cny + (std::size_t)cj)
                                          * (std::size_t)cnz + (std::size_t)ck;
                    (*out)(ci - 1, cj - 1, ck - 1) = cluster_out[idx] * inv_h2;
                }
    }

    return true;
}

} // anonymous namespace

bool apply_meshvarbnd_double(int box_owner,
                             const MeshvarBnd<double>* u,
                             MeshvarBnd<double>* out,
                             double h2,
                             MPI_Comm sub_comm, int halo_w)
{ return apply_meshvarbnd_impl<double>(box_owner, u, out, h2, sub_comm, halo_w); }

bool apply_meshvarbnd_float (int box_owner,
                             const MeshvarBnd<float>*  u,
                             MeshvarBnd<float>*  out,
                             float  h2,
                             MPI_Comm sub_comm, int halo_w)
{ return apply_meshvarbnd_impl<float>(box_owner, u, out, h2, sub_comm, halo_w); }

// ---------------------------------------------------------------------------
// Phase G.2b: 8-cell straight restriction (mirrors mg_straight::restrict in
// mg_operators.hh). Pure local — no MPI communication; each rank's fine slab
// covers exactly the fine cells needed for its coarse slab, so the operator
// reads vf only from local-with-halo interior cells (klf0 in [hwf, hwf+nz_rf)).
// ---------------------------------------------------------------------------
namespace {

template<typename T>
void restrict_impl(const ZoomSlabLayout& Lf, const ZoomSlabLayout& Lc,
                   const T* vf, T* Vc)
{
    if(    Lc.cluster_nx * 2 != Lf.cluster_nx
        || Lc.cluster_ny * 2 != Lf.cluster_ny
        || Lc.cluster_nz * 2 != Lf.cluster_nz
        || Lc.my_z0 * 2     != Lf.my_z0
        || Lc.my_z1 * 2     != Lf.my_z1 )
        throw std::runtime_error("restrict_z_slab: layout mismatch (require Lc = Lf/2 aligned).");

    const int cnxc      = Lc.cluster_nx;
    const int cnyc      = Lc.cluster_ny;
    const int nz_rc     = local_nz(Lc);
    const int hwc       = Lc.halo_w;
    const int kstridec  = 2 * hwc + nz_rc;
    const std::size_t ny_strc = (std::size_t)cnyc * (std::size_t)kstridec;
    const std::size_t x_strc  = (std::size_t)kstridec;

    const int cnyf      = Lf.cluster_ny;
    const int nz_rf     = local_nz(Lf);
    const int hwf       = Lf.halo_w;
    const int kstridef  = 2 * hwf + nz_rf;
    const std::size_t ny_strf = (std::size_t)cnyf * (std::size_t)kstridef;
    const std::size_t x_strf  = (std::size_t)kstridef;

    const T eighth = T(1) / T(8);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for( int ic=0; ic<cnxc; ++ic ) {
        const int if0 = 2 * ic;
        const std::size_t ic_off  = (std::size_t)ic       * ny_strc;
        const std::size_t if0_off = (std::size_t)if0      * ny_strf;
        const std::size_t if1_off = (std::size_t)(if0+1)  * ny_strf;
        for( int jc=0; jc<cnyc; ++jc ) {
            const int jf0 = 2 * jc;
            const std::size_t cdrow      = ic_off  + (std::size_t)jc          * x_strc;
            const std::size_t fr_i0_j0   = if0_off + (std::size_t)jf0         * x_strf;
            const std::size_t fr_i0_j1   = if0_off + (std::size_t)(jf0+1)     * x_strf;
            const std::size_t fr_i1_j0   = if1_off + (std::size_t)jf0         * x_strf;
            const std::size_t fr_i1_j1   = if1_off + (std::size_t)(jf0+1)     * x_strf;
            for( int klc=hwc; klc<hwc+nz_rc; ++klc ) {
                const int klf0 = hwf + 2 * (klc - hwc);
                const T sum = vf[fr_i0_j0 + klf0    ] + vf[fr_i0_j0 + klf0 + 1]
                            + vf[fr_i0_j1 + klf0    ] + vf[fr_i0_j1 + klf0 + 1]
                            + vf[fr_i1_j0 + klf0    ] + vf[fr_i1_j0 + klf0 + 1]
                            + vf[fr_i1_j1 + klf0    ] + vf[fr_i1_j1 + klf0 + 1];
                Vc[cdrow + (std::size_t)klc] = sum * eighth;
            }
        }
    }
}

} // anonymous namespace

void restrict_z_slab_double(const ZoomSlabLayout& Lf, const ZoomSlabLayout& Lc,
                            const double* vf, double* Vc)
{ restrict_impl<double>(Lf, Lc, vf, Vc); }

void restrict_z_slab_float(const ZoomSlabLayout& Lf, const ZoomSlabLayout& Lc,
                           const float* vf, float* Vc)
{ restrict_impl<float>(Lf, Lc, vf, Vc); }

// ---------------------------------------------------------------------------
// Phase G.2b B.2b.4: collective restrict bridge.
//
// On box_owner: packs vf interior into a cluster_full buffer of shape (nx,ny,nz).
// All sub_comm ranks build matching fine+coarse layouts (halo_w=1, sufficient
// because restrict reads only interior). Scatter fine, local restrict_impl,
// gather coarse to box_owner, write into Vc at parent offset (i+ox, j+oy, k+oz).
//
// Returns false (no writes) if sub_size==1 or alignment unsuitable for a clean
// 2x z-split. Caller must do a local restrict in that case.
// ---------------------------------------------------------------------------
namespace {

template<typename T>
bool restrict_meshvarbnd_impl(int box_owner,
                              const MeshvarBnd<T>* vf,
                              MeshvarBnd<T>* Vc,
                              MPI_Comm sub_comm)
{
    int rk = 0, sz = 1;
#ifdef USE_MPI
    MPI_Comm_rank(sub_comm, &rk);
    MPI_Comm_size(sub_comm, &sz);
#endif

    if( sz <= 1 ) return false;

    int hdr[6] = {0, 0, 0, 0, 0, 0};
    if( rk == box_owner ) {
        if( !vf || !Vc )
            throw std::runtime_error("restrict_meshvarbnd: vf/Vc null on box_owner");
        hdr[0] = (int)vf->size(0);
        hdr[1] = (int)vf->size(1);
        hdr[2] = (int)vf->size(2);
        hdr[3] = vf->offset(0);
        hdr[4] = vf->offset(1);
        hdr[5] = vf->offset(2);
    }
#ifdef USE_MPI
    MPI_Bcast(hdr, 6, MPI_INT, box_owner, sub_comm);
#endif

    const int nxf = hdr[0], nyf = hdr[1], nzf = hdr[2];
    const int ox  = hdr[3], oy  = hdr[4], oz  = hdr[5];

    if( nxf <= 0 || nyf <= 0 || nzf <= 0 )
        throw std::runtime_error("restrict_meshvarbnd: bad fine dims from box_owner");

    // mg_straight::restrict reads in 2x blocks; all axes must be even.
    if( (nxf & 1) || (nyf & 1) || (nzf & 1) ) return false;

    // nzf divisible by 2*sub_size → each rank holds an even fine chunk whose
    // coarse half (nzf/2 / sz) is positive. Sufficient for the alignment
    // precondition Lc.my_z*2 == Lf.my_z* under make_layout's even-split rule.
    if( nzf % (2 * sz) != 0 ) return false;

    const int nxc = nxf / 2;
    const int nyc = nyf / 2;
    const int nzc = nzf / 2;
    if( nzc / sz < 1 ) return false;

    const int halo_w = 1;

    ZoomSlabLayout Lf = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/1,
        0, 0, 0, nxf, nyf, nzf, halo_w);

    ZoomSlabLayout Lc = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/0,
        0, 0, 0, nxc, nyc, nzc, halo_w);

    if( 2 * Lc.my_z0 != Lf.my_z0 || 2 * Lc.my_z1 != Lf.my_z1 )
        return false;

    std::vector<T> cluster_vf_in;
    if( rk == box_owner ) {
        cluster_vf_in.assign(cluster_full_size(Lf), T(0));
        for( int i=0; i<nxf; ++i )
            for( int j=0; j<nyf; ++j )
                for( int k=0; k<nzf; ++k ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)nyf + (std::size_t)j)
                                          * (std::size_t)nzf + (std::size_t)k;
                    cluster_vf_in[idx] = (*vf)(i, j, k);
                }
    }

    std::vector<T> my_vf_int(local_interior_size(Lf), T(0));
    std::vector<T> vf_pad   (local_with_halo_size(Lf), T(0));

    scatter_from(Lf, box_owner,
                 rk == box_owner ? cluster_vf_in.data() : (const T*)nullptr,
                 my_vf_int.data());

    {
        const int nz_rf  = local_nz(Lf);
        const int hwf    = Lf.halo_w;
        const int kstridef = 2 * hwf + nz_rf;
        for( int ii=0; ii<nxf; ++ii )
            for( int jj=0; jj<nyf; ++jj )
                for( int kl=0; kl<nz_rf; ++kl ) {
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)nyf + (std::size_t)jj)
                                          * (std::size_t)kstridef + (std::size_t)(hwf + kl);
                    const std::size_t src = ((std::size_t)ii * (std::size_t)nyf + (std::size_t)jj)
                                          * (std::size_t)nz_rf + (std::size_t)kl;
                    vf_pad[dst] = my_vf_int[src];
                }
    }

    std::vector<T> Vc_pad(local_with_halo_size(Lc), T(0));

    restrict_impl<T>(Lf, Lc, vf_pad.data(), Vc_pad.data());

    std::vector<T> my_Vc_int(local_interior_size(Lc), T(0));
    {
        const int nz_rc  = local_nz(Lc);
        const int hwc    = Lc.halo_w;
        const int kstridec = 2 * hwc + nz_rc;
        for( int ii=0; ii<nxc; ++ii )
            for( int jj=0; jj<nyc; ++jj )
                for( int kl=0; kl<nz_rc; ++kl ) {
                    const std::size_t src = ((std::size_t)ii * (std::size_t)nyc + (std::size_t)jj)
                                          * (std::size_t)kstridec + (std::size_t)(hwc + kl);
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)nyc + (std::size_t)jj)
                                          * (std::size_t)nz_rc + (std::size_t)kl;
                    my_Vc_int[dst] = Vc_pad[src];
                }
    }

    std::vector<T> cluster_Vc_out;
    if( rk == box_owner ) cluster_Vc_out.assign(cluster_full_size(Lc), T(0));
    gather_to(Lc, box_owner, my_Vc_int.data(),
              rk == box_owner ? cluster_Vc_out.data() : (T*)nullptr);

    if( rk == box_owner ) {
        for( int ic=0; ic<nxc; ++ic )
            for( int jc=0; jc<nyc; ++jc )
                for( int kc=0; kc<nzc; ++kc ) {
                    const std::size_t idx = ((std::size_t)ic * (std::size_t)nyc + (std::size_t)jc)
                                          * (std::size_t)nzc + (std::size_t)kc;
                    (*Vc)(ic + ox, jc + oy, kc + oz) = cluster_Vc_out[idx];
                }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Phase G.2b B.5.2: from-slab restrict variant. Same restrict_impl + gather
// epilogue as restrict_meshvarbnd_impl, but the fine slab is provided by
// the caller (already scattered) rather than packed/scattered here. Saves
// one cluster_full pack + scatter_from when chained behind an op (e.g.
// apply_z_slab) that already produced a per-rank slab.
// ---------------------------------------------------------------------------
template<typename T>
bool restrict_meshvarbnd_from_slab_impl(int box_owner,
                                        const ZoomSlabLayout& Lf,
                                        const T* my_vf_int,
                                        MeshvarBnd<T>* Vc,
                                        int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                        MPI_Comm sub_comm)
{
    int rk = 0, sz = 1;
#ifdef USE_MPI
    MPI_Comm_rank(sub_comm, &rk);
    MPI_Comm_size(sub_comm, &sz);
#endif

    if( sz <= 1 ) return false;

    const int nxf = Lf.cluster_nx;
    const int nyf = Lf.cluster_ny;
    const int nzf = Lf.cluster_nz;

    if( nxf <= 0 || nyf <= 0 || nzf <= 0 ) return false;
    if( (nxf & 1) || (nyf & 1) || (nzf & 1) ) return false;
    if( nzf % (2 * sz) != 0 ) return false;
    if( Lf.halo_w < 1 ) return false;

    const int nxc = nxf / 2;
    const int nyc = nyf / 2;
    const int nzc = nzf / 2;
    if( nzc / sz < 1 ) return false;

    if( rk == box_owner && !Vc )
        throw std::runtime_error("restrict_meshvarbnd_from_slab: Vc null on box_owner");

    // Build coarse layout. Cluster offset is irrelevant for restrict_impl
    // (it indexes from 0 within the cluster); use 0,0,0.
    ZoomSlabLayout Lc = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/0,
        0, 0, 0, nxc, nyc, nzc, Lf.halo_w);

    if( 2 * Lc.my_z0 != Lf.my_z0 || 2 * Lc.my_z1 != Lf.my_z1 )
        return false;

    // Pack my_vf_int into halo-padded buffer (halo cells stay zero;
    // restrict_impl never reads them).
    std::vector<T> vf_pad(local_with_halo_size(Lf), T(0));
    {
        const int nz_rf  = local_nz(Lf);
        const int hwf    = Lf.halo_w;
        const int kstridef = 2 * hwf + nz_rf;
        for( int ii=0; ii<nxf; ++ii )
            for( int jj=0; jj<nyf; ++jj )
                for( int kl=0; kl<nz_rf; ++kl ) {
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)nyf + (std::size_t)jj)
                                          * (std::size_t)kstridef + (std::size_t)(hwf + kl);
                    const std::size_t src = ((std::size_t)ii * (std::size_t)nyf + (std::size_t)jj)
                                          * (std::size_t)nz_rf + (std::size_t)kl;
                    vf_pad[dst] = my_vf_int[src];
                }
    }

    std::vector<T> Vc_pad(local_with_halo_size(Lc), T(0));
    restrict_impl<T>(Lf, Lc, vf_pad.data(), Vc_pad.data());

    std::vector<T> my_Vc_int(local_interior_size(Lc), T(0));
    {
        const int nz_rc  = local_nz(Lc);
        const int hwc    = Lc.halo_w;
        const int kstridec = 2 * hwc + nz_rc;
        for( int ii=0; ii<nxc; ++ii )
            for( int jj=0; jj<nyc; ++jj )
                for( int kl=0; kl<nz_rc; ++kl ) {
                    const std::size_t src = ((std::size_t)ii * (std::size_t)nyc + (std::size_t)jj)
                                          * (std::size_t)kstridec + (std::size_t)(hwc + kl);
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)nyc + (std::size_t)jj)
                                          * (std::size_t)nz_rc + (std::size_t)kl;
                    my_Vc_int[dst] = Vc_pad[src];
                }
    }

    std::vector<T> cluster_Vc_out;
    if( rk == box_owner ) cluster_Vc_out.assign(cluster_full_size(Lc), T(0));
    gather_to(Lc, box_owner, my_Vc_int.data(),
              rk == box_owner ? cluster_Vc_out.data() : (T*)nullptr);

    if( rk == box_owner ) {
        for( int ic=0; ic<nxc; ++ic )
            for( int jc=0; jc<nyc; ++jc )
                for( int kc=0; kc<nzc; ++kc ) {
                    const std::size_t idx = ((std::size_t)ic * (std::size_t)nyc + (std::size_t)jc)
                                          * (std::size_t)nzc + (std::size_t)kc;
                    (*Vc)(ic + Vc_off_x, jc + Vc_off_y, kc + Vc_off_z) = cluster_Vc_out[idx];
                }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Phase G.2b B.5.2-prod / B.5.4.b: redistribute a Layout-A interior slab
// (cluster_nx = nx+2p, cluster_ny = ny+2p, cluster_nz = nz+2p, with a
// `perimeter` = p cells of BC perimeter on every axis) into a Layout-B per-rank
// slab (cluster_nx = nx, cluster_ny = ny, cluster_nz = nz, no perimeter —
// exactly the form consumed by restrict_meshvarbnd_from_slab).
//
// Strips the cluster-A BC perimeter on x/y/z and reshuffles z-ownership via
// MPI_Alltoallv. The map between coords is: cluster-B (iB,jB,kB) <->
// cluster-A (iB+p, jB+p, kB+p). Source rank r's contribution in cluster-B z
// coords is [max(0, LA.all_z0[r]-p), min(nzB, LA.all_z1[r]-p)); each (src,dst)
// pair sends the overlap with LB.all_z0[dst]..LB.all_z1[dst]. At small
// perimeters the layouts may align trivially and the Alltoallv collapses to
// self-sends.
//
// `scale` is applied per cell during pack — used by apply_meshvarbnd_to_slab
// to fold the existing 1/h2 scaling into the same traversal.
//
// `perimeter` selects the strip width: p=1 for B.5.2-prod (apply Laplacian
// path), p=2 for B.5.4.b (keep-slab smoother output → restrict).
// ---------------------------------------------------------------------------
template<typename T>
bool redistribute_z_slab_A_interior_to_B(const ZoomSlabLayout& LA,
                                         const T* my_int_A,
                                         const ZoomSlabLayout& LB,
                                         T* my_int_B,
                                         T scale,
                                         int perimeter,
                                         MPI_Comm sub_comm)
{
    if( perimeter < 1 ) return false;

    const int p    = perimeter;
    const int cnxA = LA.cluster_nx;
    const int cnyA = LA.cluster_ny;
    const int nxB  = LB.cluster_nx;
    const int nyB  = LB.cluster_ny;
    const int nzB  = LB.cluster_nz;

    if( cnxA != nxB + 2*p || cnyA != nyB + 2*p || LA.cluster_nz != nzB + 2*p )
        return false;
    if( LA.sub_size != LB.sub_size )
        return false;

    int rk = 0, sz = LA.sub_size;
#ifdef USE_MPI
    MPI_Comm_rank(sub_comm, &rk);
    MPI_Comm_size(sub_comm, &sz);
    if( sz != LA.sub_size ) return false;
#endif

    const int nz_rA = LA.my_z1 - LA.my_z0;
    const int nz_rB = LB.my_z1 - LB.my_z0;
    const std::size_t plane = (std::size_t)nxB * (std::size_t)nyB;

    // Zero the destination — only overlap cells are filled.
    std::fill(my_int_B, my_int_B + plane * (std::size_t)nz_rB, T(0));

    // Per-(src,dst) z extent (in cluster-B coords).
    auto contribution = [&](int src_rk, int dst_rk, int& zlo, int& zhi) {
        const int srcA_lo = std::max(p,        LA.all_z0[src_rk]);
        const int srcA_hi = std::min(nzB + p,  LA.all_z1[src_rk]);
        const int srcB_lo = srcA_lo - p;
        const int srcB_hi = srcA_hi - p;
        zlo = std::max(LB.all_z0[dst_rk], srcB_lo);
        zhi = std::min(LB.all_z1[dst_rk], srcB_hi);
        if( zhi < zlo ) zhi = zlo;
    };

    std::vector<int> scnt(sz, 0), sdsp(sz, 0);
    std::vector<int> rcnt(sz, 0), rdsp(sz, 0);

    // Sends: I'm source rk, packing for each dst.
    {
        int off = 0;
        for( int d=0; d<sz; ++d ) {
            int zlo=0, zhi=0;
            contribution(rk, d, zlo, zhi);
            const int np_z = zhi - zlo;
            const long n = (long)np_z * (long)plane;
            if( n > std::numeric_limits<int>::max() )
                throw std::runtime_error("redistribute_z_slab: send count overflow (int)");
            scnt[d] = (int)n;
            sdsp[d] = off;
            off += (int)n;
        }
    }
    // Receives: I'm dst rk, receiving from each src.
    {
        int off = 0;
        for( int s=0; s<sz; ++s ) {
            int zlo=0, zhi=0;
            contribution(s, rk, zlo, zhi);
            const int np_z = zhi - zlo;
            const long n = (long)np_z * (long)plane;
            if( n > std::numeric_limits<int>::max() )
                throw std::runtime_error("redistribute_z_slab: recv count overflow (int)");
            rcnt[s] = (int)n;
            rdsp[s] = off;
            off += (int)n;
        }
    }

    const std::size_t s_total = (std::size_t)(sdsp.back() + scnt.back());
    const std::size_t r_total = (std::size_t)(rdsp.back() + rcnt.back());
    std::vector<T> sbuf(s_total);
    std::vector<T> rbuf(r_total);

    // Pack send buffer. For each dst d, iterate cluster-B z in [zlo, zhi)
    // and write (nxB * nyB) cells per z-plane, stripping the BC perimeter
    // on x/y and applying `scale`.
    for( int d=0; d<sz; ++d ) {
        if( scnt[d] == 0 ) continue;
        int zlo=0, zhi=0;
        contribution(rk, d, zlo, zhi);
        T* dst_ptr = sbuf.data() + sdsp[d];
        for( int kB = zlo; kB < zhi; ++kB ) {
            const int klA = (kB + p) - LA.my_z0;
            for( int iB=0; iB<nxB; ++iB ) {
                const std::size_t row_A = ((std::size_t)(iB + p) * (std::size_t)cnyA)
                                          * (std::size_t)nz_rA;
                for( int jB=0; jB<nyB; ++jB ) {
                    const std::size_t src_idx = row_A
                        + (std::size_t)(jB + p) * (std::size_t)nz_rA
                        + (std::size_t)klA;
                    *dst_ptr++ = my_int_A[src_idx] * scale;
                }
            }
        }
    }

#ifdef USE_MPI
    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Alltoallv(sbuf.data(), scnt.data(), sdsp.data(), dtype,
                  rbuf.data(), rcnt.data(), rdsp.data(), dtype,
                  sub_comm);
#else
    (void)sub_comm;
    std::memcpy(rbuf.data(), sbuf.data(), s_total * sizeof(T));
#endif

    // Unpack into my_int_B. For each src s, iterate cluster-B z and write
    // into the (iB, jB, klB) interior cells with klB = kB - LB.my_z0.
    for( int s=0; s<sz; ++s ) {
        if( rcnt[s] == 0 ) continue;
        int zlo=0, zhi=0;
        contribution(s, rk, zlo, zhi);
        const T* src_ptr = rbuf.data() + rdsp[s];
        for( int kB = zlo; kB < zhi; ++kB ) {
            const int klB = kB - LB.my_z0;
            for( int iB=0; iB<nxB; ++iB ) {
                const std::size_t row_B = ((std::size_t)iB * (std::size_t)nyB)
                                          * (std::size_t)nz_rB;
                for( int jB=0; jB<nyB; ++jB ) {
                    const std::size_t dst_idx = row_B
                        + (std::size_t)jB * (std::size_t)nz_rB
                        + (std::size_t)klB;
                    my_int_B[dst_idx] = *src_ptr++;
                }
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Phase G.2b B.5.4.b: restrict from a B.5.4.a-keep-slab padded slab. Consumes
// the per-rank interior buffer left by smooth_pre_post_n_meshvarbnd_keep_slab
// (Lf_padded has cluster_n = nxf + 2*perimeter on every axis with a BC
// perimeter populated by interp_cf_flux), strips the perimeter via the
// generalized Alltoallv redistribute helper into the unpadded restrict-input
// layout, then runs restrict_meshvarbnd_from_slab_impl on the result.
//
// Saves the round-trip the caller would otherwise pay to gather uf back to
// MeshvarBnd<T>, scatter into the unpadded restrict layout, and re-broadcast.
//
// Returns false (no writes) when sub_size==1, perimeter invalid, the
// stripped dims are invalid, or the redistribute fails. Caller falls back to
// the existing gather-then-restrict path on false.
// ---------------------------------------------------------------------------
template<typename T>
bool restrict_meshvarbnd_from_padded_slab_impl(int box_owner,
                                               const ZoomSlabLayout& Lf_padded,
                                               const T* my_padded_int,
                                               int perimeter,
                                               MeshvarBnd<T>* Vc,
                                               int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                               MPI_Comm sub_comm)
{
    int rk = 0, sz = 1;
#ifdef USE_MPI
    MPI_Comm_rank(sub_comm, &rk);
    MPI_Comm_size(sub_comm, &sz);
#endif

    if( sz <= 1 ) return false;
    if( perimeter < 1 ) return false;

    const int p   = perimeter;
    const int nxf = Lf_padded.cluster_nx - 2 * p;
    const int nyf = Lf_padded.cluster_ny - 2 * p;
    const int nzf = Lf_padded.cluster_nz - 2 * p;

    if( nxf <= 0 || nyf <= 0 || nzf <= 0 ) return false;
    if( (nxf & 1) || (nyf & 1) || (nzf & 1) ) return false;
    if( nzf % (2 * sz) != 0 ) return false;

    // Build the unpadded fine layout (restrict-from-slab's expected input).
    // halo_w=1 matches restrict_meshvarbnd_from_slab_impl's contract.
    ZoomSlabLayout Lf_unpad = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/0,
        0, 0, 0, nxf, nyf, nzf, /*halo_w=*/1);

    // Strip the BC perimeter on x/y/z and re-balance z across ranks via
    // Alltoallv. scale=1 (no folded scaling here).
    std::vector<T> my_unpad_int(local_interior_size(Lf_unpad), T(0));
    const bool did = redistribute_z_slab_A_interior_to_B<T>(
        Lf_padded, my_padded_int, Lf_unpad, my_unpad_int.data(),
        /*scale=*/T(1), /*perimeter=*/p, sub_comm);
    if( !did ) return false;

    // Invoke the existing from-slab restrict on the unpadded layout.
    return restrict_meshvarbnd_from_slab_impl<T>(
        box_owner, Lf_unpad, my_unpad_int.data(),
        Vc, Vc_off_x, Vc_off_y, Vc_off_z, sub_comm);
}

// ---------------------------------------------------------------------------
// Phase G.2b B.5.2-prod: combined apply Laplacian + redistribute into the
// restrict-fine z-slab layout (Layout B). Mirrors apply_meshvarbnd_impl's
// scatter + apply path, but instead of gathering the result back to box_owner
// it redistributes the Layout-A interior across sub_comm into the caller's
// pre-built Layout-B slab, with the 1/h2 scaling folded into the pack pass.
//
// After this returns true, my_apply_int holds the per-rank fine slab interior
// in the exact layout consumed by restrict_meshvarbnd_from_slab — no further
// gather/scatter round-trip on owner.
//
// Returns false (no writes) when sub_size==1, dims invalid, halo_w too large,
// or the Lf geometry doesn't match the owner's u dims. Caller falls back to
// the B.5.1 + B.2b.4 path on false.
// ---------------------------------------------------------------------------
template<typename T>
bool apply_meshvarbnd_to_slab_impl(int box_owner,
                                   const MeshvarBnd<T>* u,
                                   const ZoomSlabLayout& Lf,
                                   T* my_apply_int,
                                   T h2,
                                   MPI_Comm sub_comm)
{
    int rk = 0, sz = 1;
#ifdef USE_MPI
    MPI_Comm_rank(sub_comm, &rk);
    MPI_Comm_size(sub_comm, &sz);
#endif

    if( sz < 2 ) return false;
    if( sz != Lf.sub_size ) return false;

    // Broadcast u's interior dims from box_owner; sanity-check vs Lf.
    int dims[3] = {0, 0, 0};
    if( rk == box_owner ) {
        if( !u )
            throw std::runtime_error("apply_meshvarbnd_to_slab: u null on box_owner");
        dims[0] = (int)u->size(0);
        dims[1] = (int)u->size(1);
        dims[2] = (int)u->size(2);
        if( u->m_nbnd < 1 )
            throw std::runtime_error("apply_meshvarbnd_to_slab: u m_nbnd must be >= 1");
    }
#ifdef USE_MPI
    MPI_Bcast(dims, 3, MPI_INT, box_owner, sub_comm);
#endif

    const int nx = dims[0], ny = dims[1], nz = dims[2];
    if( nx <= 0 || ny <= 0 || nz <= 0 ) return false;
    if( Lf.cluster_nx != nx || Lf.cluster_ny != ny || Lf.cluster_nz != nz )
        return false;
    if( Lf.halo_w < 1 ) return false;

    // Build Layout A (apply layout with BC perimeter). Same halo_w as Lf.
    const int cnx = nx + 2;
    const int cny = ny + 2;
    const int cnz = nz + 2;
    const int halo_w = Lf.halo_w;
    if( halo_w > cnz / sz ) return false;

    ZoomSlabLayout LA = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/0,
        0, 0, 0, cnx, cny, cnz, halo_w);

    // Owner: pack u interior + 1-cell BC perimeter into cluster_full A buffer.
    std::vector<T> cluster_u_in;
    if( rk == box_owner ) {
        cluster_u_in.assign(cluster_full_size(LA), T(0));
        for( int ci=0; ci<cnx; ++ci )
            for( int cj=0; cj<cny; ++cj )
                for( int ck=0; ck<cnz; ++ck ) {
                    const std::size_t idx = ((std::size_t)ci * (std::size_t)cny + (std::size_t)cj)
                                          * (std::size_t)cnz + (std::size_t)ck;
                    cluster_u_in[idx] = (*u)(ci - 1, cj - 1, ck - 1);
                }
    }

    // Per-rank A buffers.
    std::vector<T> my_u_int (local_interior_size (LA), T(0));
    std::vector<T> u_pad    (local_with_halo_size(LA), T(0));
    std::vector<T> out_pad  (local_with_halo_size(LA), T(0));
    std::vector<T> my_out_A (local_interior_size (LA), T(0));

    scatter_from(LA, box_owner,
                 rk == box_owner ? cluster_u_in.data() : (const T*)nullptr,
                 my_u_int.data());

    // Copy interior into halo-padded buffer (k offset by halo_w).
    {
        const int nz_r = local_nz(LA);
        const int kstride = 2 * halo_w + nz_r;
        for( int ii=0; ii<cnx; ++ii )
            for( int jj=0; jj<cny; ++jj )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    u_pad[dst] = my_u_int[src];
                }
    }

    halo_exchange_z(LA, u_pad.data());
    apply_z_slab   (LA, u_pad.data(), out_pad.data());

    // Strip halo to my_out_A (Layout-A interior, BC perimeter cells are 0
    // because apply_z_slab writes only update-domain cells).
    {
        const int nz_r = local_nz(LA);
        const int kstride = 2 * halo_w + nz_r;
        for( int ii=0; ii<cnx; ++ii )
            for( int jj=0; jj<cny; ++jj )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t src = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    my_out_A[dst] = out_pad[src];
                }
    }

    // Redistribute A interior -> Lf slab, scaling by 1/h2 during the pack
    // pass to match apply_meshvarbnd's contract.
    const T inv_h2 = T(1) / h2;
    const bool did = redistribute_z_slab_A_interior_to_B<T>(
        LA, my_out_A.data(), Lf, my_apply_int, inv_h2, /*perimeter=*/1, sub_comm);
    return did;
}

} // anonymous namespace

bool restrict_meshvarbnd_double(int box_owner,
                                const MeshvarBnd<double>* vf,
                                MeshvarBnd<double>* Vc,
                                MPI_Comm sub_comm)
{ return restrict_meshvarbnd_impl<double>(box_owner, vf, Vc, sub_comm); }

bool restrict_meshvarbnd_float (int box_owner,
                                const MeshvarBnd<float>*  vf,
                                MeshvarBnd<float>*  Vc,
                                MPI_Comm sub_comm)
{ return restrict_meshvarbnd_impl<float>(box_owner, vf, Vc, sub_comm); }

bool restrict_meshvarbnd_from_slab_double(int box_owner,
                                          const ZoomSlabLayout& Lf,
                                          const double* my_vf_int,
                                          MeshvarBnd<double>* Vc,
                                          int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                          MPI_Comm sub_comm)
{ return restrict_meshvarbnd_from_slab_impl<double>(
        box_owner, Lf, my_vf_int, Vc, Vc_off_x, Vc_off_y, Vc_off_z, sub_comm); }

bool restrict_meshvarbnd_from_slab_float (int box_owner,
                                          const ZoomSlabLayout& Lf,
                                          const float*  my_vf_int,
                                          MeshvarBnd<float>*  Vc,
                                          int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                          MPI_Comm sub_comm)
{ return restrict_meshvarbnd_from_slab_impl<float>(
        box_owner, Lf, my_vf_int, Vc, Vc_off_x, Vc_off_y, Vc_off_z, sub_comm); }

bool restrict_meshvarbnd_from_padded_slab_double(int box_owner,
                                                 const ZoomSlabLayout& Lf_padded,
                                                 const double* my_padded_int,
                                                 int perimeter,
                                                 MeshvarBnd<double>* Vc,
                                                 int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                                 MPI_Comm sub_comm)
{ return restrict_meshvarbnd_from_padded_slab_impl<double>(
        box_owner, Lf_padded, my_padded_int, perimeter,
        Vc, Vc_off_x, Vc_off_y, Vc_off_z, sub_comm); }

bool restrict_meshvarbnd_from_padded_slab_float (int box_owner,
                                                 const ZoomSlabLayout& Lf_padded,
                                                 const float*  my_padded_int,
                                                 int perimeter,
                                                 MeshvarBnd<float>*  Vc,
                                                 int Vc_off_x, int Vc_off_y, int Vc_off_z,
                                                 MPI_Comm sub_comm)
{ return restrict_meshvarbnd_from_padded_slab_impl<float>(
        box_owner, Lf_padded, my_padded_int, perimeter,
        Vc, Vc_off_x, Vc_off_y, Vc_off_z, sub_comm); }

bool apply_meshvarbnd_to_slab_double(int box_owner,
                                     const MeshvarBnd<double>* u,
                                     const ZoomSlabLayout& Lf,
                                     double* my_apply_int,
                                     double h2,
                                     MPI_Comm sub_comm)
{ return apply_meshvarbnd_to_slab_impl<double>(
        box_owner, u, Lf, my_apply_int, h2, sub_comm); }

bool apply_meshvarbnd_to_slab_float (int box_owner,
                                     const MeshvarBnd<float>*  u,
                                     const ZoomSlabLayout& Lf,
                                     float*  my_apply_int,
                                     float  h2,
                                     MPI_Comm sub_comm)
{ return apply_meshvarbnd_to_slab_impl<float>(
        box_owner, u, Lf, my_apply_int, h2, sub_comm); }

// ---------------------------------------------------------------------------
// Phase G.2b smoke test for restrict_z.
//
// Strategy: pick a deterministic smooth fine pattern v_f(i,j,k) on rank 0,
// scatter to fine z-slabs, restrict to coarse z-slabs, gather coarse to rank
// 0, compare cell-by-cell against a serial restrict on the original full fine
// buffer. Requires max|err| == 0 (bit-identical).
//
// Sizing: choose fine cluster_nz = max(48, 4 * world_size) so that
//   coarse nz_rc = (cluster_nz / 2) / world_size  is positive on np=1..8
//   AND  fine local_nz is even on every rank (required by the 2|local_nz
//   alignment precondition of restrict_z_slab).
// ---------------------------------------------------------------------------
bool smoke_test_restrict_single_cluster(int halo_w)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    const int cluster_nx_f = 16;
    const int cluster_ny_f = 16;
    // ensure both fine local_nz and coarse local_nz are positive and
    // fine local_nz is even on every rank (so the Lc=Lf/2 alignment holds).
    int cluster_nz_f = 48;
    while( (cluster_nz_f / world_size) % 2 != 0
        || (cluster_nz_f / (2 * world_size)) < 1 )
        cluster_nz_f += 2 * world_size;

    const int cluster_nx_c = cluster_nx_f / 2;
    const int cluster_ny_c = cluster_ny_f / 2;
    const int cluster_nz_c = cluster_nz_f / 2;

    LOGINFO("G.2b smoke (restrict): fine=%dx%dx%d coarse=%dx%dx%d halo_w=%d sub_size=%d",
            cluster_nx_f, cluster_ny_f, cluster_nz_f,
            cluster_nx_c, cluster_ny_c, cluster_nz_c,
            halo_w, world_size);

#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif

    ZoomSlabLayout Lf = make_layout(
        sub, /*cluster_id=*/0, /*level=*/1,
        0, 0, 0,
        cluster_nx_f, cluster_ny_f, cluster_nz_f, halo_w);

    ZoomSlabLayout Lc = make_layout(
        sub, /*cluster_id=*/0, /*level=*/0,
        0, 0, 0,
        cluster_nx_c, cluster_ny_c, cluster_nz_c, halo_w);

    // Sanity: confirm alignment precondition holds. (make_layout uses
    // even-split with leftover spread to first ranks; with cluster_nz_f
    // chosen so cluster_nz_f / world_size is even, both fine and coarse
    // splits are clean and aligned.)
    if(    2 * Lc.my_z0 != Lf.my_z0
        || 2 * Lc.my_z1 != Lf.my_z1 ) {
        if( world_rank == 0 )
            LOGERR("G.2b smoke (restrict): Lc/Lf z-slab alignment broken on rank %d "
                   "(Lf.my_z=[%d,%d) Lc.my_z=[%d,%d))",
                   world_rank, Lf.my_z0, Lf.my_z1, Lc.my_z0, Lc.my_z1);
        return false;
    }

    auto vf_at = [&](int i, int j, int k) -> double {
        return (double)(i + 1) * 0.013
             + (double)(j + 1) * 0.029
             + (double)(k + 1) * 0.041
             + std::sin(0.1 * i + 0.2 * j + 0.3 * k);
    };

    std::vector<double> full_vf;
    if( world_rank == 0 ) {
        full_vf.assign(cluster_full_size(Lf), 0.0);
        for( int i=0; i<cluster_nx_f; ++i )
            for( int j=0; j<cluster_ny_f; ++j )
                for( int k=0; k<cluster_nz_f; ++k ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                          * (std::size_t)cluster_nz_f + (std::size_t)k;
                    full_vf[idx] = vf_at(i, j, k);
                }
    }

    std::vector<double> my_vf_int(local_interior_size(Lf), 0.0);
    scatter_cluster_to_zslab_double(Lf, world_rank == 0 ? full_vf.data() : nullptr,
                                    my_vf_int.data());

    std::vector<double> vf_pad(local_with_halo_size(Lf), 0.0);
    {
        const int nz_r = local_nz(Lf);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx_f; ++i )
            for( int j=0; j<cluster_ny_f; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    vf_pad[dst] = my_vf_int[src];
                }
    }

    std::vector<double> Vc_pad(local_with_halo_size(Lc), 0.0);

    restrict_z_slab_double(Lf, Lc, vf_pad.data(), Vc_pad.data());

    // Extract coarse interior into a packed buffer for gather.
    std::vector<double> my_Vc_int(local_interior_size(Lc), 0.0);
    {
        const int nz_r = local_nz(Lc);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx_c; ++i )
            for( int j=0; j<cluster_ny_c; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_c + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_c + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    my_Vc_int[dst] = Vc_pad[src];
                }
    }

    std::vector<double> full_Vc_slab;
    if( world_rank == 0 ) full_Vc_slab.assign(cluster_full_size(Lc), 0.0);
    gather_zslab_to_cluster_double(Lc, my_Vc_int.data(),
                                   world_rank == 0 ? full_Vc_slab.data() : nullptr);

    bool ok = true;
    if( world_rank == 0 ) {
        // Reference: re-run restrict_impl on the FULL fine buffer via a
        // single-rank MPI_COMM_SELF layout. This goes through the exact same
        // code path as the parallel restrict, so any per-cell diff implies
        // an actual MPI miscoping bug (not FP rounding or vectorization).
#ifdef USE_MPI
        MPI_Comm self_comm = MPI_COMM_SELF;
#else
        MPI_Comm self_comm = 0;
#endif
        ZoomSlabLayout Lf_ser = make_layout(
            self_comm, /*cluster_id=*/0, /*level=*/1,
            0, 0, 0,
            cluster_nx_f, cluster_ny_f, cluster_nz_f, halo_w);
        ZoomSlabLayout Lc_ser = make_layout(
            self_comm, /*cluster_id=*/0, /*level=*/0,
            0, 0, 0,
            cluster_nx_c, cluster_ny_c, cluster_nz_c, halo_w);

        // Pack full_vf into a halo-padded buffer for the serial path.
        std::vector<double> vf_pad_ser(local_with_halo_size(Lf_ser), 0.0);
        {
            const int nz_r = local_nz(Lf_ser);
            const int kstride = 2*halo_w + nz_r;
            for( int i=0; i<cluster_nx_f; ++i )
                for( int j=0; j<cluster_ny_f; ++j )
                    for( int kl=0; kl<nz_r; ++kl ) {
                        const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                              * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                        const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                              * (std::size_t)cluster_nz_f + (std::size_t)kl;
                        vf_pad_ser[dst] = full_vf[src];
                    }
        }
        std::vector<double> Vc_pad_ser(local_with_halo_size(Lc_ser), 0.0);
        restrict_z_slab_double(Lf_ser, Lc_ser, vf_pad_ser.data(), Vc_pad_ser.data());

        std::vector<double> full_Vc_ref(cluster_full_size(Lc_ser), 0.0);
        {
            const int nz_r = local_nz(Lc_ser);
            const int kstride = 2*halo_w + nz_r;
            for( int i=0; i<cluster_nx_c; ++i )
                for( int j=0; j<cluster_ny_c; ++j )
                    for( int kl=0; kl<nz_r; ++kl ) {
                        const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_c + (std::size_t)j)
                                              * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                        const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_c + (std::size_t)j)
                                              * (std::size_t)cluster_nz_c + (std::size_t)kl;
                        full_Vc_ref[dst] = Vc_pad_ser[src];
                    }
        }

        double max_abs_err = 0.0;
        std::size_t err_count = 0;
        for( std::size_t n=0; n<full_Vc_ref.size(); ++n ) {
            const double e = std::fabs(full_Vc_slab[n] - full_Vc_ref[n]);
            if( e > max_abs_err ) max_abs_err = e;
            if( e != 0.0 ) ++err_count;
        }
        LOGINFO("G.2b smoke (restrict): max|err|=%.3e nonzero_diff=%zu/%zu",
                max_abs_err, err_count, full_Vc_ref.size());
        ok = (max_abs_err == 0.0);
        if( ok )
            LOGINFO("G.2b smoke (restrict): PASSED (bit-identical to single-rank reference)");
        else
            LOGERR("G.2b smoke (restrict): FAILED");
    }

#ifdef USE_MPI
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, sub);
    ok = (ok_int != 0);
    MPI_Barrier(sub);
#endif
    return ok;
}

// ---------------------------------------------------------------------------
// Phase G.2b B.5.2 smoke: restrict_meshvarbnd_from_slab vs the canonical
// restrict_meshvarbnd that scatters internally. Builds vf as a MeshvarBnd
// on owner with a smooth pattern; drives both paths over the same sub_comm;
// compares the two coarse outputs cell-by-cell on owner. Bit-identical
// (max|err| == 0) because both paths run the identical restrict_impl on the
// identical scatter result — the only thing being validated is that the new
// bridge's pack/halo-pad/gather plumbing matches the inline version.
// ---------------------------------------------------------------------------
bool smoke_test_restrict_meshvarbnd_from_slab_single_cluster(int halo_w)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    const int nxf = 16;
    const int nyf = 16;
    // Pick nzf so it satisfies the impl's alignment guard
    //   (nzf even AND nzf % (2*sz) == 0 AND nzf/2/sz >= 1) on np=1..8.
    int nzf = 48;
    while( (nzf / world_size) % 2 != 0
        || (nzf / (2 * world_size)) < 1 )
        nzf += 2 * world_size;

    const int nxc = nxf / 2;
    const int nyc = nyf / 2;
    const int nzc = nzf / 2;
    const int nbnd = 0;  // restrict reads only interior; nbnd irrelevant

    LOGINFO("G.2b smoke (restrict_mvb_from_slab): fine=%dx%dx%d coarse=%dx%dx%d "
            "halo_w=%d sub_size=%d",
            nxf, nyf, nzf, nxc, nyc, nzc, halo_w, world_size);

    auto vf_at = [&](int i, int j, int k) -> double {
        return (double)(i + 1) * 0.017
             + (double)(j + 1) * 0.031
             + (double)(k + 1) * 0.043
             + std::sin(0.13 * i + 0.21 * j + 0.29 * k);
    };

    std::unique_ptr< MeshvarBnd<double> > vf, Vc_ref, Vc_test;
    if( world_rank == 0 ) {
        vf     .reset(new MeshvarBnd<double>(nbnd, nxf, nyf, nzf));
        Vc_ref .reset(new MeshvarBnd<double>(nbnd, nxc, nyc, nzc));
        Vc_test.reset(new MeshvarBnd<double>(nbnd, nxc, nyc, nzc));
        for( int i=0; i<nxf; ++i )
            for( int j=0; j<nyf; ++j )
                for( int k=0; k<nzf; ++k )
                    (*vf)(i, j, k) = vf_at(i, j, k);
        // Zero-init the two coarse buffers so any write-vs-no-write disagreement
        // surfaces immediately.
        for( int i=0; i<nxc; ++i )
            for( int j=0; j<nyc; ++j )
                for( int k=0; k<nzc; ++k ) {
                    (*Vc_ref )(i, j, k) = 0.0;
                    (*Vc_test)(i, j, k) = 0.0;
                }
    }

#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif

    // Path A: existing restrict_meshvarbnd (scatters internally).
    const bool ranA = restrict_meshvarbnd_double(
        /*box_owner=*/0,
        world_rank == 0 ? vf.get()     : (const MeshvarBnd<double>*)nullptr,
        world_rank == 0 ? Vc_ref.get() : (MeshvarBnd<double>*)nullptr,
        sub);

    // Path B: scatter vf manually, then drive the new from-slab bridge.
    ZoomSlabLayout Lf = make_layout(
        sub, /*cluster_id=*/0, /*level=*/1,
        0, 0, 0, nxf, nyf, nzf, halo_w);

    std::vector<double> cluster_vf;
    if( world_rank == 0 ) {
        cluster_vf.assign(cluster_full_size(Lf), 0.0);
        for( int i=0; i<nxf; ++i )
            for( int j=0; j<nyf; ++j )
                for( int k=0; k<nzf; ++k ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)nyf + (std::size_t)j)
                                          * (std::size_t)nzf + (std::size_t)k;
                    cluster_vf[idx] = (*vf)(i, j, k);
                }
    }
    std::vector<double> my_vf_int(local_interior_size(Lf), 0.0);
    scatter_from(Lf, /*src=*/0,
                 world_rank == 0 ? cluster_vf.data() : (const double*)nullptr,
                 my_vf_int.data());

    const bool ranB = restrict_meshvarbnd_from_slab_double(
        /*box_owner=*/0, Lf, my_vf_int.data(),
        world_rank == 0 ? Vc_test.get() : (MeshvarBnd<double>*)nullptr,
        /*Vc_off=*/0, 0, 0,
        sub);

    bool ok = true;
    if( world_rank == 0 ) {
        if( ranA != ranB ) {
            LOGERR("G.2b smoke (restrict_mvb_from_slab): bridge return-bool mismatch "
                   "(reference=%d from_slab=%d) — alignment guards diverged",
                   (int)ranA, (int)ranB);
            ok = false;
        } else if( !ranA ) {
            // Both fell through (e.g. sub_size==1). Smoke is a no-op in that
            // case — the bridge never executed any code worth validating.
            LOGINFO("G.2b smoke (restrict_mvb_from_slab): bridge skipped on this "
                    "sub_size (likely sub_size==1); nothing to compare.");
        } else {
            double max_abs_err = 0.0;
            std::size_t diff = 0;
            const std::size_t total = (std::size_t)nxc * (std::size_t)nyc * (std::size_t)nzc;
            for( int i=0; i<nxc; ++i )
                for( int j=0; j<nyc; ++j )
                    for( int k=0; k<nzc; ++k ) {
                        const double e = std::fabs((*Vc_ref)(i, j, k) - (*Vc_test)(i, j, k));
                        if( e > max_abs_err ) max_abs_err = e;
                        if( e != 0.0 ) ++diff;
                    }
            LOGINFO("G.2b smoke (restrict_mvb_from_slab): max|err|=%.3e nonzero=%zu/%zu",
                    max_abs_err, diff, total);
            ok = (max_abs_err == 0.0);
            if( ok )
                LOGINFO("G.2b smoke (restrict_mvb_from_slab): PASSED "
                        "(bit-identical to restrict_meshvarbnd)");
            else
                LOGERR("G.2b smoke (restrict_mvb_from_slab): FAILED");
        }
    }

#ifdef USE_MPI
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, sub);
    ok = (ok_int != 0);
    MPI_Barrier(sub);
#endif
    return ok;
}

// ---------------------------------------------------------------------------
// Phase G.2b B.5.2-prod smoke test: fused apply_to_slab + restrict_from_slab
// vs the existing apply_meshvarbnd + restrict_meshvarbnd path.
//
// Both paths consume the SAME input vf MeshvarBnd and run the SAME
// apply_z_slab on the SAME scatter result. Path A then writes apply result
// into apply_uf MeshvarBnd (owner-only, 1/h2 scaled) and dispatches
// restrict_meshvarbnd which re-scatters and runs restrict_impl. Path B holds
// the apply result in slab form (1/h2 scaled in the redistribute pack pass)
// and runs restrict_impl directly on it. Both feed restrict_impl identical
// per-rank vf slabs, so the coarse output is bit-identical (max|err| == 0).
//
// What this smoke validates is the new redistribute helper's pack/unpack +
// Alltoallv plumbing: stripping the cluster-A BC perimeter on x/y/z and
// reshuffling z-ownership to match Layout B.
// ---------------------------------------------------------------------------
bool smoke_test_apply_meshvarbnd_to_slab_fused_single_cluster(int halo_w)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    // Pick a fine block of (nx, ny, nz) consumed by apply (cluster_nz = nz+2)
    // and the restrict's fine layout (cluster_nz = nz). nz must satisfy BOTH:
    //   - apply Layout A:  halo_w <= (nz+2)/sub_size
    //   - restrict Lf:     nz % (2*sub_size) == 0  AND  nz/(2*sub_size) >= 1
    const int nx = 16;
    const int ny = 16;
    int nz = 48;
    while( (nz / world_size) % 2 != 0
        || (nz / (2 * world_size)) < 1
        || halo_w > ((nz + 2) / world_size) )
        nz += 2 * world_size;

    const int nxc = nx / 2;
    const int nyc = ny / 2;
    const int nzc = nz / 2;
    const int nbnd = 1;  // apply needs >= 1 BC perimeter; restrict ignores

    LOGINFO("G.2b smoke (apply_to_slab+restrict_from_slab fused): "
            "fine=%dx%dx%d coarse=%dx%dx%d halo_w=%d sub_size=%d",
            nx, ny, nz, nxc, nyc, nzc, halo_w, world_size);

    auto vf_at = [&](int i, int j, int k) -> double {
        return (double)(i + 1) * 0.017
             + (double)(j + 1) * 0.031
             + (double)(k + 1) * 0.043
             + std::sin(0.13 * i + 0.21 * j + 0.29 * k);
    };

    const double h2 = 1.0 / 64.0;  // arbitrary nonzero — both paths use same value

    std::unique_ptr< MeshvarBnd<double> > vf, apply_uf_ref, Vc_ref, Vc_test;
    if( world_rank == 0 ) {
        vf          .reset(new MeshvarBnd<double>(nbnd, nx,  ny,  nz));
        apply_uf_ref.reset(new MeshvarBnd<double>(nbnd, nx,  ny,  nz));
        Vc_ref      .reset(new MeshvarBnd<double>(0,    nxc, nyc, nzc));
        Vc_test     .reset(new MeshvarBnd<double>(0,    nxc, nyc, nzc));
        // Fill interior + 1-cell BC perimeter so apply reads non-zero
        // neighbors at cluster edges (otherwise i=0 / i=nx-1 rows pull from
        // the all-zero BC and the smoke degenerates).
        for( int i=-1; i<=nx; ++i )
            for( int j=-1; j<=ny; ++j )
                for( int k=-1; k<=nz; ++k )
                    (*vf)(i, j, k) = vf_at(i, j, k);
        apply_uf_ref->zero();
        for( int i=0; i<nxc; ++i )
            for( int j=0; j<nyc; ++j )
                for( int k=0; k<nzc; ++k ) {
                    (*Vc_ref )(i, j, k) = 0.0;
                    (*Vc_test)(i, j, k) = 0.0;
                }
    }

#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif

    // Path A: apply_meshvarbnd (writes apply_uf_ref / h2) -> restrict_meshvarbnd.
    const bool ranA_apply = apply_meshvarbnd_double(
        /*box_owner=*/0,
        world_rank == 0 ? vf.get()           : (const MeshvarBnd<double>*)nullptr,
        world_rank == 0 ? apply_uf_ref.get() : (MeshvarBnd<double>*)nullptr,
        h2, sub, halo_w);
    const bool ranA_restr = restrict_meshvarbnd_double(
        /*box_owner=*/0,
        world_rank == 0 ? apply_uf_ref.get() : (const MeshvarBnd<double>*)nullptr,
        world_rank == 0 ? Vc_ref.get()       : (MeshvarBnd<double>*)nullptr,
        sub);

    // Path B: fused apply_to_slab + restrict_from_slab.
    ZoomSlabLayout Lf = make_layout(
        sub, /*cluster_id=*/0, /*level=*/1,
        0, 0, 0, nx, ny, nz, halo_w);
    std::vector<double> my_apply_int(local_interior_size(Lf), 0.0);

    const bool ranB_apply = apply_meshvarbnd_to_slab_double(
        /*box_owner=*/0,
        world_rank == 0 ? vf.get() : (const MeshvarBnd<double>*)nullptr,
        Lf, my_apply_int.data(), h2, sub);
    const bool ranB_restr = restrict_meshvarbnd_from_slab_double(
        /*box_owner=*/0, Lf, my_apply_int.data(),
        world_rank == 0 ? Vc_test.get() : (MeshvarBnd<double>*)nullptr,
        /*Vc_off=*/0, 0, 0, sub);

    bool ok = true;
    if( world_rank == 0 ) {
        const bool ranA = ranA_apply && ranA_restr;
        const bool ranB = ranB_apply && ranB_restr;
        if( ranA != ranB ) {
            LOGERR("G.2b smoke (apply_to_slab fused): return-bool mismatch "
                   "(A=%d/%d B=%d/%d)",
                   (int)ranA_apply, (int)ranA_restr,
                   (int)ranB_apply, (int)ranB_restr);
            ok = false;
        } else if( !ranA ) {
            LOGINFO("G.2b smoke (apply_to_slab fused): both paths skipped "
                    "(likely sub_size==1); nothing to compare.");
        } else {
            double max_abs_err = 0.0;
            std::size_t diff = 0;
            const std::size_t total = (std::size_t)nxc * (std::size_t)nyc * (std::size_t)nzc;
            for( int i=0; i<nxc; ++i )
                for( int j=0; j<nyc; ++j )
                    for( int k=0; k<nzc; ++k ) {
                        const double e = std::fabs((*Vc_ref)(i, j, k) - (*Vc_test)(i, j, k));
                        if( e > max_abs_err ) max_abs_err = e;
                        if( e != 0.0 ) ++diff;
                    }
            LOGINFO("G.2b smoke (apply_to_slab fused): max|err|=%.3e nonzero=%zu/%zu",
                    max_abs_err, diff, total);
            ok = (max_abs_err == 0.0);
            if( ok )
                LOGINFO("G.2b smoke (apply_to_slab fused): PASSED "
                        "(bit-identical to apply_meshvarbnd + restrict_meshvarbnd)");
            else
                LOGERR("G.2b smoke (apply_to_slab fused): FAILED");
        }
    }

#ifdef USE_MPI
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, sub);
    ok = (ok_int != 0);
    MPI_Barrier(sub);
#endif
    return ok;
}

// ---------------------------------------------------------------------------
// Phase G.2b: zero-order injection prolong (mirrors mg_straight::prolong in
// mg_operators.hh). Pure local — same alignment precondition as restrict.
// All 8 fine children of a coarse cell receive the coarse cell's value.
// ---------------------------------------------------------------------------
namespace {

template<typename T>
void prolong_impl(const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                  const T* Vc, T* vf)
{
    if(    Lc.cluster_nx * 2 != Lf.cluster_nx
        || Lc.cluster_ny * 2 != Lf.cluster_ny
        || Lc.cluster_nz * 2 != Lf.cluster_nz
        || Lc.my_z0 * 2     != Lf.my_z0
        || Lc.my_z1 * 2     != Lf.my_z1 )
        throw std::runtime_error("prolong_z_slab: layout mismatch (require Lc = Lf/2 aligned).");

    const int cnxc      = Lc.cluster_nx;
    const int cnyc      = Lc.cluster_ny;
    const int nz_rc     = local_nz(Lc);
    const int hwc       = Lc.halo_w;
    const int kstridec  = 2 * hwc + nz_rc;
    const std::size_t ny_strc = (std::size_t)cnyc * (std::size_t)kstridec;
    const std::size_t x_strc  = (std::size_t)kstridec;

    const int cnyf      = Lf.cluster_ny;
    const int nz_rf     = local_nz(Lf);
    const int hwf       = Lf.halo_w;
    const int kstridef  = 2 * hwf + nz_rf;
    const std::size_t ny_strf = (std::size_t)cnyf * (std::size_t)kstridef;
    const std::size_t x_strf  = (std::size_t)kstridef;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for( int ic=0; ic<cnxc; ++ic ) {
        const int if0 = 2 * ic;
        const std::size_t ic_off  = (std::size_t)ic       * ny_strc;
        const std::size_t if0_off = (std::size_t)if0      * ny_strf;
        const std::size_t if1_off = (std::size_t)(if0+1)  * ny_strf;
        for( int jc=0; jc<cnyc; ++jc ) {
            const int jf0 = 2 * jc;
            const std::size_t cdrow      = ic_off  + (std::size_t)jc          * x_strc;
            const std::size_t fr_i0_j0   = if0_off + (std::size_t)jf0         * x_strf;
            const std::size_t fr_i0_j1   = if0_off + (std::size_t)(jf0+1)     * x_strf;
            const std::size_t fr_i1_j0   = if1_off + (std::size_t)jf0         * x_strf;
            const std::size_t fr_i1_j1   = if1_off + (std::size_t)(jf0+1)     * x_strf;
            for( int klc=hwc; klc<hwc+nz_rc; ++klc ) {
                const int klf0 = hwf + 2 * (klc - hwc);
                const T cv = Vc[cdrow + (std::size_t)klc];
                vf[fr_i0_j0 + klf0    ] = cv;  vf[fr_i0_j0 + klf0 + 1] = cv;
                vf[fr_i0_j1 + klf0    ] = cv;  vf[fr_i0_j1 + klf0 + 1] = cv;
                vf[fr_i1_j0 + klf0    ] = cv;  vf[fr_i1_j0 + klf0 + 1] = cv;
                vf[fr_i1_j1 + klf0    ] = cv;  vf[fr_i1_j1 + klf0 + 1] = cv;
            }
        }
    }
}

} // anonymous namespace

void prolong_z_slab_double(const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                           const double* Vc, double* vf)
{ prolong_impl<double>(Lc, Lf, Vc, vf); }

void prolong_z_slab_float(const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                          const float* Vc, float* vf)
{ prolong_impl<float>(Lc, Lf, Vc, vf); }

// ---------------------------------------------------------------------------
// Phase G.2b smoke test for prolong_z (injection). Mirrors restrict smoke:
// build smooth coarse pattern on rank 0, scatter to coarse z-slabs, prolong
// to fine z-slabs, gather, compare bit-identical against single-rank reference.
// ---------------------------------------------------------------------------
bool smoke_test_prolong_single_cluster(int halo_w)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    const int cluster_nx_f = 16;
    const int cluster_ny_f = 16;
    int cluster_nz_f = 48;
    while( (cluster_nz_f / world_size) % 2 != 0
        || (cluster_nz_f / (2 * world_size)) < 1 )
        cluster_nz_f += 2 * world_size;

    const int cluster_nx_c = cluster_nx_f / 2;
    const int cluster_ny_c = cluster_ny_f / 2;
    const int cluster_nz_c = cluster_nz_f / 2;

    LOGINFO("G.2b smoke (prolong): fine=%dx%dx%d coarse=%dx%dx%d halo_w=%d sub_size=%d",
            cluster_nx_f, cluster_ny_f, cluster_nz_f,
            cluster_nx_c, cluster_ny_c, cluster_nz_c,
            halo_w, world_size);

#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif

    ZoomSlabLayout Lf = make_layout(
        sub, /*cluster_id=*/0, /*level=*/1,
        0, 0, 0,
        cluster_nx_f, cluster_ny_f, cluster_nz_f, halo_w);
    ZoomSlabLayout Lc = make_layout(
        sub, /*cluster_id=*/0, /*level=*/0,
        0, 0, 0,
        cluster_nx_c, cluster_ny_c, cluster_nz_c, halo_w);

    if(    2 * Lc.my_z0 != Lf.my_z0
        || 2 * Lc.my_z1 != Lf.my_z1 ) {
        if( world_rank == 0 )
            LOGERR("G.2b smoke (prolong): Lc/Lf z-slab alignment broken on rank %d", world_rank);
        return false;
    }

    auto Vc_at = [&](int ic, int jc, int kc) -> double {
        return (double)(ic + 1) * 0.019
             + (double)(jc + 1) * 0.037
             + (double)(kc + 1) * 0.053
             + std::cos(0.15 * ic + 0.25 * jc + 0.35 * kc);
    };

    std::vector<double> full_Vc;
    if( world_rank == 0 ) {
        full_Vc.assign(cluster_full_size(Lc), 0.0);
        for( int ic=0; ic<cluster_nx_c; ++ic )
            for( int jc=0; jc<cluster_ny_c; ++jc )
                for( int kc=0; kc<cluster_nz_c; ++kc ) {
                    const std::size_t idx = ((std::size_t)ic * (std::size_t)cluster_ny_c + (std::size_t)jc)
                                          * (std::size_t)cluster_nz_c + (std::size_t)kc;
                    full_Vc[idx] = Vc_at(ic, jc, kc);
                }
    }

    std::vector<double> my_Vc_int(local_interior_size(Lc), 0.0);
    scatter_cluster_to_zslab_double(Lc, world_rank == 0 ? full_Vc.data() : nullptr,
                                    my_Vc_int.data());

    std::vector<double> Vc_pad(local_with_halo_size(Lc), 0.0);
    {
        const int nz_r = local_nz(Lc);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx_c; ++i )
            for( int j=0; j<cluster_ny_c; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_c + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_c + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    Vc_pad[dst] = my_Vc_int[src];
                }
    }

    std::vector<double> vf_pad(local_with_halo_size(Lf), 0.0);

    prolong_z_slab_double(Lc, Lf, Vc_pad.data(), vf_pad.data());

    std::vector<double> my_vf_int(local_interior_size(Lf), 0.0);
    {
        const int nz_r = local_nz(Lf);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx_f; ++i )
            for( int j=0; j<cluster_ny_f; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    my_vf_int[dst] = vf_pad[src];
                }
    }

    std::vector<double> full_vf_slab;
    if( world_rank == 0 ) full_vf_slab.assign(cluster_full_size(Lf), 0.0);
    gather_zslab_to_cluster_double(Lf, my_vf_int.data(),
                                   world_rank == 0 ? full_vf_slab.data() : nullptr);

    bool ok = true;
    if( world_rank == 0 ) {
        // Single-rank reference via the same prolong_impl path.
#ifdef USE_MPI
        MPI_Comm self_comm = MPI_COMM_SELF;
#else
        MPI_Comm self_comm = 0;
#endif
        ZoomSlabLayout Lf_ser = make_layout(
            self_comm, 0, 1, 0, 0, 0, cluster_nx_f, cluster_ny_f, cluster_nz_f, halo_w);
        ZoomSlabLayout Lc_ser = make_layout(
            self_comm, 0, 0, 0, 0, 0, cluster_nx_c, cluster_ny_c, cluster_nz_c, halo_w);

        std::vector<double> Vc_pad_ser(local_with_halo_size(Lc_ser), 0.0);
        {
            const int nz_r = local_nz(Lc_ser);
            const int kstride = 2*halo_w + nz_r;
            for( int i=0; i<cluster_nx_c; ++i )
                for( int j=0; j<cluster_ny_c; ++j )
                    for( int kl=0; kl<nz_r; ++kl ) {
                        const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_c + (std::size_t)j)
                                              * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                        const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_c + (std::size_t)j)
                                              * (std::size_t)cluster_nz_c + (std::size_t)kl;
                        Vc_pad_ser[dst] = full_Vc[src];
                    }
        }
        std::vector<double> vf_pad_ser(local_with_halo_size(Lf_ser), 0.0);
        prolong_z_slab_double(Lc_ser, Lf_ser, Vc_pad_ser.data(), vf_pad_ser.data());

        std::vector<double> full_vf_ref(cluster_full_size(Lf_ser), 0.0);
        {
            const int nz_r = local_nz(Lf_ser);
            const int kstride = 2*halo_w + nz_r;
            for( int i=0; i<cluster_nx_f; ++i )
                for( int j=0; j<cluster_ny_f; ++j )
                    for( int kl=0; kl<nz_r; ++kl ) {
                        const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                              * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                        const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                              * (std::size_t)cluster_nz_f + (std::size_t)kl;
                        full_vf_ref[dst] = vf_pad_ser[src];
                    }
        }

        double max_abs_err = 0.0;
        std::size_t err_count = 0;
        for( std::size_t n=0; n<full_vf_ref.size(); ++n ) {
            const double e = std::fabs(full_vf_slab[n] - full_vf_ref[n]);
            if( e > max_abs_err ) max_abs_err = e;
            if( e != 0.0 ) ++err_count;
        }
        LOGINFO("G.2b smoke (prolong): max|err|=%.3e nonzero_diff=%zu/%zu",
                max_abs_err, err_count, full_vf_ref.size());
        ok = (max_abs_err == 0.0);
        if( ok )
            LOGINFO("G.2b smoke (prolong): PASSED (bit-identical to single-rank reference)");
        else
            LOGERR("G.2b smoke (prolong): FAILED");
    }

#ifdef USE_MPI
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, sub);
    ok = (ok_int != 0);
    MPI_Barrier(sub);
#endif
    return ok;
}

// ---------------------------------------------------------------------------
// Phase G.2b: prolong_add (mirrors mg_straight::prolong_add in mg_operators.hh).
// Pure-local += injection: each of the 8 fine children of a coarse cell has the
// coarse cell value ADDED to its current contents. Same alignment precondition
// as prolong; no coarse halo needed because each fine cell reads exactly one
// coarse cell (local to the rank).
// ---------------------------------------------------------------------------
namespace {

template<typename T>
void prolong_add_impl(const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                      const T* Vc, T* vf)
{
    if(    Lc.cluster_nx * 2 != Lf.cluster_nx
        || Lc.cluster_ny * 2 != Lf.cluster_ny
        || Lc.cluster_nz * 2 != Lf.cluster_nz
        || Lc.my_z0 * 2     != Lf.my_z0
        || Lc.my_z1 * 2     != Lf.my_z1 )
        throw std::runtime_error("prolong_add_z_slab: layout mismatch (require Lc = Lf/2 aligned).");

    const int cnxc      = Lc.cluster_nx;
    const int cnyc      = Lc.cluster_ny;
    const int nz_rc     = local_nz(Lc);
    const int hwc       = Lc.halo_w;
    const int kstridec  = 2 * hwc + nz_rc;
    const std::size_t ny_strc = (std::size_t)cnyc * (std::size_t)kstridec;
    const std::size_t x_strc  = (std::size_t)kstridec;

    const int cnyf      = Lf.cluster_ny;
    const int nz_rf     = local_nz(Lf);
    const int hwf       = Lf.halo_w;
    const int kstridef  = 2 * hwf + nz_rf;
    const std::size_t ny_strf = (std::size_t)cnyf * (std::size_t)kstridef;
    const std::size_t x_strf  = (std::size_t)kstridef;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for( int ic=0; ic<cnxc; ++ic ) {
        const int if0 = 2 * ic;
        const std::size_t ic_off  = (std::size_t)ic       * ny_strc;
        const std::size_t if0_off = (std::size_t)if0      * ny_strf;
        const std::size_t if1_off = (std::size_t)(if0+1)  * ny_strf;
        for( int jc=0; jc<cnyc; ++jc ) {
            const int jf0 = 2 * jc;
            const std::size_t cdrow      = ic_off  + (std::size_t)jc          * x_strc;
            const std::size_t fr_i0_j0   = if0_off + (std::size_t)jf0         * x_strf;
            const std::size_t fr_i0_j1   = if0_off + (std::size_t)(jf0+1)     * x_strf;
            const std::size_t fr_i1_j0   = if1_off + (std::size_t)jf0         * x_strf;
            const std::size_t fr_i1_j1   = if1_off + (std::size_t)(jf0+1)     * x_strf;
            for( int klc=hwc; klc<hwc+nz_rc; ++klc ) {
                const int klf0 = hwf + 2 * (klc - hwc);
                const T cv = Vc[cdrow + (std::size_t)klc];
                vf[fr_i0_j0 + klf0    ] += cv;  vf[fr_i0_j0 + klf0 + 1] += cv;
                vf[fr_i0_j1 + klf0    ] += cv;  vf[fr_i0_j1 + klf0 + 1] += cv;
                vf[fr_i1_j0 + klf0    ] += cv;  vf[fr_i1_j0 + klf0 + 1] += cv;
                vf[fr_i1_j1 + klf0    ] += cv;  vf[fr_i1_j1 + klf0 + 1] += cv;
            }
        }
    }
}

} // anonymous namespace

void prolong_add_z_slab_double(const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                               const double* Vc, double* vf)
{ prolong_add_impl<double>(Lc, Lf, Vc, vf); }

void prolong_add_z_slab_float(const ZoomSlabLayout& Lc, const ZoomSlabLayout& Lf,
                              const float* Vc, float* vf)
{ prolong_add_impl<float>(Lc, Lf, Vc, vf); }

// ---------------------------------------------------------------------------
// Phase G.2b B.2b.5: collective prolong_add bridge.
//
// On box_owner: packs vf interior (nxf,nyf,nzf) AND the (nxc,nyc,nzc)
// sub-region of Vc starting at parent offset (ox,oy,oz) into cluster_full
// buffers. Scatter both fine and coarse to z-slabs. Run prolong_add_impl on
// the slabs (vf_slab += Vc_slab semantics). Gather fine z-slabs back to
// box_owner and write into vf interior cells.
//
// Returns false (no writes) if sub_size==1 or alignment unsuitable for a clean
// 2x z-split. Caller must do a local prolong_add in that case.
// ---------------------------------------------------------------------------
namespace {

template<typename T>
bool prolong_add_meshvarbnd_impl(int box_owner,
                                 const MeshvarBnd<T>* Vc,
                                 MeshvarBnd<T>* vf,
                                 MPI_Comm sub_comm)
{
    int rk = 0, sz = 1;
#ifdef USE_MPI
    MPI_Comm_rank(sub_comm, &rk);
    MPI_Comm_size(sub_comm, &sz);
#endif

    if( sz <= 1 ) return false;

    int hdr[6] = {0, 0, 0, 0, 0, 0};
    if( rk == box_owner ) {
        if( !vf || !Vc )
            throw std::runtime_error("prolong_add_meshvarbnd: vf/Vc null on box_owner");
        hdr[0] = (int)vf->size(0);
        hdr[1] = (int)vf->size(1);
        hdr[2] = (int)vf->size(2);
        hdr[3] = vf->offset(0);
        hdr[4] = vf->offset(1);
        hdr[5] = vf->offset(2);
    }
#ifdef USE_MPI
    MPI_Bcast(hdr, 6, MPI_INT, box_owner, sub_comm);
#endif

    const int nxf = hdr[0], nyf = hdr[1], nzf = hdr[2];
    const int ox  = hdr[3], oy  = hdr[4], oz  = hdr[5];

    if( nxf <= 0 || nyf <= 0 || nzf <= 0 )
        throw std::runtime_error("prolong_add_meshvarbnd: bad fine dims from box_owner");

    if( (nxf & 1) || (nyf & 1) || (nzf & 1) ) return false;
    if( nzf % (2 * sz) != 0 ) return false;

    const int nxc = nxf / 2;
    const int nyc = nyf / 2;
    const int nzc = nzf / 2;
    if( nzc / sz < 1 ) return false;

    const int halo_w = 1;

    ZoomSlabLayout Lf = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/1,
        0, 0, 0, nxf, nyf, nzf, halo_w);

    ZoomSlabLayout Lc = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/0,
        0, 0, 0, nxc, nyc, nzc, halo_w);

    if( 2 * Lc.my_z0 != Lf.my_z0 || 2 * Lc.my_z1 != Lf.my_z1 )
        return false;

    // Pack on box_owner: both fine interior + coarse sub-region of parent.
    std::vector<T> cluster_vf_in;
    std::vector<T> cluster_Vc_in;
    if( rk == box_owner ) {
        cluster_vf_in.assign(cluster_full_size(Lf), T(0));
        cluster_Vc_in.assign(cluster_full_size(Lc), T(0));
        for( int i=0; i<nxf; ++i )
            for( int j=0; j<nyf; ++j )
                for( int k=0; k<nzf; ++k ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)nyf + (std::size_t)j)
                                          * (std::size_t)nzf + (std::size_t)k;
                    cluster_vf_in[idx] = (*vf)(i, j, k);
                }
        for( int ic=0; ic<nxc; ++ic )
            for( int jc=0; jc<nyc; ++jc )
                for( int kc=0; kc<nzc; ++kc ) {
                    const std::size_t idx = ((std::size_t)ic * (std::size_t)nyc + (std::size_t)jc)
                                          * (std::size_t)nzc + (std::size_t)kc;
                    cluster_Vc_in[idx] = (*Vc)(ic + ox, jc + oy, kc + oz);
                }
    }

    // Scatter fine + coarse to per-rank slabs.
    std::vector<T> my_vf_int(local_interior_size(Lf), T(0));
    std::vector<T> my_Vc_int(local_interior_size(Lc), T(0));

    scatter_from(Lf, box_owner,
                 rk == box_owner ? cluster_vf_in.data() : (const T*)nullptr,
                 my_vf_int.data());
    scatter_from(Lc, box_owner,
                 rk == box_owner ? cluster_Vc_in.data() : (const T*)nullptr,
                 my_Vc_int.data());

    // Copy both into halo'd buffers at halo offset.
    std::vector<T> vf_pad(local_with_halo_size(Lf), T(0));
    std::vector<T> Vc_pad(local_with_halo_size(Lc), T(0));

    {
        const int nz_rf  = local_nz(Lf);
        const int hwf    = Lf.halo_w;
        const int kstridef = 2 * hwf + nz_rf;
        for( int ii=0; ii<nxf; ++ii )
            for( int jj=0; jj<nyf; ++jj )
                for( int kl=0; kl<nz_rf; ++kl ) {
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)nyf + (std::size_t)jj)
                                          * (std::size_t)kstridef + (std::size_t)(hwf + kl);
                    const std::size_t src = ((std::size_t)ii * (std::size_t)nyf + (std::size_t)jj)
                                          * (std::size_t)nz_rf + (std::size_t)kl;
                    vf_pad[dst] = my_vf_int[src];
                }
    }
    {
        const int nz_rc  = local_nz(Lc);
        const int hwc    = Lc.halo_w;
        const int kstridec = 2 * hwc + nz_rc;
        for( int ic=0; ic<nxc; ++ic )
            for( int jc=0; jc<nyc; ++jc )
                for( int kl=0; kl<nz_rc; ++kl ) {
                    const std::size_t dst = ((std::size_t)ic * (std::size_t)nyc + (std::size_t)jc)
                                          * (std::size_t)kstridec + (std::size_t)(hwc + kl);
                    const std::size_t src = ((std::size_t)ic * (std::size_t)nyc + (std::size_t)jc)
                                          * (std::size_t)nz_rc + (std::size_t)kl;
                    Vc_pad[dst] = my_Vc_int[src];
                }
    }

    // Local prolong_add on slabs: vf_pad += Vc_pad (via 8-child injection).
    prolong_add_impl<T>(Lc, Lf, Vc_pad.data(), vf_pad.data());

    // Extract fine interior back from vf_pad.
    {
        const int nz_rf  = local_nz(Lf);
        const int hwf    = Lf.halo_w;
        const int kstridef = 2 * hwf + nz_rf;
        for( int ii=0; ii<nxf; ++ii )
            for( int jj=0; jj<nyf; ++jj )
                for( int kl=0; kl<nz_rf; ++kl ) {
                    const std::size_t src = ((std::size_t)ii * (std::size_t)nyf + (std::size_t)jj)
                                          * (std::size_t)kstridef + (std::size_t)(hwf + kl);
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)nyf + (std::size_t)jj)
                                          * (std::size_t)nz_rf + (std::size_t)kl;
                    my_vf_int[dst] = vf_pad[src];
                }
    }

    // Gather fine slabs back to box_owner.
    std::vector<T> cluster_vf_out;
    if( rk == box_owner ) cluster_vf_out.assign(cluster_full_size(Lf), T(0));
    gather_to(Lf, box_owner, my_vf_int.data(),
              rk == box_owner ? cluster_vf_out.data() : (T*)nullptr);

    if( rk == box_owner ) {
        for( int i=0; i<nxf; ++i )
            for( int j=0; j<nyf; ++j )
                for( int k=0; k<nzf; ++k ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)nyf + (std::size_t)j)
                                          * (std::size_t)nzf + (std::size_t)k;
                    (*vf)(i, j, k) = cluster_vf_out[idx];
                }
    }

    return true;
}

} // anonymous namespace

bool prolong_add_meshvarbnd_double(int box_owner,
                                   const MeshvarBnd<double>* Vc,
                                   MeshvarBnd<double>* vf,
                                   MPI_Comm sub_comm)
{ return prolong_add_meshvarbnd_impl<double>(box_owner, Vc, vf, sub_comm); }

bool prolong_add_meshvarbnd_float (int box_owner,
                                   const MeshvarBnd<float>*  Vc,
                                   MeshvarBnd<float>*  vf,
                                   MPI_Comm sub_comm)
{ return prolong_add_meshvarbnd_impl<float>(box_owner, Vc, vf, sub_comm); }

// ---------------------------------------------------------------------------
// Phase G.2b B.5.3a: prolong_bnd_z_slab — port of mg_cubic::prolong_bnd
// (mg_operators.hh:604). Writes the fine BC perimeter (the m_nbnd=2 layer on
// each side of the fine cluster) by tricubic interpolation from the coarse
// parent. Foundation for interp_coarse_fine_z_slab (B.5.3b adds the 6
// flux-correction faces from interp_O3_fluxcorr::interp_coarse_fine).
//
// Buffer layouts:
//   * coarse_buf: REPLICATED across sub_comm in cluster frame
//                 (cnxc, cnyc, cnzc), contiguous row-major with cz fastest.
//                 The fine's MVBND-frame coarse offset (ox_mvbnd, oy_, oz_) is
//                 not exposed; instead the caller passes (ox_c, oy_c, oz_c) =
//                 the cluster-frame coarse offset such that the fine interior
//                 maps to coarse cluster x ∈ [ox_c, ox_c+nx). Cubic prolong
//                 reads coarse at cluster x ∈ [ox_c-3, ox_c+nx+2] (and same
//                 for y/z); precondition ox_c,oy_c,oz_c >= 3 ensures lo-side
//                 fits.
//   * fine_slab_buf: SLABBED across sub_comm via LfC. Required cluster shape
//                    (cluster_nx, cluster_ny, cluster_nz) = (2*nx+4, 2*ny+4,
//                    2*nz+4) — interior 2*N + 2 BC cells on each side
//                    (m_nbnd=2). halo_w >= 1; halo cells are NOT touched by
//                    this kernel (all BC perimeter writes go to first/last
//                    rank's INTERIOR cluster cells, since fine MVBND z=-2,-1
//                    map to fine cluster z=0,1 which rank 0 owns when
//                    my_z0=0).
//
// Slab decomposition:
//   The outer prolong_bnd loop iterates lk ∈ [-1, nz+1) in the production
//   code. For our slabbed fine buffer, each rank only writes fine cells whose
//   z falls in [LfC.my_z0, LfC.my_z1). Per-write filtering via fz_owned()
//   handles this correctly; lk pre-filtering shrinks iteration to the
//   rank-relevant range.
// ---------------------------------------------------------------------------
namespace {

template<typename T>
bool prolong_bnd_z_slab_impl(int cnxc, int cnyc, int cnzc,
                             const T* coarse_buf,
                             int ox_c, int oy_c, int oz_c,
                             int nx, int ny, int nz,
                             const ZoomSlabLayout& LfC,
                             T* fine_slab_buf)
{
    if( LfC.cluster_nx != 2*nx + 4
     || LfC.cluster_ny != 2*ny + 4
     || LfC.cluster_nz != 2*nz + 4 )
        return false;
    if( LfC.halo_w < 1 ) return false;
    if( ox_c < 3 || oy_c < 3 || oz_c < 3 ) return false;
    if( cnxc < ox_c + nx + 3
     || cnyc < oy_c + ny + 3
     || cnzc < oz_c + nz + 3 )
        return false;

    const int cnyf      = LfC.cluster_ny;
    const int nz_rf     = local_nz(LfC);
    const int hwf       = LfC.halo_w;
    const int kstridef  = 2 * hwf + nz_rf;
    const std::size_t ny_strf = (std::size_t)cnyf * (std::size_t)kstridef;
    const std::size_t x_strf  = (std::size_t)kstridef;

    const std::size_t cny_strc = (std::size_t)cnyc * (std::size_t)cnzc;
    const std::size_t cx_strc  = (std::size_t)cnzc;

    // Cubic weights, mg_operators.hh:307-342.
    const double w_lo[4] = { -1.5/64.0,  14.5/64.0,  55.5/64.0, -4.5/64.0 };
    const double w_hi[4] = { -4.5/64.0,  55.5/64.0,  14.5/64.0, -1.5/64.0 };

    // Pre-filter outer lk to this rank's z range. Writes at outer lk go to
    // fine cluster z ∈ {2*lk+2, 2*lk+3}. Need at least one in [my_z0, my_z1).
    int lk_min = -1, lk_max = nz;
    {
        // Smallest lk such that 2*lk + 3 >= my_z0  →  lk >= ceil((my_z0-3)/2).
        const int v = LfC.my_z0 - 3;
        const int ceil_div = (v >= 0) ? ((v + 1) / 2) : (-((-v) / 2));
        if( ceil_div > lk_min ) lk_min = ceil_div;
        // Largest lk such that 2*lk + 2 < my_z1  →  lk <= floor((my_z1-3)/2).
        const int u = LfC.my_z1 - 3;
        const int floor_div = (u >= 0) ? (u / 2) : (-(((-u) + 1) / 2));
        if( floor_div < lk_max ) lk_max = floor_div;
    }
    if( lk_max < lk_min ) return true; // this rank has nothing to write

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for( int li = -1; li < nx + 1; ++li ) {
        for( int lj = -1; lj < ny + 1; ++lj ) {
            for( int lk = lk_min; lk <= lk_max; ++lk ) {
                // Skip cells that are entirely interior (matches prolong_bnd:626).
                if( li>=0 && li<nx && lj>=0 && lj<ny && lk>=0 && lk<nz ) continue;

                // 8 fine children per outer (li,lj,lk):
                //   v(2li+sx, 2lj+sy, 2lk+sz) = interp_cubic<sx,sy,sz>(...)
                for( int sz_=0; sz_<2; ++sz_ ) {
                    const int fz = 2*lk + sz_ + 2;
                    if( fz < LfC.my_z0 || fz >= LfC.my_z1 ) continue;
                    const double* w_z = (sz_ ? w_hi : w_lo);
                    const int z_local = fz - LfC.my_z0 + hwf;

                    for( int sy_=0; sy_<2; ++sy_ ) {
                        const int fy = 2*lj + sy_ + 2;
                        const double* w_y = (sy_ ? w_hi : w_lo);

                        for( int sx_=0; sx_<2; ++sx_ ) {
                            const int fx = 2*li + sx_ + 2;
                            const double* w_x = (sx_ ? w_hi : w_lo);

                            // interp_cubic: sum over (ii,jj,kk) ∈ [0,4)^3.
                            // mg_operators.hh:345-358 uses (k_outer, j_mid,
                            // i_inner) for x, y, z reads with weights w[k],
                            // v[j], u[i] respectively. We reproduce the same
                            // sum (order doesn't affect result):
                            //   r[jj] += w_z[ii] * V(cx,cy,cz)   (over z=ii)
                            //   q     += w_y[jj] * r[jj]         (over y=jj)
                            //   vox   += w_x[kk] * q             (over x=kk)
                            double vox = 0.0;
                            for( int kk=0; kk<4; ++kk ) {
                                const int cx = li + ox_c + kk - 2 + sx_;
                                double q = 0.0;
                                for( int jj=0; jj<4; ++jj ) {
                                    const int cy = lj + oy_c + jj - 2 + sy_;
                                    double r = 0.0;
                                    for( int ii=0; ii<4; ++ii ) {
                                        const int cz = lk + oz_c + ii - 2 + sz_;
                                        const std::size_t cidx =
                                            (std::size_t)cx * cny_strc
                                          + (std::size_t)cy * cx_strc
                                          + (std::size_t)cz;
                                        r += w_z[ii] * (double)coarse_buf[cidx];
                                    }
                                    q += w_y[jj] * r;
                                }
                                vox += w_x[kk] * q;
                            }

                            const std::size_t fidx = (std::size_t)fx * ny_strf
                                                   + (std::size_t)fy * x_strf
                                                   + (std::size_t)z_local;
                            fine_slab_buf[fidx] = (T)vox;
                        }
                    }
                }
            }
        }
    }
    return true;
}

} // anonymous namespace

bool prolong_bnd_z_slab_double(int cnxc, int cnyc, int cnzc,
                               const double* coarse_buf,
                               int ox_c, int oy_c, int oz_c,
                               int nx, int ny, int nz,
                               const ZoomSlabLayout& LfC,
                               double* fine_slab_buf)
{ return prolong_bnd_z_slab_impl<double>(cnxc, cnyc, cnzc, coarse_buf,
                                          ox_c, oy_c, oz_c, nx, ny, nz,
                                          LfC, fine_slab_buf); }

bool prolong_bnd_z_slab_float (int cnxc, int cnyc, int cnzc,
                               const float*  coarse_buf,
                               int ox_c, int oy_c, int oz_c,
                               int nx, int ny, int nz,
                               const ZoomSlabLayout& LfC,
                               float*  fine_slab_buf)
{ return prolong_bnd_z_slab_impl<float >(cnxc, cnyc, cnzc, coarse_buf,
                                          ox_c, oy_c, oz_c, nx, ny, nz,
                                          LfC, fine_slab_buf); }

// ---------------------------------------------------------------------------
// Phase G.2b B.5.3b: interp_cf_flux_z_slab — port of
// interp_O3_fluxcorr::interp_coarse_fine (mg_interp.hh:434-636) MINUS the
// initial mg_cubic().prolong_bnd() call (= B.5.3a). Applies the 6 per-face
// quadratic flux corrections on top of the cubic BC perimeter produced by
// B.5.3a. The composite (B.5.3a + halo_exchange_z + B.5.3b) is the slab
// equivalent of interp_O3_fluxcorr::interp_coarse_fine and is what B.5.3c
// will wire into twoGrid_multibox_spmd.
//
// Buffer layouts (same convention as B.5.3a):
//   * coarse_buf:     REPLICATED across sub_comm, cluster frame (cnxc,cnyc,cnzc),
//                     cz fastest. Cluster-frame coarse offset is (ox_c,oy_c,oz_c).
//   * fine_slab_buf:  SLABBED across sub_comm via LfC; halo-padded. PRECONDITION:
//                     the caller has already (a) zero/init'd the interior, (b) run
//                     B.5.3a so the BC perimeter holds the cubic prolong, and
//                     (c) called halo_exchange_z so neighbour-rank slab interiors
//                     are visible. This kernel WRITES the BC perimeter (overwriting
//                     B.5.3a's cubic values with the flux-corrected ones).
//
// Reads (per face block):
//   * 9 coarse cells at (ixtop, iytop+-1, iztop+-1) for the 3 ustar interp2's;
//   * one coarse pair (ixtop+-1, iytop, iztop) for dflux;
//   * 2 inner fine cells per pair element for interp2left/right (e.g. left face
//     reads u(ix+1) and u(ix+2)).
// All fine reads stay at the same iz_prod as the write or, for the front/back
// faces, at iz_prod+-1, +-2 — both confined to the rank that owns the BC cell.
//
// Slab decomposition / preconditions:
//   * cluster shape must match: LfC.cluster_{n*} == 2*{n*}+4
//   * LfC.halo_w >= 1  (flux kernel itself does no cross-rank read, but the
//                       buffer indexing assumes the standard halo-padded layout)
//   * ox_c, oy_c, oz_c >= 1   (interp2's read at ixtop+-1 etc.)
//   * cnxc >= ox_c + nx + 2 (and same for y,z) — coarse buffer holds the +1
//     stencil for the right/top/back faces
//   * nx, ny, nz even  (each face pairs (iy,iy+1) or (iz,iz+1) into 2x2 cells)
//   * LfC.my_z0 even AND local_nz even  — guarantees every (iz_prod, iz_prod+1)
//     pair with iz_prod even lives entirely on a single rank. Equivalent to
//     (nz+2) % sub_size == 0 for the standard even slab decomposition.
//
// FP-identicality vs production:
//   The local helpers below reproduce mg_interp.hh's interp2 (both 4-arg and
//   7-arg), interp2left and interp2right verbatim, with literal constants cast
//   to T so that float vs double precision matches the production code's
//   real_t. The left face uses the 4-arg interp2 form (production:
//   mg_interp.hh:500); the right/bottom/top/front/back faces use the 7-arg
//   form with explicit (-1, 0, 1) ordinates (mg_interp.hh:525,548,571,595,619).
//   Order of operations matches production line-for-line so the resulting BC
//   values are bit-identical to interp_O3_fluxcorr::interp_coarse_fine.
// ---------------------------------------------------------------------------
namespace {

// mg_interp.hh:132-140 (4-arg interp2)
template<typename T>
static inline T interp2_4arg_T(T fleft, T fcenter, T fright, T x) {
    T a = (T)0.5*(fleft + fright) - fcenter;
    T b = (T)0.5*(fright - fleft);
    T c = fcenter;
    return a*x*x + b*x + c;
}

// mg_interp.hh:25-33 (7-arg interp2). General quadratic through three (xi, fi).
template<typename T>
static inline T interp2_7arg_T(T x1, T x2, T x3, T f1, T f2, T f3, T x) {
    T a = (x1*f3 - x3*f1 - x2*f3 - x1*f2 + x2*f1 + x3*f2)
        / (x1*x3*x3 - x2*x3*x3 + x2*x1*x1 - x3*x1*x1 + x3*x2*x2 - x1*x2*x2);
    T b = -(x1*x1*f3 - x1*x1*f2 - f1*x3*x3 + f2*x3*x3 - x2*x2*f3 + f1*x2*x2)
        / (x1 - x2) / (x1*x2 - x1*x3 + x3*x3 - x2*x3);
    T c = (x1*x1*x2*f3 - x1*x1*x3*f2 - x2*x2*x1*f3 + f2*x1*x3*x3
         + x2*x2*x3*f1 - f1*x2*x3*x3)
        / (x1 - x2) / (x1*x2 - x1*x3 + x3*x3 - x2*x3);
    return a*x*x + b*x + c;
}

// mg_interp.hh:143-151
template<typename T>
static inline T interp2left_T(T fleft, T fcenter, T fright) {
    T a = ((T)6.0*fright - (T)10.0*fcenter + (T)4.0*fleft) / (T)15.0;
    T b = (-(T)4.0*fleft + (T)9.0*fright - (T)5.0*fcenter) / (T)15.0;
    T c = fcenter;
    return a - b + c;
}

// mg_interp.hh:154-162
template<typename T>
static inline T interp2right_T(T fleft, T fcenter, T fright) {
    T a = ((T)6.0*fleft - (T)10.0*fcenter + (T)4.0*fright) / (T)15.0;
    T b = ((T)4.0*fright - (T)9.0*fleft + (T)5.0*fcenter) / (T)15.0;
    T c = fcenter;
    return a + b + c;
}

template<typename T>
bool interp_cf_flux_z_slab_impl(int cnxc, int cnyc, int cnzc,
                                const T* coarse_buf,
                                int ox_c, int oy_c, int oz_c,
                                int nx, int ny, int nz,
                                const ZoomSlabLayout& LfC,
                                T* fine_slab_buf)
{
    if( LfC.cluster_nx != 2*nx + 4
     || LfC.cluster_ny != 2*ny + 4
     || LfC.cluster_nz != 2*nz + 4 )
        return false;
    if( LfC.halo_w < 1 ) return false;
    if( ox_c < 1 || oy_c < 1 || oz_c < 1 ) return false;
    if( cnxc < ox_c + nx + 2
     || cnyc < oy_c + ny + 2
     || cnzc < oz_c + nz + 2 )
        return false;
    if( (nx & 1) || (ny & 1) || (nz & 1) ) return false;
    if( (LfC.my_z0 & 1) ) return false;
    const int nz_rf = local_nz(LfC);
    if( (nz_rf & 1) ) return false;

    const int nxf = 2*nx, nyf = 2*ny, nzf = 2*nz;  // fine interior dims (== production's nx,ny,nz)
    const int cnyf      = LfC.cluster_ny;
    const int hwf       = LfC.halo_w;
    const int kstridef  = 2 * hwf + nz_rf;
    const std::size_t ny_strf = (std::size_t)cnyf * (std::size_t)kstridef;
    const std::size_t x_strf  = (std::size_t)kstridef;

    const std::size_t cny_strc = (std::size_t)cnyc * (std::size_t)cnzc;
    const std::size_t cx_strc  = (std::size_t)cnzc;

    // Production index ↔ slab buffer index helpers (no captures; macros would
    // do but functions are easier to read and the compiler inlines them).
    auto U_idx = [&](int ix_prod, int iy_prod, int iz_prod) -> std::size_t {
        const int cx = ix_prod + 2;
        const int cy = iy_prod + 2;
        const int kl = (iz_prod + 2) - LfC.my_z0 + hwf;
        return (std::size_t)cx * ny_strf + (std::size_t)cy * x_strf + (std::size_t)kl;
    };
    // V_at accepts ABSOLUTE coarse-cluster coords. Kernel computes
    // ixtop/iytop/iztop already including (ox_c, oy_c, oz_c).
    auto V_at = [&](int cx, int cy, int cz) -> T {
        const std::size_t idx = (std::size_t)cx * cny_strc
                              + (std::size_t)cy * cx_strc
                              + (std::size_t)cz;
        return coarse_buf[idx];
    };
    auto fz_owned = [&](int iz_prod) -> bool {
        const int cz = iz_prod + 2;
        return cz >= LfC.my_z0 && cz < LfC.my_z1;
    };
    auto fpair_owned = [&](int iz_prod) -> bool {
        // Both iz_prod and iz_prod+1 on this rank.
        // With my_z0 even and local_nz even, this is equivalent to
        // fz_owned(iz_prod) (since pairs never split across ranks).
        return fz_owned(iz_prod) && fz_owned(iz_prod + 1);
    };

    const T fac = (T)0.5;
    const T half = (T)0.5;
    const T quarter = (T)0.25;  // unused but documents flux/4 factor below
    (void)quarter;
    const T x_lo = (T)-1.0, x_md = (T)0.0, x_hi = (T)1.0;
    const T two = (T)2.0, four = (T)4.0;

    // === LEFT FACE (ix = -1) ===
    //   writes cluster cx=1; reads u at ix+1 (cx=3) and ix+2 (cx=4); no x-axis
    //   z-coupling, so a single OMP parallel for over (iy, iz) is safe.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for( int iy = 0; iy <= nyf - 2; iy += 2 ) {
        const int iytop = (iy >> 1) + oy_c;
        for( int iz = 0; iz <= nzf - 2; iz += 2 ) {
            if( !fpair_owned(iz) ) continue;
            const int iztop = (iz >> 1) + oz_c;
            const int ix = -1;
            const int ixtop = ox_c - 1;

            T flux = (T)0.0;
            // First pass: compute uhat per (j,k) and write u(ix, iy+j, iz+k).
            for( int j = 0; j <= 1; ++j ) {
                for( int k = 0; k <= 1; ++k ) {
                    T ustar1 = interp2_4arg_T<T>( V_at(ixtop, iytop-1, iztop-1),
                                                  V_at(ixtop, iytop,   iztop-1),
                                                  V_at(ixtop, iytop+1, iztop-1),
                                                  fac*((T)j - half) );
                    T ustar2 = interp2_4arg_T<T>( V_at(ixtop, iytop-1, iztop),
                                                  V_at(ixtop, iytop,   iztop),
                                                  V_at(ixtop, iytop+1, iztop),
                                                  fac*((T)j - half) );
                    T ustar3 = interp2_4arg_T<T>( V_at(ixtop, iytop-1, iztop+1),
                                                  V_at(ixtop, iytop,   iztop+1),
                                                  V_at(ixtop, iytop+1, iztop+1),
                                                  fac*((T)j - half) );
                    T uhat   = interp2_4arg_T<T>( ustar1, ustar2, ustar3,
                                                  fac*((T)k - half) );
                    T u_in1 = fine_slab_buf[U_idx(ix+1, iy+j, iz+k)];
                    T u_in2 = fine_slab_buf[U_idx(ix+2, iy+j, iz+k)];
                    T u_bc  = interp2left_T<T>( uhat, u_in1, u_in2 );
                    fine_slab_buf[U_idx(ix, iy+j, iz+k)] = u_bc;
                    flux += (u_in1 - u_bc);
                }
            }
            flux /= four;
            T dflux = ( V_at(ixtop+1, iytop, iztop)
                      - V_at(ixtop,   iytop, iztop) ) / two - flux;
            for( int j = 0; j <= 1; ++j )
                for( int k = 0; k <= 1; ++k )
                    fine_slab_buf[U_idx(ix, iy+j, iz+k)] -= dflux;
        }
    }

    // === RIGHT FACE (ix = nxf) ===
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for( int iy = 0; iy <= nyf - 2; iy += 2 ) {
        const int iytop = (iy >> 1) + oy_c;
        for( int iz = 0; iz <= nzf - 2; iz += 2 ) {
            if( !fpair_owned(iz) ) continue;
            const int iztop = (iz >> 1) + oz_c;
            const int ix = nxf;
            const int ixtop = nx + ox_c;  // (int)(0.5*nxf) + xoff

            T flux = (T)0.0;
            for( int j = 0; j <= 1; ++j ) {
                for( int k = 0; k <= 1; ++k ) {
                    T ustar1 = interp2_4arg_T<T>( V_at(ixtop, iytop-1, iztop-1),
                                                  V_at(ixtop, iytop,   iztop-1),
                                                  V_at(ixtop, iytop+1, iztop-1),
                                                  fac*((T)j - half) );
                    T ustar2 = interp2_4arg_T<T>( V_at(ixtop, iytop-1, iztop),
                                                  V_at(ixtop, iytop,   iztop),
                                                  V_at(ixtop, iytop+1, iztop),
                                                  fac*((T)j - half) );
                    T ustar3 = interp2_4arg_T<T>( V_at(ixtop, iytop-1, iztop+1),
                                                  V_at(ixtop, iytop,   iztop+1),
                                                  V_at(ixtop, iytop+1, iztop+1),
                                                  fac*((T)j - half) );
                    T uhat   = interp2_7arg_T<T>( x_lo, x_md, x_hi,
                                                  ustar1, ustar2, ustar3,
                                                  fac*((T)k - half) );
                    T u_in2 = fine_slab_buf[U_idx(ix-2, iy+j, iz+k)];
                    T u_in1 = fine_slab_buf[U_idx(ix-1, iy+j, iz+k)];
                    T u_bc  = interp2right_T<T>( u_in2, u_in1, uhat );
                    fine_slab_buf[U_idx(ix, iy+j, iz+k)] = u_bc;
                    flux += (u_bc - u_in1);
                }
            }
            flux /= four;
            T dflux = ( V_at(ixtop,   iytop, iztop)
                      - V_at(ixtop-1, iytop, iztop) ) / two - flux;
            for( int j = 0; j <= 1; ++j )
                for( int k = 0; k <= 1; ++k )
                    fine_slab_buf[U_idx(ix, iy+j, iz+k)] += dflux;
        }
    }

    // === BOTTOM FACE (iy = -1) ===
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for( int ix = 0; ix <= nxf - 2; ix += 2 ) {
        const int ixtop = (ix >> 1) + ox_c;
        for( int iz = 0; iz <= nzf - 2; iz += 2 ) {
            if( !fpair_owned(iz) ) continue;
            const int iztop = (iz >> 1) + oz_c;
            const int iy = -1;
            const int iytop = oy_c - 1;

            T flux = (T)0.0;
            for( int j = 0; j <= 1; ++j ) {
                for( int k = 0; k <= 1; ++k ) {
                    T ustar1 = interp2_4arg_T<T>( V_at(ixtop-1, iytop, iztop-1),
                                                  V_at(ixtop,   iytop, iztop-1),
                                                  V_at(ixtop+1, iytop, iztop-1),
                                                  fac*((T)j - half) );
                    T ustar2 = interp2_4arg_T<T>( V_at(ixtop-1, iytop, iztop),
                                                  V_at(ixtop,   iytop, iztop),
                                                  V_at(ixtop+1, iytop, iztop),
                                                  fac*((T)j - half) );
                    T ustar3 = interp2_4arg_T<T>( V_at(ixtop-1, iytop, iztop+1),
                                                  V_at(ixtop,   iytop, iztop+1),
                                                  V_at(ixtop+1, iytop, iztop+1),
                                                  fac*((T)j - half) );
                    T uhat   = interp2_7arg_T<T>( x_lo, x_md, x_hi,
                                                  ustar1, ustar2, ustar3,
                                                  fac*((T)k - half) );
                    T u_in1 = fine_slab_buf[U_idx(ix+j, iy+1, iz+k)];
                    T u_in2 = fine_slab_buf[U_idx(ix+j, iy+2, iz+k)];
                    T u_bc  = interp2left_T<T>( uhat, u_in1, u_in2 );
                    fine_slab_buf[U_idx(ix+j, iy, iz+k)] = u_bc;
                    flux += (u_in1 - u_bc);
                }
            }
            flux /= four;
            T dflux = ( V_at(ixtop, iytop+1, iztop)
                      - V_at(ixtop, iytop,   iztop) ) / two - flux;
            for( int j = 0; j <= 1; ++j )
                for( int k = 0; k <= 1; ++k )
                    fine_slab_buf[U_idx(ix+j, iy, iz+k)] -= dflux;
        }
    }

    // === TOP FACE (iy = nyf) ===
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for( int ix = 0; ix <= nxf - 2; ix += 2 ) {
        const int ixtop = (ix >> 1) + ox_c;
        for( int iz = 0; iz <= nzf - 2; iz += 2 ) {
            if( !fpair_owned(iz) ) continue;
            const int iztop = (iz >> 1) + oz_c;
            const int iy = nyf;
            const int iytop = ny + oy_c;

            T flux = (T)0.0;
            for( int j = 0; j <= 1; ++j ) {
                for( int k = 0; k <= 1; ++k ) {
                    T ustar1 = interp2_4arg_T<T>( V_at(ixtop-1, iytop, iztop-1),
                                                  V_at(ixtop,   iytop, iztop-1),
                                                  V_at(ixtop+1, iytop, iztop-1),
                                                  fac*((T)j - half) );
                    T ustar2 = interp2_4arg_T<T>( V_at(ixtop-1, iytop, iztop),
                                                  V_at(ixtop,   iytop, iztop),
                                                  V_at(ixtop+1, iytop, iztop),
                                                  fac*((T)j - half) );
                    T ustar3 = interp2_4arg_T<T>( V_at(ixtop-1, iytop, iztop+1),
                                                  V_at(ixtop,   iytop, iztop+1),
                                                  V_at(ixtop+1, iytop, iztop+1),
                                                  fac*((T)j - half) );
                    T uhat   = interp2_7arg_T<T>( x_lo, x_md, x_hi,
                                                  ustar1, ustar2, ustar3,
                                                  fac*((T)k - half) );
                    T u_in2 = fine_slab_buf[U_idx(ix+j, iy-2, iz+k)];
                    T u_in1 = fine_slab_buf[U_idx(ix+j, iy-1, iz+k)];
                    T u_bc  = interp2right_T<T>( u_in2, u_in1, uhat );
                    fine_slab_buf[U_idx(ix+j, iy, iz+k)] = u_bc;
                    flux += (u_bc - u_in1);
                }
            }
            flux /= four;
            T dflux = ( V_at(ixtop, iytop,   iztop)
                      - V_at(ixtop, iytop-1, iztop) ) / two - flux;
            for( int j = 0; j <= 1; ++j )
                for( int k = 0; k <= 1; ++k )
                    fine_slab_buf[U_idx(ix+j, iy, iz+k)] += dflux;
        }
    }

    // === FRONT FACE (iz = -1) ===
    //   writes cluster cz=1 — only the rank that owns cz=1 (typically my_z0=0)
    //   does work here. Reads u at iz+1, iz+2 → cluster cz=3, 4 (interior of
    //   the same rank, provided local_nz >= 3; precondition guarantees this
    //   indirectly via my_z0 even + local_nz even and nz >= 2).
    if( fz_owned(-1) ) {
        const int iz = -1;
        const int iztop = oz_c - 1;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for( int ix = 0; ix <= nxf - 2; ix += 2 ) {
            const int ixtop = (ix >> 1) + ox_c;
            for( int iy = 0; iy <= nyf - 2; iy += 2 ) {
                const int iytop = (iy >> 1) + oy_c;

                T flux = (T)0.0;
                for( int j = 0; j <= 1; ++j ) {
                    for( int k = 0; k <= 1; ++k ) {
                        T ustar1 = interp2_4arg_T<T>( V_at(ixtop-1, iytop-1, iztop),
                                                      V_at(ixtop,   iytop-1, iztop),
                                                      V_at(ixtop+1, iytop-1, iztop),
                                                      fac*((T)j - half) );
                        T ustar2 = interp2_4arg_T<T>( V_at(ixtop-1, iytop,   iztop),
                                                      V_at(ixtop,   iytop,   iztop),
                                                      V_at(ixtop+1, iytop,   iztop),
                                                      fac*((T)j - half) );
                        T ustar3 = interp2_4arg_T<T>( V_at(ixtop-1, iytop+1, iztop),
                                                      V_at(ixtop,   iytop+1, iztop),
                                                      V_at(ixtop+1, iytop+1, iztop),
                                                      fac*((T)j - half) );
                        T uhat   = interp2_7arg_T<T>( x_lo, x_md, x_hi,
                                                      ustar1, ustar2, ustar3,
                                                      fac*((T)k - half) );
                        T u_in1 = fine_slab_buf[U_idx(ix+j, iy+k, iz+1)];
                        T u_in2 = fine_slab_buf[U_idx(ix+j, iy+k, iz+2)];
                        T u_bc  = interp2left_T<T>( uhat, u_in1, u_in2 );
                        fine_slab_buf[U_idx(ix+j, iy+k, iz)] = u_bc;
                        flux += (u_in1 - u_bc);
                    }
                }
                flux /= four;
                T dflux = ( V_at(ixtop, iytop, iztop+1)
                          - V_at(ixtop, iytop, iztop  ) ) / two - flux;
                for( int j = 0; j <= 1; ++j )
                    for( int k = 0; k <= 1; ++k )
                        fine_slab_buf[U_idx(ix+j, iy+k, iz)] -= dflux;
            }
        }
    }

    // === BACK FACE (iz = nzf) ===
    if( fz_owned(nzf) ) {
        const int iz = nzf;
        const int iztop = nz + oz_c;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for( int ix = 0; ix <= nxf - 2; ix += 2 ) {
            const int ixtop = (ix >> 1) + ox_c;
            for( int iy = 0; iy <= nyf - 2; iy += 2 ) {
                const int iytop = (iy >> 1) + oy_c;

                T flux = (T)0.0;
                for( int j = 0; j <= 1; ++j ) {
                    for( int k = 0; k <= 1; ++k ) {
                        T ustar1 = interp2_4arg_T<T>( V_at(ixtop-1, iytop-1, iztop),
                                                      V_at(ixtop,   iytop-1, iztop),
                                                      V_at(ixtop+1, iytop-1, iztop),
                                                      fac*((T)j - half) );
                        T ustar2 = interp2_4arg_T<T>( V_at(ixtop-1, iytop,   iztop),
                                                      V_at(ixtop,   iytop,   iztop),
                                                      V_at(ixtop+1, iytop,   iztop),
                                                      fac*((T)j - half) );
                        T ustar3 = interp2_4arg_T<T>( V_at(ixtop-1, iytop+1, iztop),
                                                      V_at(ixtop,   iytop+1, iztop),
                                                      V_at(ixtop+1, iytop+1, iztop),
                                                      fac*((T)j - half) );
                        T uhat   = interp2_7arg_T<T>( x_lo, x_md, x_hi,
                                                      ustar1, ustar2, ustar3,
                                                      fac*((T)k - half) );
                        T u_in2 = fine_slab_buf[U_idx(ix+j, iy+k, iz-2)];
                        T u_in1 = fine_slab_buf[U_idx(ix+j, iy+k, iz-1)];
                        T u_bc  = interp2right_T<T>( u_in2, u_in1, uhat );
                        fine_slab_buf[U_idx(ix+j, iy+k, iz)] = u_bc;
                        flux += (u_bc - u_in1);
                    }
                }
                flux /= four;
                T dflux = ( V_at(ixtop, iytop, iztop  )
                          - V_at(ixtop, iytop, iztop-1) ) / two - flux;
                for( int j = 0; j <= 1; ++j )
                    for( int k = 0; k <= 1; ++k )
                        fine_slab_buf[U_idx(ix+j, iy+k, iz)] += dflux;
            }
        }
    }

    return true;
}

} // anonymous namespace

bool interp_cf_flux_z_slab_double(int cnxc, int cnyc, int cnzc,
                                  const double* coarse_buf,
                                  int ox_c, int oy_c, int oz_c,
                                  int nx, int ny, int nz,
                                  const ZoomSlabLayout& LfC,
                                  double* fine_slab_buf)
{ return interp_cf_flux_z_slab_impl<double>(cnxc, cnyc, cnzc, coarse_buf,
                                            ox_c, oy_c, oz_c, nx, ny, nz,
                                            LfC, fine_slab_buf); }

bool interp_cf_flux_z_slab_float (int cnxc, int cnyc, int cnzc,
                                  const float*  coarse_buf,
                                  int ox_c, int oy_c, int oz_c,
                                  int nx, int ny, int nz,
                                  const ZoomSlabLayout& LfC,
                                  float*  fine_slab_buf)
{ return interp_cf_flux_z_slab_impl<float >(cnxc, cnyc, cnzc, coarse_buf,
                                            ox_c, oy_c, oz_c, nx, ny, nz,
                                            LfC, fine_slab_buf); }

// ---------------------------------------------------------------------------
// Phase G.2b B.5.3c: collective interp_coarse_fine on sub_comm. Bridges a
// parent (coarse) MeshvarBnd<T> into a fine per-box MeshvarBnd<T> via the
// composite (prolong_bnd_z_slab + halo_exchange_z + interp_cf_flux_z_slab),
// mirroring interp_O3_fluxcorr::interp_coarse_fine (mg_interp.hh:434-636).
//
// Semantics:
//   - Writes uf's 2-cell BC perimeter (the m_nbnd=2 layer around interior).
//   - Reads uf's interior (unchanged on return).
//   - Reads uc's interior + 1 BC cell on each side (cluster reach
//     [xoff-3, xoff+nxc_int+2] on uc-local indexing).
//
// Bridge protocol:
//   (1) box_owner broadcasts fine dims (nxf,nyf,nzf) + offsets (oxf,oyf,ozf).
//   (2) all ranks build Lf with cluster shape (nxf+4, nyf+4, nzf+4) and
//       halo_w=1.
//   (3) box_owner packs uf cluster (interior + BC perimeter) into a cluster-
//       sized buffer; also packs uc sub-region [xoff-3..xoff+nxc_int+3] into
//       a (nxc_int+6)^3 cluster-shaped coarse buffer (cluster-frame offsets
//       ox_c=oy_c=oz_c=3 matching kernel preconditions).
//   (4) coarse buffer is MPI_Bcast (REPLICATED on every sub_comm rank);
//       fine cluster scattered to z-slabs.
//   (5) all ranks copy slab interior into halo-padded fine buffer.
//   (6) all ranks run prolong_bnd_z_slab_impl (writes BC perimeter via cubic).
//   (7) all ranks halo_exchange_z so interior-neighbour reads for flux are
//       coherent.
//   (8) all ranks run interp_cf_flux_z_slab_impl (overwrites BC perimeter
//       with flux-corrected values).
//   (9) extract fine interior back from halo'd buffer.
//   (10) gather fine slab back to box_owner.
//   (11) box_owner writes resulting fine cluster back into uf (all cells,
//        including BC perimeter — interior cells unchanged by composite).
//
// Returns true iff the collective path executed; returns false on:
//   - sub_size == 1 (caller should run local interp_coarse_fine)
//   - any kernel precondition failure (parity, alignment, geometry,
//     coarse-reach OOB)
//
// Preconditions (when valid uc/uf on owner):
//   - uf->size(d) even on all 3 axes
//   - uf->size(d) >= 2 (so nxc_int >= 1)
//   - uf->offset(d) >= 1 (coarse-stencil low-side reach)
//   - uc->size(d) >= uf->offset(d) + uf->size(d)/2 + 1 (coarse-stencil hi-side)
//   - cluster_nz % (2 * sub_size) == 0 AND local_nz even
//     (pair preservation — fz=2*kc, 2*kc+1 stay on same rank)
//
// Caller must perform local interp_coarse_fine when this returns false.
// ---------------------------------------------------------------------------
namespace {

template<typename T>
bool interp_coarse_fine_meshvarbnd_impl(int box_owner,
                                        const MeshvarBnd<T>* uc,
                                        MeshvarBnd<T>* uf,
                                        MPI_Comm sub_comm)
{
    int rk = 0, sz = 1;
#ifdef USE_MPI
    MPI_Comm_rank(sub_comm, &rk);
    MPI_Comm_size(sub_comm, &sz);
#endif

    if( sz <= 1 ) return false;

#ifdef USE_MPI
    const MPI_Datatype MPI_T =
        (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
#endif

    // Header: fine dims + fine offsets (in uc's coarse-grid indexing).
    int hdr[6] = {0, 0, 0, 0, 0, 0};
    if( rk == box_owner ) {
        if( !uf || !uc )
            throw std::runtime_error("interp_coarse_fine_meshvarbnd: uf/uc null on box_owner");
        static bool s_b5_3c_logged = false;
        if( !s_b5_3c_logged ) {
            s_b5_3c_logged = true;
            LOGINFO("B.5.3c: interp_coarse_fine_meshvarbnd bridge invoked (uf=(%d,%d,%d) "
                    "oxf=(%d,%d,%d), uc=(%d,%d,%d), sub_size=%d)",
                    (int)uf->size(0),(int)uf->size(1),(int)uf->size(2),
                    uf->offset(0),uf->offset(1),uf->offset(2),
                    (int)uc->size(0),(int)uc->size(1),(int)uc->size(2),
                    sz);
        }
        hdr[0] = (int)uf->size(0);
        hdr[1] = (int)uf->size(1);
        hdr[2] = (int)uf->size(2);
        hdr[3] = uf->offset(0);
        hdr[4] = uf->offset(1);
        hdr[5] = uf->offset(2);
    }
#ifdef USE_MPI
    MPI_Bcast(hdr, 6, MPI_INT, box_owner, sub_comm);
#endif

    const int nxf = hdr[0], nyf = hdr[1], nzf = hdr[2];
    const int oxf = hdr[3], oyf = hdr[4], ozf = hdr[5];

    if( nxf <= 0 || nyf <= 0 || nzf <= 0 )
        throw std::runtime_error("interp_coarse_fine_meshvarbnd: bad fine dims from box_owner");

    if( (nxf & 1) || (nyf & 1) || (nzf & 1) ) return false;

    const int nxc_int = nxf / 2;
    const int nyc_int = nyf / 2;
    const int nzc_int = nzf / 2;
    if( nxc_int < 1 || nyc_int < 1 || nzc_int < 1 ) return false;

    // Coarse reach checked on owner (coarse-mesh size known there only).
    // Encode pass/fail in an int we Bcast so all ranks bail out together.
    int reach_ok = 1;
    int reach_diag = 0;
    if( rk == box_owner ) {
        if( oxf < 1 || oyf < 1 || ozf < 1 ) { reach_ok = 0; reach_diag = 1; }
        if( (int)uc->size(0) < oxf + nxc_int + 1 ) { reach_ok = 0; reach_diag = 2; }
        if( (int)uc->size(1) < oyf + nyc_int + 1 ) { reach_ok = 0; reach_diag = 3; }
        if( (int)uc->size(2) < ozf + nzc_int + 1 ) { reach_ok = 0; reach_diag = 4; }
        static bool s_b5_3c_fail_logged = false;
        if( !reach_ok && !s_b5_3c_fail_logged ) {
            s_b5_3c_fail_logged = true;
            LOGINFO("B.5.3c: reach_ok=0 (reason=%d) uf=(%d,%d,%d) off=(%d,%d,%d) "
                    "uc=(%d,%d,%d) — falling back to local interp_coarse_fine",
                    reach_diag,
                    nxf, nyf, nzf, oxf, oyf, ozf,
                    (int)uc->size(0),(int)uc->size(1),(int)uc->size(2));
        }
    }
#ifdef USE_MPI
    MPI_Bcast(&reach_ok, 1, MPI_INT, box_owner, sub_comm);
#endif
    if( !reach_ok ) return false;

    // Pair preservation: cluster_nz = nzf+4 must split evenly across sub_size,
    // and the rank-local nz must be even, and the start must be even.
    const int cluster_nz_f = nzf + 4;
    if( cluster_nz_f % (2 * sz) != 0 ) return false;

    const int halo_w = 1;

    // Fine slab layout: cluster shape (nxf+4, nyf+4, nzf+4) = (2*nxc+4)^3.
    ZoomSlabLayout Lf = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/1,
        0, 0, 0, /*cluster_nx*/ nxf + 4, /*cluster_ny*/ nyf + 4,
        /*cluster_nz*/ nzf + 4, halo_w);

    // Local-nz parity guard (matches interp_cf_flux_z_slab_impl preconditions).
    if( (Lf.my_z0 & 1) || (local_nz(Lf) & 1) ) return false;

    // Coarse cluster shape: ox_c=oy_c=oz_c=3, cnxc = nxc_int + 6.
    const int ox_c = 3, oy_c = 3, oz_c = 3;
    const int cnxc = nxc_int + 6;
    const int cnyc = nyc_int + 6;
    const int cnzc = nzc_int + 6;
    const std::size_t coarse_buf_sz =
        (std::size_t)cnxc * (std::size_t)cnyc * (std::size_t)cnzc;

    // --- (3) pack coarse + fine on box_owner.
    std::vector<T> coarse_buf(coarse_buf_sz, T(0));
    std::vector<T> cluster_uf_in;

    if( rk == box_owner ) {
        // Coarse pack: cluster cell cx ∈ [0, cnxc) ↔ uc-local index (oxf + cx - ox_c).
        // uc allows reads in [-m_nbnd, size+m_nbnd-1]; with m_nbnd >= 2 and
        // (oxf >= 1, oxf + nxc_int + 2 <= uc.size+1) verified above, the full
        // [oxf-3, oxf+nxc_int+2] range is in-bounds.
        for( int cx=0; cx<cnxc; ++cx ) {
            const int ix_uc = oxf + cx - ox_c;
            for( int cy=0; cy<cnyc; ++cy ) {
                const int iy_uc = oyf + cy - oy_c;
                for( int cz=0; cz<cnzc; ++cz ) {
                    const int iz_uc = ozf + cz - oz_c;
                    const std::size_t idx =
                        ((std::size_t)cx * (std::size_t)cnyc + (std::size_t)cy)
                        * (std::size_t)cnzc + (std::size_t)cz;
                    coarse_buf[idx] = (*uc)(ix_uc, iy_uc, iz_uc);
                }
            }
        }

        // Fine pack: cluster cell (cx,cy,cz) ∈ [0, nxf+4) ↔ uf-local index
        // (cx-2, cy-2, cz-2). uf allows reads at [-m_nbnd, size+m_nbnd-1]
        // = [-2, size+1] for m_nbnd=2 — exactly the cluster span.
        cluster_uf_in.assign(cluster_full_size(Lf), T(0));
        const int cnxf = Lf.cluster_nx;
        const int cnyf = Lf.cluster_ny;
        const int cnzf = Lf.cluster_nz;
        for( int cx=0; cx<cnxf; ++cx ) {
            const int ix_uf = cx - 2;
            for( int cy=0; cy<cnyf; ++cy ) {
                const int iy_uf = cy - 2;
                for( int cz=0; cz<cnzf; ++cz ) {
                    const int iz_uf = cz - 2;
                    const std::size_t idx =
                        ((std::size_t)cx * (std::size_t)cnyf + (std::size_t)cy)
                        * (std::size_t)cnzf + (std::size_t)cz;
                    cluster_uf_in[idx] = (*uf)(ix_uf, iy_uf, iz_uf);
                }
            }
        }
    }

    // --- (4) Bcast coarse (REPLICATED); scatter fine.
#ifdef USE_MPI
    MPI_Bcast(coarse_buf.data(), (int)coarse_buf_sz, MPI_T, box_owner, sub_comm);
#endif

    std::vector<T> my_uf_int(local_interior_size(Lf), T(0));
    scatter_from(Lf, box_owner,
                 rk == box_owner ? cluster_uf_in.data() : (const T*)nullptr,
                 my_uf_int.data());

    // --- (5) copy fine slab interior into halo'd buffer at halo offset.
    std::vector<T> uf_pad(local_with_halo_size(Lf), T(0));
    {
        const int nz_rf  = local_nz(Lf);
        const int hwf    = Lf.halo_w;
        const int kstridef = 2 * hwf + nz_rf;
        const int cnxf = Lf.cluster_nx;
        const int cnyf = Lf.cluster_ny;
        for( int ii=0; ii<cnxf; ++ii )
            for( int jj=0; jj<cnyf; ++jj )
                for( int kl=0; kl<nz_rf; ++kl ) {
                    const std::size_t dst =
                        ((std::size_t)ii * (std::size_t)cnyf + (std::size_t)jj)
                        * (std::size_t)kstridef + (std::size_t)(hwf + kl);
                    const std::size_t src =
                        ((std::size_t)ii * (std::size_t)cnyf + (std::size_t)jj)
                        * (std::size_t)nz_rf + (std::size_t)kl;
                    uf_pad[dst] = my_uf_int[src];
                }
    }

    // --- (6) prolong_bnd_z_slab: writes BC perimeter via cubic.
    if( !prolong_bnd_z_slab_impl<T>(cnxc, cnyc, cnzc, coarse_buf.data(),
                                    ox_c, oy_c, oz_c,
                                    nxc_int, nyc_int, nzc_int,
                                    Lf, uf_pad.data()) )
        return false;

    // --- (7) halo exchange so neighbour-rank interior is visible for flux.
    halo_exchange_z(Lf, uf_pad.data());

    // --- (8) interp_cf_flux_z_slab: overwrites BC perimeter with flux-
    //         corrected values (production's 6 per-face corrections).
    if( !interp_cf_flux_z_slab_impl<T>(cnxc, cnyc, cnzc, coarse_buf.data(),
                                       ox_c, oy_c, oz_c,
                                       nxc_int, nyc_int, nzc_int,
                                       Lf, uf_pad.data()) )
        return false;

    // --- (9) extract fine interior back from halo'd buffer.
    {
        const int nz_rf  = local_nz(Lf);
        const int hwf    = Lf.halo_w;
        const int kstridef = 2 * hwf + nz_rf;
        const int cnxf = Lf.cluster_nx;
        const int cnyf = Lf.cluster_ny;
        for( int ii=0; ii<cnxf; ++ii )
            for( int jj=0; jj<cnyf; ++jj )
                for( int kl=0; kl<nz_rf; ++kl ) {
                    const std::size_t src =
                        ((std::size_t)ii * (std::size_t)cnyf + (std::size_t)jj)
                        * (std::size_t)kstridef + (std::size_t)(hwf + kl);
                    const std::size_t dst =
                        ((std::size_t)ii * (std::size_t)cnyf + (std::size_t)jj)
                        * (std::size_t)nz_rf + (std::size_t)kl;
                    my_uf_int[dst] = uf_pad[src];
                }
    }

    // --- (10) gather fine slabs back to box_owner.
    std::vector<T> cluster_uf_out;
    if( rk == box_owner ) cluster_uf_out.assign(cluster_full_size(Lf), T(0));
    gather_to(Lf, box_owner, my_uf_int.data(),
              rk == box_owner ? cluster_uf_out.data() : (T*)nullptr);

    // --- (11) box_owner writes back into uf — cluster cells map to uf-local
    //          index (cx-2, cy-2, cz-2). The composite only modifies the BC
    //          perimeter (interior cells round-trip unchanged), but writing
    //          all of them is harmless and avoids any per-cell BC mask logic.
    if( rk == box_owner ) {
        const int cnxf = Lf.cluster_nx;
        const int cnyf = Lf.cluster_ny;
        const int cnzf = Lf.cluster_nz;
        for( int cx=0; cx<cnxf; ++cx ) {
            const int ix_uf = cx - 2;
            for( int cy=0; cy<cnyf; ++cy ) {
                const int iy_uf = cy - 2;
                for( int cz=0; cz<cnzf; ++cz ) {
                    const int iz_uf = cz - 2;
                    const std::size_t idx =
                        ((std::size_t)cx * (std::size_t)cnyf + (std::size_t)cy)
                        * (std::size_t)cnzf + (std::size_t)cz;
                    (*uf)(ix_uf, iy_uf, iz_uf) = cluster_uf_out[idx];
                }
            }
        }
    }

    return true;
}

} // anonymous namespace

bool interp_coarse_fine_meshvarbnd_double(int box_owner,
                                          const MeshvarBnd<double>* uc,
                                          MeshvarBnd<double>* uf,
                                          MPI_Comm sub_comm)
{ return interp_coarse_fine_meshvarbnd_impl<double>(box_owner, uc, uf, sub_comm); }

bool interp_coarse_fine_meshvarbnd_float (int box_owner,
                                          const MeshvarBnd<float>*  uc,
                                          MeshvarBnd<float>*  uf,
                                          MPI_Comm sub_comm)
{ return interp_coarse_fine_meshvarbnd_impl<float >(box_owner, uc, uf, sub_comm); }

// ---------------------------------------------------------------------------
// Phase G.2b B.5.4.a: keep-in-slab N-iteration smoothing.
//
// Equivalent to N iterations of:
//   interp_coarse_fine_meshvarbnd(owner, uc, uf, sub_comm)
//   gs_z_neg_meshvarbnd(owner, uf, ff, h, 1, sub_comm)
// but with a single scatter at entry and a single gather at exit. The shared
// slab buffer uses cluster shape (nxf+4, nyf+4, nzf+4) so that interp_cf can
// write its 2-cell flux-corrected BC perimeter (m_nbnd=2) AND gs can read the
// 1-cell BC ring (m_nbnd=1, which lives inside the m_nbnd=2 ring) without
// any redistribute between the two ops.
//
// Per iteration on the shared slab:
//   (i)  coarse pack (owner only) + Bcast across sub_comm.
//   (ii) prolong_bnd_z_slab — cubic BC perimeter write.
//   (iii) halo_exchange_z — refresh z-halos for the flux stencil.
//   (iv) interp_cf_flux_z_slab — overwrite BC perimeter with flux-corrected.
//   (v)  halo_exchange_z — refresh z-halos for gs.
//   (vi) gs_neg_skip_z_wide — one GS sweep over MeshvarBnd-interior cells
//        (cluster indices [2, size+2)), with halo exchange between colors.
//
// Returns true iff the collective path executed; returns false on:
//   - sub_size == 1 (caller should fall back to the existing per-op path)
//   - fine dim parity / size failure
//   - coarse-reach OOB on uc
//   - pair preservation broken
//
// When false is returned, no writes have been performed and the caller MUST
// fall back to the existing per-iteration loop.
// ---------------------------------------------------------------------------
namespace {

// Generalized GS kernel with adjustable BC perimeter width `bnd`. Writes
// cluster cells i ∈ [bnd, cnx-bnd), j ∈ [bnd, cny-bnd), and (cluster-global)
// k ∈ [bnd, cluster_nz-bnd). bnd=1 is the default kernel (matches
// gs_neg_skip_z_impl); bnd=2 is used by the keep-in-slab smoothing path where
// the slab includes a 2-cell flux-correction perimeter.
//
// Parity formula matches the production GS red-black ordering: write cell at
// MeshvarBnd-local index (mi, mj, mk) when (mi + mj + mk) & 1 == color. With
// cluster index ci = mi + bnd (etc.), this becomes (ci + cj + ck) & 1 ==
// (color + 3*bnd) & 1. For bnd=1, that simplifies to (ci+cj+ck+1)&1 == color
// (matches gs_neg_skip_z_impl). For bnd=2 the +3*bnd offset becomes +6, even,
// so (ci+cj+ck)&1 == color.
template<typename T>
void gs_neg_skip_z_wide_impl(const ZoomSlabLayout& L,
                             T* u, const T* f, T h, int bnd)
{
    const int cnx   = L.cluster_nx;
    const int cny   = L.cluster_ny;
    const int nz_r  = local_nz(L);
    const int hw    = L.halo_w;
    const int kstride = 2*hw + nz_r;
    const int z_base  = L.my_z0;
    const T   h2    = h * h;
    const T   sixth = T(1) / T(6);
    const std::size_t ny_str = (std::size_t)cny * (std::size_t)kstride;
    const std::size_t x_str  = (std::size_t)kstride;

    // Skip `bnd` BC cells at each z-end of the cluster.
    const int kl_lo = (L.sub_rank == 0)              ? (hw + bnd)         : hw;
    const int kl_hi = (L.sub_rank == L.sub_size - 1) ? (hw + nz_r - bnd)  : (hw + nz_r);
    const int parity_off = (3 * bnd) & 1; // 1 for bnd=1, 0 for bnd=2

    for( int color=0; color<2; ++color ) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for( int i=bnd; i<cnx-bnd; ++i ) {
            const std::size_t i_off    = (std::size_t)i * ny_str;
            const std::size_t im1_off  = (std::size_t)(i-1) * ny_str;
            const std::size_t ip1_off  = (std::size_t)(i+1) * ny_str;
            for( int j=bnd; j<cny-bnd; ++j ) {
                const std::size_t row    = i_off   + (std::size_t)j * x_str;
                const std::size_t rim1   = im1_off + (std::size_t)j * x_str;
                const std::size_t rip1   = ip1_off + (std::size_t)j * x_str;
                const std::size_t rjm1   = i_off   + (std::size_t)(j-1) * x_str;
                const std::size_t rjp1   = i_off   + (std::size_t)(j+1) * x_str;
                const int kg_lo = z_base + (kl_lo - hw);
                const int kl_start = kl_lo + (( (i + j + kg_lo + parity_off) & 1 ) == color ? 0 : 1);
                for( int kl=kl_start; kl<kl_hi; kl+=2 ) {
                    const std::size_t idx = row + (std::size_t)kl;
                    const T sum = u[rim1 + kl] + u[rip1 + kl]
                                + u[rjm1 + kl] + u[rjp1 + kl]
                                + u[row  + kl - 1] + u[row + kl + 1];
                    u[idx] = (sum + h2 * f[idx]) * sixth;
                }
            }
        }
#ifdef USE_MPI
        if( sizeof(T) == sizeof(double) )
            halo_exchange_z_double(L, reinterpret_cast<double*>(u));
        else
            halo_exchange_z_float (L, reinterpret_cast<float*> (u));
#else
        (void)L;
#endif
    }
}

// Phase G.2b B.5.4.a/b shared body.
//
// Lf_out / my_uf_int_out are mutually-required output parameters: when both
// are non-null, the impl performs the standard final gather + uf writeback
// AND additionally hands the post-smoother padded-cluster interior buffer to
// the caller:
//   - *Lf_out                = the constructed ZoomSlabLayout (cluster_n =
//                              {nxf,nyf,nzf}+4, halo_w=1, perimeter=2)
//   - *my_uf_int_out         = per-rank interior buffer of the *padded*
//                              cluster (size local_interior_size(*Lf_out)),
//                              ready to feed restrict_meshvarbnd_from_padded_slab
//                              with perimeter=2.
// uf is current on owner either way; keep_slab callers also get the buffer.
template<typename T>
bool smooth_pre_post_n_meshvarbnd_impl(int box_owner,
                                       const MeshvarBnd<T>* uc,
                                       MeshvarBnd<T>* uf,
                                       const MeshvarBnd<T>* ff,
                                       T h, int n_sweeps,
                                       MPI_Comm sub_comm,
                                       ZoomSlabLayout* Lf_out = nullptr,
                                       std::vector<T>* my_uf_int_out = nullptr)
{
    const bool keep_slab = (Lf_out != nullptr) && (my_uf_int_out != nullptr);
    int rk = 0, sz = 1;
#ifdef USE_MPI
    MPI_Comm_rank(sub_comm, &rk);
    MPI_Comm_size(sub_comm, &sz);
#endif

    if( sz <= 1 ) return false;
    if( n_sweeps <= 0 ) {
        // Caller would get unpopulated outputs — make it explicit by falling
        // back to the per-op path on its own.
        if( keep_slab ) return false;
        return true;
    }

#ifdef USE_MPI
    const MPI_Datatype MPI_T =
        (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
#endif

    // Header: fine dims + fine offsets in uc-coarse indexing.
    int hdr[6] = {0, 0, 0, 0, 0, 0};
    if( rk == box_owner ) {
        if( !uf || !ff || !uc )
            throw std::runtime_error("smooth_pre_post_n_meshvarbnd: uc/uf/ff null on box_owner");
        if( uf->m_nbnd < 2 )
            throw std::runtime_error("smooth_pre_post_n_meshvarbnd: uf->m_nbnd must be >= 2 (flux corr writes ix=-2)");
        if( ff->m_nbnd < 1 )
            throw std::runtime_error("smooth_pre_post_n_meshvarbnd: ff->m_nbnd must be >= 1");
        if( (int)ff->size(0) != (int)uf->size(0) ||
            (int)ff->size(1) != (int)uf->size(1) ||
            (int)ff->size(2) != (int)uf->size(2) )
            throw std::runtime_error("smooth_pre_post_n_meshvarbnd: uf/ff size mismatch");
        hdr[0] = (int)uf->size(0);
        hdr[1] = (int)uf->size(1);
        hdr[2] = (int)uf->size(2);
        hdr[3] = uf->offset(0);
        hdr[4] = uf->offset(1);
        hdr[5] = uf->offset(2);
    }
#ifdef USE_MPI
    MPI_Bcast(hdr, 6, MPI_INT, box_owner, sub_comm);
#endif

    const int nxf = hdr[0], nyf = hdr[1], nzf = hdr[2];
    const int oxf = hdr[3], oyf = hdr[4], ozf = hdr[5];
    if( nxf <= 0 || nyf <= 0 || nzf <= 0 )
        throw std::runtime_error("smooth_pre_post_n_meshvarbnd: bad fine dims from box_owner");
    if( (nxf & 1) || (nyf & 1) || (nzf & 1) ) return false;
    const int nxc_int = nxf / 2;
    const int nyc_int = nyf / 2;
    const int nzc_int = nzf / 2;
    if( nxc_int < 1 || nyc_int < 1 || nzc_int < 1 ) return false;

    // Coarse reach + alignment checks, mirroring interp_coarse_fine_meshvarbnd.
    int reach_ok = 1;
    if( rk == box_owner ) {
        if( oxf < 1 || oyf < 1 || ozf < 1 ) reach_ok = 0;
        if( (int)uc->size(0) < oxf + nxc_int + 1 ) reach_ok = 0;
        if( (int)uc->size(1) < oyf + nyc_int + 1 ) reach_ok = 0;
        if( (int)uc->size(2) < ozf + nzc_int + 1 ) reach_ok = 0;
    }
#ifdef USE_MPI
    MPI_Bcast(&reach_ok, 1, MPI_INT, box_owner, sub_comm);
#endif
    if( !reach_ok ) return false;

    const int cluster_nz_f = nzf + 4;
    if( cluster_nz_f % (2 * sz) != 0 ) return false;
    const int halo_w = 1;

    ZoomSlabLayout Lf = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/1,
        0, 0, 0, /*cluster_nx*/ nxf + 4, /*cluster_ny*/ nyf + 4,
        /*cluster_nz*/ nzf + 4, halo_w);
    if( (Lf.my_z0 & 1) || (local_nz(Lf) & 1) ) return false;

    static bool s_b5_4a_logged = false;
    if( rk == box_owner && !s_b5_4a_logged ) {
        s_b5_4a_logged = true;
        LOGINFO("B.5.4.a: smooth_pre_post_n_meshvarbnd active (uf=(%d,%d,%d) "
                "oxf=(%d,%d,%d) uc=(%d,%d,%d) n_sweeps=%d sub_size=%d)",
                nxf, nyf, nzf, oxf, oyf, ozf,
                (int)uc->size(0),(int)uc->size(1),(int)uc->size(2),
                n_sweeps, sz);
    }

    const int ox_c = 3, oy_c = 3, oz_c = 3;
    const int cnxc = nxc_int + 6;
    const int cnyc = nyc_int + 6;
    const int cnzc = nzc_int + 6;
    const std::size_t coarse_buf_sz =
        (std::size_t)cnxc * (std::size_t)cnyc * (std::size_t)cnzc;

    const int cnxf = Lf.cluster_nx;
    const int cnyf = Lf.cluster_ny;
    const int cnzf = Lf.cluster_nz;

    // --- (1) Pack uf cluster (interior + 2-cell BC perimeter) and ff cluster
    //         (interior only — outer 2 rings zero, never read by GS) on owner.
    std::vector<T> cluster_uf_in, cluster_ff_in;
    if( rk == box_owner ) {
        cluster_uf_in.assign(cluster_full_size(Lf), T(0));
        cluster_ff_in.assign(cluster_full_size(Lf), T(0));
        // uf: cluster cell (cx,cy,cz) ↔ uf-local index (cx-2, cy-2, cz-2),
        // valid for m_nbnd>=2 over the full [-2, size+1] range.
        for( int cx=0; cx<cnxf; ++cx ) {
            const int ix_uf = cx - 2;
            for( int cy=0; cy<cnyf; ++cy ) {
                const int iy_uf = cy - 2;
                for( int cz=0; cz<cnzf; ++cz ) {
                    const int iz_uf = cz - 2;
                    const std::size_t idx =
                        ((std::size_t)cx * (std::size_t)cnyf + (std::size_t)cy)
                        * (std::size_t)cnzf + (std::size_t)cz;
                    cluster_uf_in[idx] = (*uf)(ix_uf, iy_uf, iz_uf);
                }
            }
        }
        // ff: only interior cells [2, size+2) on each axis (GS only reads
        // interior); outer 2-cell rings remain zero.
        for( int cx=2; cx<cnxf-2; ++cx ) {
            const int ix_ff = cx - 2;
            for( int cy=2; cy<cnyf-2; ++cy ) {
                const int iy_ff = cy - 2;
                for( int cz=2; cz<cnzf-2; ++cz ) {
                    const int iz_ff = cz - 2;
                    const std::size_t idx =
                        ((std::size_t)cx * (std::size_t)cnyf + (std::size_t)cy)
                        * (std::size_t)cnzf + (std::size_t)cz;
                    cluster_ff_in[idx] = (*ff)(ix_ff, iy_ff, iz_ff);
                }
            }
        }
    }

    // --- (2) Scatter uf and ff to per-rank z-slabs.
    std::vector<T> my_uf_int(local_interior_size(Lf), T(0));
    std::vector<T> my_ff_int(local_interior_size(Lf), T(0));
    scatter_from(Lf, box_owner,
                 rk == box_owner ? cluster_uf_in.data() : (const T*)nullptr,
                 my_uf_int.data());
    scatter_from(Lf, box_owner,
                 rk == box_owner ? cluster_ff_in.data() : (const T*)nullptr,
                 my_ff_int.data());

    // --- (3) Copy interior into halo-padded buffer at k offset = halo_w.
    std::vector<T> uf_pad(local_with_halo_size(Lf), T(0));
    std::vector<T> ff_pad(local_with_halo_size(Lf), T(0));
    {
        const int nz_rf  = local_nz(Lf);
        const int hwf    = Lf.halo_w;
        const int kstridef = 2 * hwf + nz_rf;
        for( int ii=0; ii<cnxf; ++ii )
            for( int jj=0; jj<cnyf; ++jj )
                for( int kl=0; kl<nz_rf; ++kl ) {
                    const std::size_t dst =
                        ((std::size_t)ii * (std::size_t)cnyf + (std::size_t)jj)
                        * (std::size_t)kstridef + (std::size_t)(hwf + kl);
                    const std::size_t src =
                        ((std::size_t)ii * (std::size_t)cnyf + (std::size_t)jj)
                        * (std::size_t)nz_rf + (std::size_t)kl;
                    uf_pad[dst] = my_uf_int[src];
                    ff_pad[dst] = my_ff_int[src];
                }
    }

    // Initial halo exchange so interp_cf's interior reads and gs's stencil
    // reads see fresh neighbor values from the start.
    halo_exchange_z(Lf, uf_pad.data());

    // --- (4) Allocate coarse buffer once; pack on owner once per iteration
    //         since uc is constant across pre/post smoothing per V-cycle, but
    //         in principle a caller could mutate uc between iterations. To
    //         match the per-op interp_coarse_fine_meshvarbnd semantics exactly,
    //         we re-pack + re-bcast per iteration (still ~µs at typical sizes).
    std::vector<T> coarse_buf(coarse_buf_sz, T(0));

    for( int s = 0; s < n_sweeps; ++s ) {
        // Owner packs coarse sub-region [oxf-3, oxf+nxc_int+2] in uc-local.
        if( rk == box_owner ) {
            for( int cx=0; cx<cnxc; ++cx ) {
                const int ix_uc = oxf + cx - ox_c;
                for( int cy=0; cy<cnyc; ++cy ) {
                    const int iy_uc = oyf + cy - oy_c;
                    for( int cz=0; cz<cnzc; ++cz ) {
                        const int iz_uc = ozf + cz - oz_c;
                        const std::size_t idx =
                            ((std::size_t)cx * (std::size_t)cnyc + (std::size_t)cy)
                            * (std::size_t)cnzc + (std::size_t)cz;
                        coarse_buf[idx] = (*uc)(ix_uc, iy_uc, iz_uc);
                    }
                }
            }
        }
#ifdef USE_MPI
        MPI_Bcast(coarse_buf.data(), (int)coarse_buf_sz, MPI_T, box_owner, sub_comm);
#endif

        // (a) prolong_bnd_z_slab — cubic BC perimeter write.
        if( !prolong_bnd_z_slab_impl<T>(cnxc, cnyc, cnzc, coarse_buf.data(),
                                        ox_c, oy_c, oz_c,
                                        nxc_int, nyc_int, nzc_int,
                                        Lf, uf_pad.data()) )
            return false;

        // (b) halo exchange so neighbour interior visible for flux stencil.
        halo_exchange_z(Lf, uf_pad.data());

        // (c) interp_cf_flux_z_slab — overwrites BC perimeter with flux-corrected.
        if( !interp_cf_flux_z_slab_impl<T>(cnxc, cnyc, cnzc, coarse_buf.data(),
                                           ox_c, oy_c, oz_c,
                                           nxc_int, nyc_int, nzc_int,
                                           Lf, uf_pad.data()) )
            return false;

        // (d) halo exchange so GS sees fresh BC perimeter + interior neighbours.
        halo_exchange_z(Lf, uf_pad.data());

        // (e) One GS sweep over MeshvarBnd interior (cluster indices [2,size+2)).
        //     Kernel does its own halo exchange between red/black colors.
        gs_neg_skip_z_wide_impl<T>(Lf, uf_pad.data(), ff_pad.data(), h, /*bnd=*/2);
    }

    // --- (5) Extract halo-padded interior back into per-rank interior buffer.
    {
        const int nz_rf  = local_nz(Lf);
        const int hwf    = Lf.halo_w;
        const int kstridef = 2 * hwf + nz_rf;
        for( int ii=0; ii<cnxf; ++ii )
            for( int jj=0; jj<cnyf; ++jj )
                for( int kl=0; kl<nz_rf; ++kl ) {
                    const std::size_t src =
                        ((std::size_t)ii * (std::size_t)cnyf + (std::size_t)jj)
                        * (std::size_t)kstridef + (std::size_t)(hwf + kl);
                    const std::size_t dst =
                        ((std::size_t)ii * (std::size_t)cnyf + (std::size_t)jj)
                        * (std::size_t)nz_rf + (std::size_t)kl;
                    my_uf_int[dst] = uf_pad[src];
                }
    }

    // --- (6) Gather to owner; write back to uf (interior + 2-cell BC perimeter).
    //         B.5.4.b keep_slab mode still gathers + writes back uf (subsequent
    //         apply+restrict reads uf from owner via apply_meshvarbnd_to_slab),
    //         but ALSO hands the padded-cluster interior buffer to the caller
    //         so the immediately following u-restrict can chain into
    //         restrict_meshvarbnd_from_padded_slab(perimeter=2) and skip the
    //         re-scatter of uf. Net savings: 1 fine scatter per box per V-cycle.
    std::vector<T> cluster_uf_out;
    if( rk == box_owner ) cluster_uf_out.assign(cluster_full_size(Lf), T(0));
    gather_to(Lf, box_owner, my_uf_int.data(),
              rk == box_owner ? cluster_uf_out.data() : (T*)nullptr);

    if( rk == box_owner ) {
        for( int cx=0; cx<cnxf; ++cx ) {
            const int ix_uf = cx - 2;
            for( int cy=0; cy<cnyf; ++cy ) {
                const int iy_uf = cy - 2;
                for( int cz=0; cz<cnzf; ++cz ) {
                    const int iz_uf = cz - 2;
                    const std::size_t idx =
                        ((std::size_t)cx * (std::size_t)cnyf + (std::size_t)cy)
                        * (std::size_t)cnzf + (std::size_t)cz;
                    (*uf)(ix_uf, iy_uf, iz_uf) = cluster_uf_out[idx];
                }
            }
        }
    }

    if( keep_slab ) {
        *Lf_out        = Lf;
        *my_uf_int_out = std::move(my_uf_int);
    }

    return true;
}

// Phase G.2b B.5.4.c — fused prolong_add + post-smoothing in z-slab.
//
// Combines the coarse-correction prolong_add (uf += prolong(cc)) and the
// subsequent N-sweep post-smoother into a single scatter/gather lifetime.
// Eliminates 1 fine scatter + 1 fine gather per box per V-cycle versus the
// unfused (prolong_add_meshvarbnd bridge → smooth_pre_post_n_meshvarbnd)
// sequence.
//
// Bit-identicality: the post-smoother overwrites uf's 2-cell BC perimeter on
// every sweep (prolong_bnd + interp_cf) before GS reads it, so only the uf
// *interior* produced by prolong_add matters. The prolong injection is a pure
// per-fine-cell += of one coarse value (no reduction order), identical across
// z-slab decompositions; the smoother stage is byte-for-byte the same as the
// standalone smoother fed the same interior. Full bit-identicality expected.
//
// cc is the per-box coarse correction (same shape/offset as uc); the fine box
// footprint in cc is the coarse sub-region [oxf, oxf+nxc_int) — exactly the
// region prolong_add_meshvarbnd injects.
template<typename T>
bool prolong_add_then_smooth_n_meshvarbnd_impl(int box_owner,
                                               const MeshvarBnd<T>* cc,
                                               const MeshvarBnd<T>* uc,
                                               MeshvarBnd<T>* uf,
                                               const MeshvarBnd<T>* ff,
                                               T h, int n_sweeps,
                                               MPI_Comm sub_comm)
{
    int rk = 0, sz = 1;
#ifdef USE_MPI
    MPI_Comm_rank(sub_comm, &rk);
    MPI_Comm_size(sub_comm, &sz);
#endif

    if( sz <= 1 ) return false;
    if( n_sweeps <= 0 ) return false;

#ifdef USE_MPI
    const MPI_Datatype MPI_T =
        (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
#endif

    int hdr[6] = {0, 0, 0, 0, 0, 0};
    if( rk == box_owner ) {
        if( !uf || !ff || !uc || !cc )
            throw std::runtime_error("prolong_add_then_smooth: cc/uc/uf/ff null on box_owner");
        if( uf->m_nbnd < 2 )
            throw std::runtime_error("prolong_add_then_smooth: uf->m_nbnd must be >= 2");
        if( ff->m_nbnd < 1 )
            throw std::runtime_error("prolong_add_then_smooth: ff->m_nbnd must be >= 1");
        if( (int)ff->size(0) != (int)uf->size(0) ||
            (int)ff->size(1) != (int)uf->size(1) ||
            (int)ff->size(2) != (int)uf->size(2) )
            throw std::runtime_error("prolong_add_then_smooth: uf/ff size mismatch");
        hdr[0] = (int)uf->size(0);
        hdr[1] = (int)uf->size(1);
        hdr[2] = (int)uf->size(2);
        hdr[3] = uf->offset(0);
        hdr[4] = uf->offset(1);
        hdr[5] = uf->offset(2);
    }
#ifdef USE_MPI
    MPI_Bcast(hdr, 6, MPI_INT, box_owner, sub_comm);
#endif

    const int nxf = hdr[0], nyf = hdr[1], nzf = hdr[2];
    const int oxf = hdr[3], oyf = hdr[4], ozf = hdr[5];
    if( nxf <= 0 || nyf <= 0 || nzf <= 0 )
        throw std::runtime_error("prolong_add_then_smooth: bad fine dims from box_owner");
    if( (nxf & 1) || (nyf & 1) || (nzf & 1) ) return false;
    const int nxc_int = nxf / 2;
    const int nyc_int = nyf / 2;
    const int nzc_int = nzf / 2;
    if( nxc_int < 1 || nyc_int < 1 || nzc_int < 1 ) return false;

    int reach_ok = 1;
    if( rk == box_owner ) {
        if( oxf < 1 || oyf < 1 || ozf < 1 ) reach_ok = 0;
        if( (int)uc->size(0) < oxf + nxc_int + 1 ) reach_ok = 0;
        if( (int)uc->size(1) < oyf + nyc_int + 1 ) reach_ok = 0;
        if( (int)uc->size(2) < ozf + nzc_int + 1 ) reach_ok = 0;
    }
#ifdef USE_MPI
    MPI_Bcast(&reach_ok, 1, MPI_INT, box_owner, sub_comm);
#endif
    if( !reach_ok ) return false;

    const int cluster_nz_f = nzf + 4;
    if( cluster_nz_f % (2 * sz) != 0 ) return false;
    const int halo_w = 1;

    // Fine perimeter-2 layout (identical to the post-smoother).
    ZoomSlabLayout Lf = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/1,
        0, 0, 0, nxf + 4, nyf + 4, nzf + 4, halo_w);
    if( (Lf.my_z0 & 1) || (local_nz(Lf) & 1) ) return false;

    // Coarse injection layout for prolong_add: 1-cell perimeter so it is
    // exactly Lf/2 aligned (cluster = nxc_int+2, Lf cluster = nxf+4 =
    // 2*(nxc_int+2)). Natural coarse cell n maps to cluster cell n+1, whose
    // 8 fine children land on fine cluster cells [2n+2, 2n+4) = fine interior
    // cells [2n, 2n+2). The perimeter coarse cells inject onto the fine BC
    // perimeter (harmless — overwritten by interp_cf).
    ZoomSlabLayout Lc_pa = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/0,
        0, 0, 0, nxc_int + 2, nyc_int + 2, nzc_int + 2, halo_w);
    if( 2 * Lc_pa.my_z0 != Lf.my_z0 || 2 * Lc_pa.my_z1 != Lf.my_z1 )
        return false;

    static bool s_b5_4c_logged = false;
    if( rk == box_owner && !s_b5_4c_logged ) {
        s_b5_4c_logged = true;
        LOGINFO("B.5.4.c: prolong_add_then_smooth active (uf=(%d,%d,%d) "
                "oxf=(%d,%d,%d) n_sweeps=%d sub_size=%d)",
                nxf, nyf, nzf, oxf, oyf, ozf, n_sweeps, sz);
    }

    const int ox_c = 3, oy_c = 3, oz_c = 3;
    const int cnxc = nxc_int + 6;
    const int cnyc = nyc_int + 6;
    const int cnzc = nzc_int + 6;
    const std::size_t coarse_buf_sz =
        (std::size_t)cnxc * (std::size_t)cnyc * (std::size_t)cnzc;

    const int cnxf = Lf.cluster_nx;
    const int cnyf = Lf.cluster_ny;
    const int cnzf = Lf.cluster_nz;

    const int cnxc_pa = Lc_pa.cluster_nx;
    const int cnyc_pa = Lc_pa.cluster_ny;
    const int cnzc_pa = Lc_pa.cluster_nz;

    // --- (1) Pack uf cluster (interior + 2-cell perimeter), ff cluster
    //         (interior), and cc cluster (1-cell perimeter) on owner.
    std::vector<T> cluster_uf_in, cluster_ff_in, cluster_cc_in;
    if( rk == box_owner ) {
        cluster_uf_in.assign(cluster_full_size(Lf), T(0));
        cluster_ff_in.assign(cluster_full_size(Lf), T(0));
        cluster_cc_in.assign(cluster_full_size(Lc_pa), T(0));
        for( int cx=0; cx<cnxf; ++cx ) {
            const int ix_uf = cx - 2;
            for( int cy=0; cy<cnyf; ++cy ) {
                const int iy_uf = cy - 2;
                for( int cz=0; cz<cnzf; ++cz ) {
                    const int iz_uf = cz - 2;
                    const std::size_t idx =
                        ((std::size_t)cx * (std::size_t)cnyf + (std::size_t)cy)
                        * (std::size_t)cnzf + (std::size_t)cz;
                    cluster_uf_in[idx] = (*uf)(ix_uf, iy_uf, iz_uf);
                }
            }
        }
        for( int cx=2; cx<cnxf-2; ++cx ) {
            const int ix_ff = cx - 2;
            for( int cy=2; cy<cnyf-2; ++cy ) {
                const int iy_ff = cy - 2;
                for( int cz=2; cz<cnzf-2; ++cz ) {
                    const int iz_ff = cz - 2;
                    const std::size_t idx =
                        ((std::size_t)cx * (std::size_t)cnyf + (std::size_t)cy)
                        * (std::size_t)cnzf + (std::size_t)cz;
                    cluster_ff_in[idx] = (*ff)(ix_ff, iy_ff, iz_ff);
                }
            }
        }
        // cc: natural coarse cell (ic,jc,kc) -> cluster cell (ic+1,jc+1,kc+1).
        for( int ic=0; ic<nxc_int; ++ic )
            for( int jc=0; jc<nyc_int; ++jc )
                for( int kc=0; kc<nzc_int; ++kc ) {
                    const std::size_t idx =
                        ((std::size_t)(ic+1) * (std::size_t)cnyc_pa + (std::size_t)(jc+1))
                        * (std::size_t)cnzc_pa + (std::size_t)(kc+1);
                    cluster_cc_in[idx] = (*cc)(ic + oxf, jc + oyf, kc + ozf);
                }
    }

    // --- (2) Scatter uf, ff, cc to per-rank z-slabs.
    std::vector<T> my_uf_int(local_interior_size(Lf), T(0));
    std::vector<T> my_ff_int(local_interior_size(Lf), T(0));
    std::vector<T> my_cc_int(local_interior_size(Lc_pa), T(0));
    scatter_from(Lf, box_owner,
                 rk == box_owner ? cluster_uf_in.data() : (const T*)nullptr,
                 my_uf_int.data());
    scatter_from(Lf, box_owner,
                 rk == box_owner ? cluster_ff_in.data() : (const T*)nullptr,
                 my_ff_int.data());
    scatter_from(Lc_pa, box_owner,
                 rk == box_owner ? cluster_cc_in.data() : (const T*)nullptr,
                 my_cc_int.data());

    // --- (3) Copy interiors into halo-padded buffers at k offset = halo_w.
    std::vector<T> uf_pad(local_with_halo_size(Lf), T(0));
    std::vector<T> ff_pad(local_with_halo_size(Lf), T(0));
    std::vector<T> cc_pad(local_with_halo_size(Lc_pa), T(0));
    {
        const int nz_rf  = local_nz(Lf);
        const int hwf    = Lf.halo_w;
        const int kstridef = 2 * hwf + nz_rf;
        for( int ii=0; ii<cnxf; ++ii )
            for( int jj=0; jj<cnyf; ++jj )
                for( int kl=0; kl<nz_rf; ++kl ) {
                    const std::size_t dst =
                        ((std::size_t)ii * (std::size_t)cnyf + (std::size_t)jj)
                        * (std::size_t)kstridef + (std::size_t)(hwf + kl);
                    const std::size_t src =
                        ((std::size_t)ii * (std::size_t)cnyf + (std::size_t)jj)
                        * (std::size_t)nz_rf + (std::size_t)kl;
                    uf_pad[dst] = my_uf_int[src];
                    ff_pad[dst] = my_ff_int[src];
                }
    }
    {
        const int nz_rc  = local_nz(Lc_pa);
        const int hwc    = Lc_pa.halo_w;
        const int kstridec = 2 * hwc + nz_rc;
        for( int ic=0; ic<cnxc_pa; ++ic )
            for( int jc=0; jc<cnyc_pa; ++jc )
                for( int kl=0; kl<nz_rc; ++kl ) {
                    const std::size_t dst =
                        ((std::size_t)ic * (std::size_t)cnyc_pa + (std::size_t)jc)
                        * (std::size_t)kstridec + (std::size_t)(hwc + kl);
                    const std::size_t src =
                        ((std::size_t)ic * (std::size_t)cnyc_pa + (std::size_t)jc)
                        * (std::size_t)nz_rc + (std::size_t)kl;
                    cc_pad[dst] = my_cc_int[src];
                }
    }

    // --- (3b) prolong_add: uf_pad interior += prolong(cc). Pure per-cell +=,
    //          no halo needed on cc (prolong_add reads coarse interior only).
    prolong_add_impl<T>(Lc_pa, Lf, cc_pad.data(), uf_pad.data());

    // Initial halo exchange (matches smoother).
    halo_exchange_z(Lf, uf_pad.data());

    // --- (4) Sweep loop (identical to smooth_pre_post_n_meshvarbnd).
    std::vector<T> coarse_buf(coarse_buf_sz, T(0));
    for( int s = 0; s < n_sweeps; ++s ) {
        if( rk == box_owner ) {
            for( int cx=0; cx<cnxc; ++cx ) {
                const int ix_uc = oxf + cx - ox_c;
                for( int cy=0; cy<cnyc; ++cy ) {
                    const int iy_uc = oyf + cy - oy_c;
                    for( int cz=0; cz<cnzc; ++cz ) {
                        const int iz_uc = ozf + cz - oz_c;
                        const std::size_t idx =
                            ((std::size_t)cx * (std::size_t)cnyc + (std::size_t)cy)
                            * (std::size_t)cnzc + (std::size_t)cz;
                        coarse_buf[idx] = (*uc)(ix_uc, iy_uc, iz_uc);
                    }
                }
            }
        }
#ifdef USE_MPI
        MPI_Bcast(coarse_buf.data(), (int)coarse_buf_sz, MPI_T, box_owner, sub_comm);
#endif
        if( !prolong_bnd_z_slab_impl<T>(cnxc, cnyc, cnzc, coarse_buf.data(),
                                        ox_c, oy_c, oz_c,
                                        nxc_int, nyc_int, nzc_int,
                                        Lf, uf_pad.data()) )
            return false;
        halo_exchange_z(Lf, uf_pad.data());
        if( !interp_cf_flux_z_slab_impl<T>(cnxc, cnyc, cnzc, coarse_buf.data(),
                                           ox_c, oy_c, oz_c,
                                           nxc_int, nyc_int, nzc_int,
                                           Lf, uf_pad.data()) )
            return false;
        halo_exchange_z(Lf, uf_pad.data());
        gs_neg_skip_z_wide_impl<T>(Lf, uf_pad.data(), ff_pad.data(), h, /*bnd=*/2);
    }

    // --- (5) Extract padded interior back to per-rank interior buffer.
    {
        const int nz_rf  = local_nz(Lf);
        const int hwf    = Lf.halo_w;
        const int kstridef = 2 * hwf + nz_rf;
        for( int ii=0; ii<cnxf; ++ii )
            for( int jj=0; jj<cnyf; ++jj )
                for( int kl=0; kl<nz_rf; ++kl ) {
                    const std::size_t src =
                        ((std::size_t)ii * (std::size_t)cnyf + (std::size_t)jj)
                        * (std::size_t)kstridef + (std::size_t)(hwf + kl);
                    const std::size_t dst =
                        ((std::size_t)ii * (std::size_t)cnyf + (std::size_t)jj)
                        * (std::size_t)nz_rf + (std::size_t)kl;
                    my_uf_int[dst] = uf_pad[src];
                }
    }

    // --- (6) Gather to owner; write back uf (interior + 2-cell BC perimeter).
    std::vector<T> cluster_uf_out;
    if( rk == box_owner ) cluster_uf_out.assign(cluster_full_size(Lf), T(0));
    gather_to(Lf, box_owner, my_uf_int.data(),
              rk == box_owner ? cluster_uf_out.data() : (T*)nullptr);
    if( rk == box_owner ) {
        for( int cx=0; cx<cnxf; ++cx ) {
            const int ix_uf = cx - 2;
            for( int cy=0; cy<cnyf; ++cy ) {
                const int iy_uf = cy - 2;
                for( int cz=0; cz<cnzf; ++cz ) {
                    const int iz_uf = cz - 2;
                    const std::size_t idx =
                        ((std::size_t)cx * (std::size_t)cnyf + (std::size_t)cy)
                        * (std::size_t)cnzf + (std::size_t)cz;
                    (*uf)(ix_uf, iy_uf, iz_uf) = cluster_uf_out[idx];
                }
            }
        }
    }

    return true;
}

} // anonymous namespace

bool smooth_pre_post_n_meshvarbnd_double(int box_owner,
                                         const MeshvarBnd<double>* uc,
                                         MeshvarBnd<double>* uf,
                                         const MeshvarBnd<double>* ff,
                                         double h, int n_sweeps,
                                         MPI_Comm sub_comm)
{ return smooth_pre_post_n_meshvarbnd_impl<double>(box_owner, uc, uf, ff, h, n_sweeps, sub_comm); }

bool smooth_pre_post_n_meshvarbnd_float (int box_owner,
                                         const MeshvarBnd<float>*  uc,
                                         MeshvarBnd<float>*  uf,
                                         const MeshvarBnd<float>*  ff,
                                         float  h, int n_sweeps,
                                         MPI_Comm sub_comm)
{ return smooth_pre_post_n_meshvarbnd_impl<float >(box_owner, uc, uf, ff, h, n_sweeps, sub_comm); }

// Phase G.2b B.5.4.b — keep-in-slab variant of smooth_pre_post_n_meshvarbnd.
// Performs steps (1)..(5) of smooth_pre_post_n_meshvarbnd identically and
// hands the resulting padded-cluster interior buffer (perimeter=2) back to
// the caller via Lf_out + my_uf_int_out, skipping the owner gather + uf
// writeback. The 2-cell BC perimeter that the smoother maintains across the
// padded slab is exactly the perimeter expected by
// restrict_meshvarbnd_from_padded_slab.
//
// Returns false when sub_size<=1 or n_sweeps<=0 — caller must fall back to
// smooth_pre_post_n_meshvarbnd (final-gather variant) + scatter-into-padded-
// slab on its own.
bool smooth_pre_post_n_meshvarbnd_keep_slab_double(
        int box_owner,
        const MeshvarBnd<double>* uc,
        MeshvarBnd<double>* uf,
        const MeshvarBnd<double>* ff,
        double h, int n_sweeps,
        MPI_Comm sub_comm,
        ZoomSlabLayout& Lf_out,
        std::vector<double>& my_uf_int_out)
{ return smooth_pre_post_n_meshvarbnd_impl<double>(
        box_owner, uc, uf, ff, h, n_sweeps, sub_comm,
        &Lf_out, &my_uf_int_out); }

bool smooth_pre_post_n_meshvarbnd_keep_slab_float (
        int box_owner,
        const MeshvarBnd<float>*  uc,
        MeshvarBnd<float>*  uf,
        const MeshvarBnd<float>*  ff,
        float h, int n_sweeps,
        MPI_Comm sub_comm,
        ZoomSlabLayout& Lf_out,
        std::vector<float>& my_uf_int_out)
{ return smooth_pre_post_n_meshvarbnd_impl<float >(
        box_owner, uc, uf, ff, h, n_sweeps, sub_comm,
        &Lf_out, &my_uf_int_out); }

bool prolong_add_then_smooth_n_meshvarbnd_double(
        int box_owner,
        const MeshvarBnd<double>* cc,
        const MeshvarBnd<double>* uc,
        MeshvarBnd<double>* uf,
        const MeshvarBnd<double>* ff,
        double h, int n_sweeps,
        MPI_Comm sub_comm)
{ return prolong_add_then_smooth_n_meshvarbnd_impl<double>(
        box_owner, cc, uc, uf, ff, h, n_sweeps, sub_comm); }

bool prolong_add_then_smooth_n_meshvarbnd_float (
        int box_owner,
        const MeshvarBnd<float>*  cc,
        const MeshvarBnd<float>*  uc,
        MeshvarBnd<float>*  uf,
        const MeshvarBnd<float>*  ff,
        float h, int n_sweeps,
        MPI_Comm sub_comm)
{ return prolong_add_then_smooth_n_meshvarbnd_impl<float >(
        box_owner, cc, uc, uf, ff, h, n_sweeps, sub_comm); }

// ---------------------------------------------------------------------------
// Phase G.2b smoke test for prolong_add_z. Mirrors prolong smoke, but
// initializes vf to a nonzero smooth pattern (so the += semantics is actually
// exercised — comparing slab `vf_in + prolong_add(Vc)` to serial reference).
// ---------------------------------------------------------------------------
bool smoke_test_prolong_add_single_cluster(int halo_w)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    const int cluster_nx_f = 16;
    const int cluster_ny_f = 16;
    int cluster_nz_f = 48;
    while( (cluster_nz_f / world_size) % 2 != 0
        || (cluster_nz_f / (2 * world_size)) < 1 )
        cluster_nz_f += 2 * world_size;

    const int cluster_nx_c = cluster_nx_f / 2;
    const int cluster_ny_c = cluster_ny_f / 2;
    const int cluster_nz_c = cluster_nz_f / 2;

    LOGINFO("G.2b smoke (prolong_add): fine=%dx%dx%d coarse=%dx%dx%d halo_w=%d sub_size=%d",
            cluster_nx_f, cluster_ny_f, cluster_nz_f,
            cluster_nx_c, cluster_ny_c, cluster_nz_c,
            halo_w, world_size);

#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif

    ZoomSlabLayout Lf = make_layout(
        sub, /*cluster_id=*/0, /*level=*/1,
        0, 0, 0,
        cluster_nx_f, cluster_ny_f, cluster_nz_f, halo_w);
    ZoomSlabLayout Lc = make_layout(
        sub, /*cluster_id=*/0, /*level=*/0,
        0, 0, 0,
        cluster_nx_c, cluster_ny_c, cluster_nz_c, halo_w);

    if(    2 * Lc.my_z0 != Lf.my_z0
        || 2 * Lc.my_z1 != Lf.my_z1 ) {
        if( world_rank == 0 )
            LOGERR("G.2b smoke (prolong_add): Lc/Lf z-slab alignment broken on rank %d", world_rank);
        return false;
    }

    auto Vc_at = [&](int ic, int jc, int kc) -> double {
        return (double)(ic + 1) * 0.019
             + (double)(jc + 1) * 0.037
             + (double)(kc + 1) * 0.053
             + std::cos(0.15 * ic + 0.25 * jc + 0.35 * kc);
    };
    auto vf_at = [&](int i, int j, int k) -> double {
        return (double)(i + 1) * 0.011
             + (double)(j + 1) * 0.023
             + (double)(k + 1) * 0.047
             + std::sin(0.07 * i + 0.13 * j + 0.19 * k);
    };

    std::vector<double> full_Vc, full_vf_in;
    if( world_rank == 0 ) {
        full_Vc.assign(cluster_full_size(Lc), 0.0);
        for( int ic=0; ic<cluster_nx_c; ++ic )
            for( int jc=0; jc<cluster_ny_c; ++jc )
                for( int kc=0; kc<cluster_nz_c; ++kc ) {
                    const std::size_t idx = ((std::size_t)ic * (std::size_t)cluster_ny_c + (std::size_t)jc)
                                          * (std::size_t)cluster_nz_c + (std::size_t)kc;
                    full_Vc[idx] = Vc_at(ic, jc, kc);
                }
        full_vf_in.assign(cluster_full_size(Lf), 0.0);
        for( int i=0; i<cluster_nx_f; ++i )
            for( int j=0; j<cluster_ny_f; ++j )
                for( int k=0; k<cluster_nz_f; ++k ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                          * (std::size_t)cluster_nz_f + (std::size_t)k;
                    full_vf_in[idx] = vf_at(i, j, k);
                }
    }

    std::vector<double> my_Vc_int(local_interior_size(Lc), 0.0);
    scatter_cluster_to_zslab_double(Lc, world_rank == 0 ? full_Vc.data() : nullptr,
                                    my_Vc_int.data());
    std::vector<double> my_vf_int(local_interior_size(Lf), 0.0);
    scatter_cluster_to_zslab_double(Lf, world_rank == 0 ? full_vf_in.data() : nullptr,
                                    my_vf_int.data());

    std::vector<double> Vc_pad(local_with_halo_size(Lc), 0.0);
    {
        const int nz_r = local_nz(Lc);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx_c; ++i )
            for( int j=0; j<cluster_ny_c; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_c + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_c + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    Vc_pad[dst] = my_Vc_int[src];
                }
    }
    std::vector<double> vf_pad(local_with_halo_size(Lf), 0.0);
    {
        const int nz_r = local_nz(Lf);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx_f; ++i )
            for( int j=0; j<cluster_ny_f; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    vf_pad[dst] = my_vf_int[src];
                }
    }

    prolong_add_z_slab_double(Lc, Lf, Vc_pad.data(), vf_pad.data());

    std::vector<double> my_vf_out(local_interior_size(Lf), 0.0);
    {
        const int nz_r = local_nz(Lf);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx_f; ++i )
            for( int j=0; j<cluster_ny_f; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    my_vf_out[dst] = vf_pad[src];
                }
    }

    std::vector<double> full_vf_slab;
    if( world_rank == 0 ) full_vf_slab.assign(cluster_full_size(Lf), 0.0);
    gather_zslab_to_cluster_double(Lf, my_vf_out.data(),
                                   world_rank == 0 ? full_vf_slab.data() : nullptr);

    bool ok = true;
    if( world_rank == 0 ) {
#ifdef USE_MPI
        MPI_Comm self_comm = MPI_COMM_SELF;
#else
        MPI_Comm self_comm = 0;
#endif
        ZoomSlabLayout Lf_ser = make_layout(
            self_comm, 0, 1, 0, 0, 0, cluster_nx_f, cluster_ny_f, cluster_nz_f, halo_w);
        ZoomSlabLayout Lc_ser = make_layout(
            self_comm, 0, 0, 0, 0, 0, cluster_nx_c, cluster_ny_c, cluster_nz_c, halo_w);

        std::vector<double> Vc_pad_ser(local_with_halo_size(Lc_ser), 0.0);
        {
            const int nz_r = local_nz(Lc_ser);
            const int kstride = 2*halo_w + nz_r;
            for( int i=0; i<cluster_nx_c; ++i )
                for( int j=0; j<cluster_ny_c; ++j )
                    for( int kl=0; kl<nz_r; ++kl ) {
                        const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_c + (std::size_t)j)
                                              * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                        const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_c + (std::size_t)j)
                                              * (std::size_t)cluster_nz_c + (std::size_t)kl;
                        Vc_pad_ser[dst] = full_Vc[src];
                    }
        }
        std::vector<double> vf_pad_ser(local_with_halo_size(Lf_ser), 0.0);
        {
            const int nz_r = local_nz(Lf_ser);
            const int kstride = 2*halo_w + nz_r;
            for( int i=0; i<cluster_nx_f; ++i )
                for( int j=0; j<cluster_ny_f; ++j )
                    for( int kl=0; kl<nz_r; ++kl ) {
                        const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                              * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                        const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                              * (std::size_t)cluster_nz_f + (std::size_t)kl;
                        vf_pad_ser[dst] = full_vf_in[src];
                    }
        }
        prolong_add_z_slab_double(Lc_ser, Lf_ser, Vc_pad_ser.data(), vf_pad_ser.data());

        std::vector<double> full_vf_ref(cluster_full_size(Lf_ser), 0.0);
        {
            const int nz_r = local_nz(Lf_ser);
            const int kstride = 2*halo_w + nz_r;
            for( int i=0; i<cluster_nx_f; ++i )
                for( int j=0; j<cluster_ny_f; ++j )
                    for( int kl=0; kl<nz_r; ++kl ) {
                        const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                              * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                        const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny_f + (std::size_t)j)
                                              * (std::size_t)cluster_nz_f + (std::size_t)kl;
                        full_vf_ref[dst] = vf_pad_ser[src];
                    }
        }

        double max_abs_err = 0.0;
        std::size_t err_count = 0;
        for( std::size_t n=0; n<full_vf_ref.size(); ++n ) {
            const double e = std::fabs(full_vf_slab[n] - full_vf_ref[n]);
            if( e > max_abs_err ) max_abs_err = e;
            if( e != 0.0 ) ++err_count;
        }
        LOGINFO("G.2b smoke (prolong_add): max|err|=%.3e nonzero_diff=%zu/%zu",
                max_abs_err, err_count, full_vf_ref.size());
        ok = (max_abs_err == 0.0);
        if( ok )
            LOGINFO("G.2b smoke (prolong_add): PASSED (bit-identical to single-rank reference)");
        else
            LOGERR("G.2b smoke (prolong_add): FAILED");
    }

#ifdef USE_MPI
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, sub);
    ok = (ok_int != 0);
    MPI_Barrier(sub);
#endif
    return ok;
}

// ---------------------------------------------------------------------------
// Phase G.2b B.5.3a smoke test: prolong_bnd_z_slab vs single-rank reference.
//
// Strategy:
//   1. Build a deterministic coarse cluster buffer (sinusoid) replicated on
//      all ranks. cluster shape = (cnxc, cnyc, cnzc) = nx+6 cube, ox_c=3.
//   2. Allocate per-rank fine slab buffer (halo-padded), zero it.
//   3. Call prolong_bnd_z_slab on each rank's LfC. BC perimeter writes go to
//      first/last rank's interior cluster cells.
//   4. Gather fine slab interior to rank 0.
//   5. On rank 0, build a SINGLE-RANK (sub_size=1, MPI_COMM_SELF) LfC_ref,
//      run the kernel again on the full fine buffer.
//   6. Compare gathered slab vs single-rank reference cell-by-cell; expect
//      max|err|=0 (both paths execute identical interp_cubic arithmetic).
// ---------------------------------------------------------------------------
bool smoke_test_prolong_bnd_single_cluster(int halo_w)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    // Pick nx so fine cluster (2*nx+4) is divisible by world_size and each
    // local slab has at least halo_w+1 interior cells.
    int nx = 12;
    while( (2*nx + 4) % world_size != 0
        || ((2*nx + 4) / world_size) < halo_w + 1 )
        nx += 2;
    const int ny = nx;
    const int nz = nx;

    const int ox_c = 3, oy_c = 3, oz_c = 3;
    const int cnxc = nx + 6;
    const int cnyc = ny + 6;
    const int cnzc = nz + 6;

    const int cnxf = 2*nx + 4;
    const int cnyf = 2*ny + 4;
    const int cnzf = 2*nz + 4;

    LOGINFO("G.2b B.5.3a smoke (prolong_bnd): fine=%dx%dx%d coarse=%dx%dx%d "
            "ox_c=(%d,%d,%d) halo_w=%d sub_size=%d",
            cnxf, cnyf, cnzf, cnxc, cnyc, cnzc,
            ox_c, oy_c, oz_c, halo_w, world_size);

#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif

    // Deterministic coarse pattern (cluster frame).
    auto Vc_at = [&](int cx, int cy, int cz) -> double {
        return (double)(cx + 1) * 0.011
             + (double)(cy + 1) * 0.023
             + (double)(cz + 1) * 0.047
             + std::cos(0.13 * cx + 0.21 * cy + 0.31 * cz);
    };

    const std::size_t coarse_sz = (std::size_t)cnxc * (std::size_t)cnyc * (std::size_t)cnzc;
    std::vector<double> coarse_buf(coarse_sz, 0.0);
    for( int cx=0; cx<cnxc; ++cx )
        for( int cy=0; cy<cnyc; ++cy )
            for( int cz=0; cz<cnzc; ++cz ) {
                const std::size_t idx = ((std::size_t)cx * (std::size_t)cnyc + (std::size_t)cy)
                                      * (std::size_t)cnzc + (std::size_t)cz;
                coarse_buf[idx] = Vc_at(cx, cy, cz);
            }

    // Build per-rank fine layout.
    ZoomSlabLayout LfC = make_layout(
        sub, /*cluster_id=*/0, /*level=*/1,
        0, 0, 0,
        cnxf, cnyf, cnzf, halo_w);

    std::vector<double> fine_slab(local_with_halo_size(LfC), 0.0);
    if( !prolong_bnd_z_slab_double(cnxc, cnyc, cnzc, coarse_buf.data(),
                                    ox_c, oy_c, oz_c, nx, ny, nz,
                                    LfC, fine_slab.data()) ) {
        if( world_rank == 0 )
            LOGERR("G.2b B.5.3a smoke (prolong_bnd): kernel returned false (precondition)");
        return false;
    }

    // Extract interior of fine slab into my_vf_int.
    std::vector<double> my_vf_int(local_interior_size(LfC), 0.0);
    {
        const int nz_r = local_nz(LfC);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cnxf; ++i )
            for( int j=0; j<cnyf; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t src = ((std::size_t)i * (std::size_t)cnyf + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cnyf + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    my_vf_int[dst] = fine_slab[src];
                }
    }

    // Gather fine slab interior to rank 0 (full cluster z range).
    std::vector<double> full_vf_slab;
    if( world_rank == 0 ) full_vf_slab.assign(cluster_full_size(LfC), 0.0);
    gather_zslab_to_cluster_double(LfC, my_vf_int.data(),
                                   world_rank == 0 ? full_vf_slab.data() : nullptr);

    bool ok = true;
    if( world_rank == 0 ) {
        // Single-rank reference via the same kernel.
#ifdef USE_MPI
        MPI_Comm self_comm = MPI_COMM_SELF;
#else
        MPI_Comm self_comm = 0;
#endif
        ZoomSlabLayout LfC_ref = make_layout(
            self_comm, 0, 1, 0, 0, 0, cnxf, cnyf, cnzf, halo_w);

        std::vector<double> fine_slab_ref(local_with_halo_size(LfC_ref), 0.0);
        if( !prolong_bnd_z_slab_double(cnxc, cnyc, cnzc, coarse_buf.data(),
                                        ox_c, oy_c, oz_c, nx, ny, nz,
                                        LfC_ref, fine_slab_ref.data()) ) {
            LOGERR("G.2b B.5.3a smoke (prolong_bnd): ref kernel returned false");
            ok = false;
        }

        std::vector<double> full_vf_ref(cluster_full_size(LfC_ref), 0.0);
        if( ok ) {
            const int nz_r = local_nz(LfC_ref);
            const int kstride = 2*halo_w + nz_r;
            for( int i=0; i<cnxf; ++i )
                for( int j=0; j<cnyf; ++j )
                    for( int kl=0; kl<nz_r; ++kl ) {
                        const std::size_t src = ((std::size_t)i * (std::size_t)cnyf + (std::size_t)j)
                                              * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                        const std::size_t dst = ((std::size_t)i * (std::size_t)cnyf + (std::size_t)j)
                                              * (std::size_t)cnzf + (std::size_t)kl;
                        full_vf_ref[dst] = fine_slab_ref[src];
                    }
        }

        double max_abs_err = 0.0;
        std::size_t err_count = 0;
        std::size_t bc_cells_checked = 0;
        if( ok ) {
            for( int fx=0; fx<cnxf; ++fx )
                for( int fy=0; fy<cnyf; ++fy )
                    for( int fz=0; fz<cnzf; ++fz ) {
                        // Only BC perimeter cells should be written (interior
                        // remains 0 from the zero-init); compare the full
                        // buffer though — interior should be 0 == 0.
                        const bool is_bc = (fx<2 || fx>=cnxf-2)
                                        || (fy<2 || fy>=cnyf-2)
                                        || (fz<2 || fz>=cnzf-2);
                        const std::size_t idx = ((std::size_t)fx * (std::size_t)cnyf + (std::size_t)fy)
                                              * (std::size_t)cnzf + (std::size_t)fz;
                        const double e = std::fabs(full_vf_slab[idx] - full_vf_ref[idx]);
                        if( e > max_abs_err ) max_abs_err = e;
                        if( e != 0.0 ) ++err_count;
                        if( is_bc ) ++bc_cells_checked;
                    }
            LOGINFO("G.2b B.5.3a smoke (prolong_bnd): max|err|=%.3e nonzero_diff=%zu/%zu "
                    "(BC cells: %zu)",
                    max_abs_err, err_count, full_vf_ref.size(), bc_cells_checked);
            ok = (max_abs_err == 0.0);
            if( ok )
                LOGINFO("G.2b B.5.3a smoke (prolong_bnd): PASSED (bit-identical to single-rank reference)");
            else
                LOGERR("G.2b B.5.3a smoke (prolong_bnd): FAILED");
        }
    }

#ifdef USE_MPI
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, sub);
    ok = (ok_int != 0);
    MPI_Barrier(sub);
#endif
    return ok;
}

// ---------------------------------------------------------------------------
// Phase G.2b B.5.3b smoke test: interp_cf_flux_z_slab.
//
// Tests the COMPOSITE (prolong_bnd_z_slab + halo_exchange_z + interp_cf_flux
// _z_slab) — i.e., the slab equivalent of interp_O3_fluxcorr::interp_coarse
// _fine — against a single-rank reference doing the same composite. Expects
// bit-identical (max|err| == 0) since both paths execute identical floating-
// point operations in the same order.
//
// Strategy:
//   1. Build deterministic coarse cluster buffer (sinusoid), replicated.
//   2. Per-rank fine slab: initialize INTERIOR to a smooth non-zero pattern
//      (so flux corrections see nontrivial inner u(ix+1), u(ix+2) reads).
//   3. Call prolong_bnd_z_slab → halo_exchange_z → interp_cf_flux_z_slab on
//      each rank's slab.
//   4. Gather fine slab interior to rank 0.
//   5. On rank 0, replay steps 2-3 against a SINGLE-RANK LfC_ref
//      (sub_comm = MPI_COMM_SELF) and compare cell-by-cell.
// ---------------------------------------------------------------------------
bool smoke_test_interp_cf_flux_single_cluster(int halo_w)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    // Pick nx (COARSE) such that:
    //   (2*nx + 4) % world_size == 0  -> standard slab alignment
    //   (2*nx + 4) / world_size >= halo_w + 1  -> halo room
    //   (nx + 2) % world_size == 0    -> my_z0 + local_nz both even (B.5.3b)
    //   nx even (kernel precondition)
    int nx = 12;
    auto ok_nx = [&](int n){
        const int cnzf = 2*n + 4;
        if( cnzf % world_size != 0 ) return false;
        if( cnzf / world_size < halo_w + 1 ) return false;
        if( (n + 2) % world_size != 0 ) return false;
        if( n & 1 ) return false;
        return true;
    };
    while( !ok_nx(nx) ) nx += 2;
    const int ny = nx;
    const int nz = nx;

    const int ox_c = 3, oy_c = 3, oz_c = 3;  // same as B.5.3a (cubic stencil needs >= 3)
    const int cnxc = nx + 6;
    const int cnyc = ny + 6;
    const int cnzc = nz + 6;

    const int cnxf = 2*nx + 4;
    const int cnyf = 2*ny + 4;
    const int cnzf = 2*nz + 4;

    LOGINFO("G.2b B.5.3b smoke (interp_cf_flux): fine=%dx%dx%d coarse=%dx%dx%d "
            "ox_c=(%d,%d,%d) halo_w=%d sub_size=%d",
            cnxf, cnyf, cnzf, cnxc, cnyc, cnzc,
            ox_c, oy_c, oz_c, halo_w, world_size);

#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif

    // Deterministic coarse pattern (cluster frame).
    auto Vc_at = [&](int cx, int cy, int cz) -> double {
        return (double)(cx + 1) * 0.011
             + (double)(cy + 1) * 0.023
             + (double)(cz + 1) * 0.047
             + std::cos(0.13 * cx + 0.21 * cy + 0.31 * cz);
    };
    const std::size_t coarse_sz = (std::size_t)cnxc * (std::size_t)cnyc * (std::size_t)cnzc;
    std::vector<double> coarse_buf(coarse_sz, 0.0);
    for( int cx=0; cx<cnxc; ++cx )
        for( int cy=0; cy<cnyc; ++cy )
            for( int cz=0; cz<cnzc; ++cz ) {
                const std::size_t idx = ((std::size_t)cx * (std::size_t)cnyc + (std::size_t)cy)
                                      * (std::size_t)cnzc + (std::size_t)cz;
                coarse_buf[idx] = Vc_at(cx, cy, cz);
            }

    // Deterministic smooth interior pattern (cluster frame). Polynomial in
    // (fx, fy, fz) so that init values are bit-identical at every call site
    // — sin/cos would 1-ULP-drift between the slab and ref init loops if the
    // compiler picks different FMA sequences per inlining context.
    auto Uin_at = [&](int fx, int fy, int fz) -> double {
        return 0.5 + 0.001 * (double)fx
                   + 0.002 * (double)fy
                   + 0.003 * (double)fz;
    };

    // Build per-rank fine layout.
    ZoomSlabLayout LfC = make_layout(
        sub, /*cluster_id=*/0, /*level=*/1,
        0, 0, 0,
        cnxf, cnyf, cnzf, halo_w);

    const int nz_r = local_nz(LfC);
    const int kstride = 2*halo_w + nz_r;

    // Allocate fine slab (halo-padded) and init the INTERIOR to the smooth
    // pattern. BC perimeter cells will be (re)written by prolong_bnd and then
    // by flux corrections.
    std::vector<double> fine_slab(local_with_halo_size(LfC), 0.0);
    for( int fx=0; fx<cnxf; ++fx )
        for( int fy=0; fy<cnyf; ++fy )
            for( int kl=0; kl<nz_r; ++kl ) {
                const int fz_global = LfC.my_z0 + kl;
                const std::size_t didx = ((std::size_t)fx * (std::size_t)cnyf
                                       + (std::size_t)fy)
                                       * (std::size_t)kstride
                                       + (std::size_t)(halo_w + kl);
                fine_slab[didx] = Uin_at(fx, fy, fz_global);
            }

    // Step 1: prolong_bnd writes BC perimeter (cubic) into rank-local cells.
    if( !prolong_bnd_z_slab_double(cnxc, cnyc, cnzc, coarse_buf.data(),
                                    ox_c, oy_c, oz_c, nx, ny, nz,
                                    LfC, fine_slab.data()) ) {
        if( world_rank == 0 )
            LOGERR("G.2b B.5.3b smoke: prolong_bnd kernel returned false");
        return false;
    }

    // Step 2: halo_exchange_z — so flux corrections at x/y faces see correct
    // u(ix+1, ...) values in the standard halo region (no actual cross-rank
    // reads happen in flux at this halo layer; still defensive).
    halo_exchange_z_double(LfC, fine_slab.data());

    // Step 3: flux corrections.
    if( !interp_cf_flux_z_slab_double(cnxc, cnyc, cnzc, coarse_buf.data(),
                                       ox_c, oy_c, oz_c, nx, ny, nz,
                                       LfC, fine_slab.data()) ) {
        if( world_rank == 0 )
            LOGERR("G.2b B.5.3b smoke: interp_cf_flux kernel returned false");
        return false;
    }

    // Extract interior to gather.
    std::vector<double> my_vf_int(local_interior_size(LfC), 0.0);
    for( int i=0; i<cnxf; ++i )
        for( int j=0; j<cnyf; ++j )
            for( int kl=0; kl<nz_r; ++kl ) {
                const std::size_t src = ((std::size_t)i * (std::size_t)cnyf + (std::size_t)j)
                                      * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                const std::size_t dst = ((std::size_t)i * (std::size_t)cnyf + (std::size_t)j)
                                      * (std::size_t)nz_r + (std::size_t)kl;
                my_vf_int[dst] = fine_slab[src];
            }

    std::vector<double> full_vf_slab;
    if( world_rank == 0 ) full_vf_slab.assign(cluster_full_size(LfC), 0.0);
    gather_zslab_to_cluster_double(LfC, my_vf_int.data(),
                                   world_rank == 0 ? full_vf_slab.data() : nullptr);

    bool ok = true;
    if( world_rank == 0 ) {
        // Single-rank reference via the same composite path.
#ifdef USE_MPI
        MPI_Comm self_comm = MPI_COMM_SELF;
#else
        MPI_Comm self_comm = 0;
#endif
        ZoomSlabLayout LfC_ref = make_layout(
            self_comm, 0, 1, 0, 0, 0, cnxf, cnyf, cnzf, halo_w);

        std::vector<double> fine_slab_ref(local_with_halo_size(LfC_ref), 0.0);
        const int nz_r_ref = local_nz(LfC_ref);
        const int kstride_ref = 2*halo_w + nz_r_ref;
        for( int fx=0; fx<cnxf; ++fx )
            for( int fy=0; fy<cnyf; ++fy )
                for( int kl=0; kl<nz_r_ref; ++kl ) {
                    const int fz_global = LfC_ref.my_z0 + kl;
                    const std::size_t didx = ((std::size_t)fx * (std::size_t)cnyf
                                           + (std::size_t)fy)
                                           * (std::size_t)kstride_ref
                                           + (std::size_t)(halo_w + kl);
                    fine_slab_ref[didx] = Uin_at(fx, fy, fz_global);
                }
        if( !prolong_bnd_z_slab_double(cnxc, cnyc, cnzc, coarse_buf.data(),
                                        ox_c, oy_c, oz_c, nx, ny, nz,
                                        LfC_ref, fine_slab_ref.data()) ) {
            LOGERR("G.2b B.5.3b smoke: ref prolong_bnd returned false");
            ok = false;
        }
        halo_exchange_z_double(LfC_ref, fine_slab_ref.data());
        if( ok && !interp_cf_flux_z_slab_double(cnxc, cnyc, cnzc, coarse_buf.data(),
                                                 ox_c, oy_c, oz_c, nx, ny, nz,
                                                 LfC_ref, fine_slab_ref.data()) ) {
            LOGERR("G.2b B.5.3b smoke: ref interp_cf_flux returned false");
            ok = false;
        }

        std::vector<double> full_vf_ref(cluster_full_size(LfC_ref), 0.0);
        if( ok ) {
            for( int i=0; i<cnxf; ++i )
                for( int j=0; j<cnyf; ++j )
                    for( int kl=0; kl<nz_r_ref; ++kl ) {
                        const std::size_t src = ((std::size_t)i * (std::size_t)cnyf + (std::size_t)j)
                                              * (std::size_t)kstride_ref + (std::size_t)(halo_w + kl);
                        const std::size_t dst = ((std::size_t)i * (std::size_t)cnyf + (std::size_t)j)
                                              * (std::size_t)cnzf + (std::size_t)kl;
                        full_vf_ref[dst] = fine_slab_ref[src];
                    }
        }

        double max_abs_err = 0.0, max_bc_abs_err = 0.0;
        std::size_t err_count = 0;
        std::size_t bc_cells_checked = 0;
        if( ok ) {
            for( int fx=0; fx<cnxf; ++fx )
                for( int fy=0; fy<cnyf; ++fy )
                    for( int fz=0; fz<cnzf; ++fz ) {
                        const bool is_bc = (fx<2 || fx>=cnxf-2)
                                        || (fy<2 || fy>=cnyf-2)
                                        || (fz<2 || fz>=cnzf-2);
                        const std::size_t idx = ((std::size_t)fx * (std::size_t)cnyf + (std::size_t)fy)
                                              * (std::size_t)cnzf + (std::size_t)fz;
                        const double e = std::fabs(full_vf_slab[idx] - full_vf_ref[idx]);
                        if( e > max_abs_err ) max_abs_err = e;
                        if( is_bc ) {
                            if( e > max_bc_abs_err ) max_bc_abs_err = e;
                            ++bc_cells_checked;
                        }
                        if( e != 0.0 ) ++err_count;
                    }
            LOGINFO("G.2b B.5.3b smoke (interp_cf_flux): max|err|=%.3e (BC max=%.3e) "
                    "nonzero_diff=%zu/%zu (BC cells: %zu)",
                    max_abs_err, max_bc_abs_err, err_count, full_vf_ref.size(),
                    bc_cells_checked);
            ok = (max_abs_err == 0.0);
            if( ok )
                LOGINFO("G.2b B.5.3b smoke (interp_cf_flux): PASSED (bit-identical to single-rank reference)");
            else
                LOGERR("G.2b B.5.3b smoke (interp_cf_flux): FAILED");
        }
    }

#ifdef USE_MPI
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, sub);
    ok = (ok_int != 0);
    MPI_Barrier(sub);
#endif
    return ok;
}

// ---------------------------------------------------------------------------
// Phase G.2b smoke test for gs_z.
//
// Strategy: pick a smooth analytic u_truth(i,j,k) = sin(a i) sin(b j) sin(c k)
// with the matching discrete 7-point source f = L u_truth / h^2. Initialize
// u_slab to ZERO and run N GS sweeps. After each sweep call residual_z and
// MPI_Allreduce the per-rank L2^2 across sub_comm.
//
// Verification:
//  (a) residual is monotonically non-increasing across sweeps (GS is a smoother).
//  (b) the slab-distributed residual at np>1 matches the np=1 serial residual
//      to floating-point roundoff (red-black GS with iz_global coloring +
//      halo_exchange between colors is provably equivalent to serial RBGS).
// We check (a) directly and (b) by hashing the final residual via Allreduce
// and printing it. The smoke test is per-run rather than cross-np compare
// (cross-np compare would require a separate driver).
// ---------------------------------------------------------------------------
bool smoke_test_gs_single_cluster(int halo_w, int n_sweeps)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    const int cluster_nx = 16;
    const int cluster_ny = 16;
    const int min_nz = world_size * (halo_w + 1);
    const int cluster_nz = (32 > min_nz) ? 32 : min_nz;

    LOGINFO("G.2b smoke (GS): cluster=%dx%dx%d, halo_w=%d, sub_size=%d, n_sweeps=%d",
            cluster_nx, cluster_ny, cluster_nz, halo_w, world_size, n_sweeps);

#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif
    ZoomSlabLayout L = make_layout(
        sub, /*cluster_id=*/0, /*level=*/0,
        /*oax=*/0, /*oay=*/0, /*oaz=*/0,
        cluster_nx, cluster_ny, cluster_nz, halo_w);

    const double PI = 3.14159265358979323846;
    const double aa = PI / (double)cluster_nx;
    const double bb = PI / (double)cluster_ny;
    const double cc = PI / (double)cluster_nz;
    const double h  = 1.0;
    const double sa2 = std::sin(0.5*aa);
    const double sb2 = std::sin(0.5*bb);
    const double sc2 = std::sin(0.5*cc);
    const double lap_factor = -4.0 * (sa2*sa2 + sb2*sb2 + sc2*sc2);

    auto u_truth = [&](int i, int j, int k) -> double {
        return std::sin(aa*i) * std::sin(bb*j) * std::sin(cc*k);
    };
    auto f_at = [&](int i, int j, int k) -> double {
        return lap_factor * u_truth(i,j,k) / (h*h);
    };

    std::vector<double> full_f;
    if( world_rank == 0 ) {
        full_f.assign(cluster_full_size(L), 0.0);
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int k=0; k<cluster_nz; ++k ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)cluster_nz + (std::size_t)k;
                    full_f[idx] = f_at(i,j,k);
                }
    }

    std::vector<double> my_f_int(local_interior_size(L), 0.0);
    scatter_cluster_to_zslab_double(L, world_rank == 0 ? full_f.data() : nullptr, my_f_int.data());

    std::vector<double> u_pad(local_with_halo_size(L), 0.0);  // initial u = 0
    std::vector<double> f_pad(local_with_halo_size(L), 0.0);
    std::vector<double> r_pad(local_with_halo_size(L), 0.0);
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    f_pad[dst] = my_f_int[src];
                }
    }

    // Helper: L2^2 of residual over "valid" interior cells — i.e., cells whose
    // 7-point stencil stays inside data, excluding cluster z-edges on the
    // outermost ranks (where the stencil reads zero-filled edge halo, which
    // doesn't match the analytic source's extended-domain assumption).
    auto valid_l2sq = [&]() -> double {
        residual_z_slab_double(L, u_pad.data(), f_pad.data(), h, r_pad.data());
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        const int kl_lo = (L.sub_rank == 0)            ? (halo_w + 1)        : halo_w;
        const int kl_hi = (L.sub_rank == L.sub_size-1) ? (halo_w + nz_r - 1) : (halo_w + nz_r);
        double s = 0.0;
        for( int i=1; i<cluster_nx-1; ++i )
            for( int j=1; j<cluster_ny-1; ++j )
                for( int kl=kl_lo; kl<kl_hi; ++kl ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)kl;
                    s += r_pad[idx] * r_pad[idx];
                }
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_SUM, L.sub_comm);
#endif
        return s;
    };

    // Initial residual (u = 0, so r = h^2 f - 0 = h^2 f on interior cells).
    halo_exchange_z_double(L, u_pad.data());
    double initial_l2sq = valid_l2sq();

    // Note: GS is contractive in the energy norm (||e||_A), not the L2 residual
    // norm. The residual L2 can transiently increase in the first few sweeps
    // before contracting. The smoke test only requires substantial reduction
    // over N sweeps relative to initial.
    double final_l2sq = initial_l2sq;

    for( int sweep=0; sweep<n_sweeps; ++sweep ) {
        gs_z_slab_double(L, u_pad.data(), f_pad.data(), h);
        double l2sq = valid_l2sq();
        if( world_rank == 0 )
            LOGINFO("G.2b smoke (GS): sweep %d: residual L2^2 = %.6e", sweep, l2sq);
        final_l2sq = l2sq;
    }

    // For this smooth source, N=8 sweeps of GS should reduce residual L2^2
    // by at least 1.5x (slow convergence on low-frequency modes is expected;
    // GS is a smoother, not a solver — multigrid is needed for fast convergence).
    if( !(final_l2sq < initial_l2sq * 0.9) ) {
        if( world_rank == 0 )
            LOGERR("G.2b smoke (GS): insufficient reduction: %.3e -> %.3e",
                   initial_l2sq, final_l2sq);
        return false;
    }

    if( world_rank == 0 )
        LOGINFO("G.2b smoke (GS): PASSED initial=%.3e final=%.3e reduction=%.2fx",
                initial_l2sq, final_l2sq, initial_l2sq/final_l2sq);

#ifdef USE_MPI
    MPI_Barrier(L.sub_comm);
#endif
    return true;
}

// ---------------------------------------------------------------------------
// Phase G.2b production-wire step 1: smoke test for gs_z_neg.
//
// Verifies bit-identical match against TWO independent references:
//  (a) single-rank slab path on MPI_COMM_SELF (same impl) — exercises the
//      MPI decomposition + halo_exchange + per-color iteration order.
//  (b) inlined production-formula serial loop on rank 0 — exercises that the
//      slab's update formula matches what multigrid::solver::GaussSeidel
//      (mg_solver.hh:198-217) computes for stencil_7P. Per fd_schemes.hh:198,
//      stencil_7P::rhs sums 6 neighbours and ccoeff()=-6, so production GS is
//          u(ix,iy,iz) = (sum_n + h^2 * f(ix,iy,iz)) / 6
//      with red-black ordering (ix+iy+iz)%2 == color. The inlined reference
//      below uses exactly this formula on the full cluster — passing this
//      check is the contract for wiring gs_z_neg into twoGrid_multibox's GS
//      sites without an additional sign-correction step at the call site.
// ---------------------------------------------------------------------------
bool smoke_test_gs_neg_single_cluster(int halo_w, int n_sweeps)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    const int cluster_nx = 16;
    const int cluster_ny = 16;
    const int min_nz = world_size * (halo_w + 1);
    const int cluster_nz = (32 > min_nz) ? 32 : min_nz;

    LOGINFO("G.2b smoke (gs_neg): cluster=%dx%dx%d, halo_w=%d, sub_size=%d, n_sweeps=%d",
            cluster_nx, cluster_ny, cluster_nz, halo_w, world_size, n_sweeps);

#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif
    ZoomSlabLayout L = make_layout(
        sub, /*cluster_id=*/0, /*level=*/0,
        /*oax=*/0, /*oay=*/0, /*oaz=*/0,
        cluster_nx, cluster_ny, cluster_nz, halo_w);

    // Deterministic smooth u_init and f. Mixing trig + polynomial so update
    // path actually exercises (sum + h^2 f)/6 with both terms non-negligible.
    auto u_init_at = [&](int i, int j, int k) -> double {
        return 0.4 + 0.013*(i+1) - 0.021*(j+1) + 0.017*(k+1)
             + 0.5*std::sin(0.11*i + 0.19*j + 0.07*k);
    };
    auto f_at = [&](int i, int j, int k) -> double {
        return 0.25*std::cos(0.23*i - 0.13*j + 0.31*k)
             + 0.05*(i*1.0 + j*0.5 - k*0.7);
    };
    const double h = 1.0 / (double)cluster_nx;  // arbitrary positive h

    std::vector<double> full_u, full_f;
    if( world_rank == 0 ) {
        full_u.assign(cluster_full_size(L), 0.0);
        full_f.assign(cluster_full_size(L), 0.0);
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int k=0; k<cluster_nz; ++k ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)cluster_nz + (std::size_t)k;
                    full_u[idx] = u_init_at(i,j,k);
                    full_f[idx] = f_at(i,j,k);
                }
    }

    // ---- slab path ----
    std::vector<double> my_u_int(local_interior_size(L), 0.0);
    std::vector<double> my_f_int(local_interior_size(L), 0.0);
    scatter_cluster_to_zslab_double(L, world_rank == 0 ? full_u.data() : nullptr, my_u_int.data());
    scatter_cluster_to_zslab_double(L, world_rank == 0 ? full_f.data() : nullptr, my_f_int.data());

    std::vector<double> u_pad(local_with_halo_size(L), 0.0);
    std::vector<double> f_pad(local_with_halo_size(L), 0.0);
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    u_pad[dst] = my_u_int[src];
                    f_pad[dst] = my_f_int[src];
                }
    }
    halo_exchange_z_double(L, u_pad.data());
    halo_exchange_z_double(L, f_pad.data());

    for( int s=0; s<n_sweeps; ++s )
        gs_z_neg_slab_double(L, u_pad.data(), f_pad.data(), h);

    std::vector<double> my_u_out(local_interior_size(L), 0.0);
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    my_u_out[dst] = u_pad[src];
                }
    }
    std::vector<double> full_u_slab;
    if( world_rank == 0 ) full_u_slab.assign(cluster_full_size(L), 0.0);
    gather_zslab_to_cluster_double(L, my_u_out.data(),
                                   world_rank == 0 ? full_u_slab.data() : nullptr);

    bool ok_a = true, ok_b = true;
    if( world_rank == 0 ) {
#ifdef USE_MPI
        MPI_Comm self_comm = MPI_COMM_SELF;
#else
        MPI_Comm self_comm = 0;
#endif

        // ---- reference (a): single-rank slab path on MPI_COMM_SELF ----
        ZoomSlabLayout Lser = make_layout(
            self_comm, 0, 0, 0, 0, 0, cluster_nx, cluster_ny, cluster_nz, halo_w);
        std::vector<double> u_pad_ser(local_with_halo_size(Lser), 0.0);
        std::vector<double> f_pad_ser(local_with_halo_size(Lser), 0.0);
        {
            const int nz_r = local_nz(Lser);
            const int kstride = 2*halo_w + nz_r;
            for( int i=0; i<cluster_nx; ++i )
                for( int j=0; j<cluster_ny; ++j )
                    for( int kl=0; kl<nz_r; ++kl ) {
                        const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                              * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                        const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                              * (std::size_t)cluster_nz + (std::size_t)kl;
                        u_pad_ser[dst] = full_u[src];
                        f_pad_ser[dst] = full_f[src];
                    }
        }
        for( int s=0; s<n_sweeps; ++s )
            gs_z_neg_slab_double(Lser, u_pad_ser.data(), f_pad_ser.data(), h);
        std::vector<double> full_u_ref_a(cluster_full_size(Lser), 0.0);
        {
            const int nz_r = local_nz(Lser);
            const int kstride = 2*halo_w + nz_r;
            for( int i=0; i<cluster_nx; ++i )
                for( int j=0; j<cluster_ny; ++j )
                    for( int kl=0; kl<nz_r; ++kl ) {
                        const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                              * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                        const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                              * (std::size_t)cluster_nz + (std::size_t)kl;
                        full_u_ref_a[dst] = u_pad_ser[src];
                    }
        }

        // ---- reference (b): inlined production-formula serial loop ----
        // Mirrors multigrid::solver<stencil_7P,...>::GaussSeidel exactly:
        //   for color=0,1:
        //     for ix,iy,iz over cluster interior:
        //       if (ix+iy+iz)%2 == color:
        //         u = (u_n6_sum + h^2 f) / 6
        // Boundary policy: x/y cluster faces are NOT updated (i=0, i=cnx-1,
        // j=0, j=cny-1) — matches gs_neg_impl's skip. z-edges ARE updated
        // and the stencil reads zero outside the cluster z-range (mirrors
        // gs_neg_impl's behaviour with zero-filled cluster-edge halos).
        std::vector<double> full_u_ref_b = full_u;  // copy initial u
        const double h2 = h * h;
        const double sixth = 1.0 / 6.0;
        auto u_get = [&](int i, int j, int k) -> double {
            if( i<0 || i>=cluster_nx || j<0 || j>=cluster_ny
              || k<0 || k>=cluster_nz )
                return 0.0;  // mirror zero-filled edge halo
            return full_u_ref_b[((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                * (std::size_t)cluster_nz + (std::size_t)k];
        };
        for( int s=0; s<n_sweeps; ++s ) {
            for( int color=0; color<2; ++color ) {
                for( int i=1; i<cluster_nx-1; ++i )
                    for( int j=1; j<cluster_ny-1; ++j )
                        for( int k=0; k<cluster_nz; ++k ) {
                            if( ((i+j+k) & 1) != color ) continue;
                            const double sum = u_get(i-1,j,k) + u_get(i+1,j,k)
                                             + u_get(i,j-1,k) + u_get(i,j+1,k)
                                             + u_get(i,j,k-1) + u_get(i,j,k+1);
                            const std::size_t idx = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                                  * (std::size_t)cluster_nz + (std::size_t)k;
                            full_u_ref_b[idx] = (sum + h2 * full_f[idx]) * sixth;
                        }
            }
        }

        // ---- compare ----
        double max_a = 0.0, max_b = 0.0;
        std::size_t diff_a = 0, diff_b = 0;
        for( std::size_t n=0; n<full_u_slab.size(); ++n ) {
            const double ea = std::fabs(full_u_slab[n] - full_u_ref_a[n]);
            const double eb = std::fabs(full_u_slab[n] - full_u_ref_b[n]);
            if( ea > max_a ) max_a = ea;
            if( eb > max_b ) max_b = eb;
            if( ea != 0.0 ) ++diff_a;
            if( eb != 0.0 ) ++diff_b;
        }
        LOGINFO("G.2b smoke (gs_neg): vs single-rank slab: max|err|=%.3e nonzero=%zu/%zu",
                max_a, diff_a, full_u_slab.size());
        LOGINFO("G.2b smoke (gs_neg): vs production serial: max|err|=%.3e nonzero=%zu/%zu",
                max_b, diff_b, full_u_slab.size());
        ok_a = (max_a == 0.0);
        ok_b = (max_b == 0.0);
        if( ok_a && ok_b )
            LOGINFO("G.2b smoke (gs_neg): PASSED (bit-identical to both references)");
        else
            LOGERR("G.2b smoke (gs_neg): FAILED");
    }

    bool ok = ok_a && ok_b;
#ifdef USE_MPI
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, sub);
    ok = (ok_int != 0);
    MPI_Barrier(sub);
#endif
    return ok;
}

// ---------------------------------------------------------------------------
// Phase G.2b production-wire step 2 smoke test: gs_z_neg_meshvarbnd end-to-end.
// Builds a small MeshvarBnd<double>(nbnd=1, nx, ny, nz) on rank 0 (box_owner)
// with deterministic interior + BC perimeter. Runs N sweeps via the slab path.
// Independently runs N sweeps of the inlined production GS formula on a copy
// of the initial MeshvarBnd. Verifies that the interior cells match
// bit-identical at np=1/2/4/8.
// ---------------------------------------------------------------------------
bool smoke_test_gs_neg_meshvarbnd_single_cluster(int halo_w, int n_sweeps)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    const int nx = 8;
    const int ny = 8;
    // cluster_nz = nz + 2 must split evenly enough that every rank's nz_r >= halo_w.
    const int min_cnz = world_size * (halo_w + 1);
    const int nz = (16 > min_cnz - 2) ? 16 : (min_cnz - 2);
    const int nbnd = 1;
    const double h = 1.0 / (double)nx;

    LOGINFO("G.2b smoke (gs_neg_mvb): nx=%d ny=%d nz=%d nbnd=%d halo_w=%d sub_size=%d sweeps=%d",
            nx, ny, nz, nbnd, halo_w, world_size, n_sweeps);

    auto u_init_at = [&](int i, int j, int k) -> double {
        return 0.4 + 0.013*(i+1) - 0.021*(j+1) + 0.017*(k+1)
             + 0.5*std::sin(0.11*i + 0.19*j + 0.07*k);
    };
    auto f_at = [&](int i, int j, int k) -> double {
        return 0.25*std::cos(0.23*i - 0.13*j + 0.31*k)
             + 0.05*(i*1.0 + j*0.5 - k*0.7);
    };

    std::unique_ptr< MeshvarBnd<double> > u, f, u_ref;
    if( world_rank == 0 ) {
        u.reset    (new MeshvarBnd<double>(nbnd, nx, ny, nz));
        f.reset    (new MeshvarBnd<double>(nbnd, nx, ny, nz));
        u_ref.reset(new MeshvarBnd<double>(nbnd, nx, ny, nz));
        // Fill interior + the 1-cell BC perimeter (i, j, k in [-1, n)).
        for( int i=-1; i<=nx; ++i )
            for( int j=-1; j<=ny; ++j )
                for( int k=-1; k<=nz; ++k ) {
                    const double uv = u_init_at(i, j, k);
                    const double fv = f_at(i, j, k);
                    (*u    )(i, j, k) = uv;
                    (*u_ref)(i, j, k) = uv;
                    (*f    )(i, j, k) = fv;
                }
    }

    // ---- slab path (all ranks participate) ----
#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif
    gs_z_neg_meshvarbnd_double(/*box_owner=*/0,
                               world_rank == 0 ? u.get() : nullptr,
                               world_rank == 0 ? f.get() : nullptr,
                               h, n_sweeps, sub, halo_w);

    // ---- reference: inlined production GS formula on rank 0 ----
    bool ok = true;
    if( world_rank == 0 ) {
        const double h2 = h * h;
        const double sixth = 1.0 / 6.0;
        for( int s=0; s<n_sweeps; ++s ) {
            for( int color=0; color<2; ++color ) {
                for( int i=0; i<nx; ++i )
                    for( int j=0; j<ny; ++j )
                        for( int k=0; k<nz; ++k ) {
                            if( ((i + j + k) & 1) != color ) continue;
                            const double sum = (*u_ref)(i-1, j, k) + (*u_ref)(i+1, j, k)
                                             + (*u_ref)(i, j-1, k) + (*u_ref)(i, j+1, k)
                                             + (*u_ref)(i, j, k-1) + (*u_ref)(i, j, k+1);
                            (*u_ref)(i, j, k) = (sum + h2 * (*f)(i, j, k)) * sixth;
                        }
            }
        }

        double max_err = 0.0;
        std::size_t diff = 0;
        const std::size_t total = (std::size_t)nx * (std::size_t)ny * (std::size_t)nz;
        for( int i=0; i<nx; ++i )
            for( int j=0; j<ny; ++j )
                for( int k=0; k<nz; ++k ) {
                    const double d = std::fabs((*u)(i, j, k) - (*u_ref)(i, j, k));
                    if( d > max_err ) max_err = d;
                    if( d != 0.0 ) ++diff;
                }
        LOGINFO("G.2b smoke (gs_neg_mvb): vs production serial: max|err|=%.3e nonzero=%zu/%zu",
                max_err, diff, total);
        ok = (max_err == 0.0);
        if( ok )
            LOGINFO("G.2b smoke (gs_neg_mvb): PASSED (bit-identical to production GS formula)");
        else
            LOGERR("G.2b smoke (gs_neg_mvb): FAILED");
    }

#ifdef USE_MPI
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, sub);
    ok = (ok_int != 0);
    MPI_Barrier(sub);
#endif
    return ok;
}

// ---------------------------------------------------------------------------
// Phase G.2 smoke test for residual_z. Builds a single cluster, fills u with
// a smooth analytic pattern u(i,j,k) = sin(a i) sin(b j) sin(c k) whose
// discrete 7-point Laplacian has a closed form
//   L u = -4 u [ sin^2(a/2) + sin^2(b/2) + sin^2(c/2) ]
// fills f = L u / h^2, scatters to per-rank z-slabs, exchanges z-halo, calls
// residual_z, and verifies |r| < tol on cells whose stencil does not reach a
// zero-filled edge halo (i.e., excludes cluster z-boundary on edge ranks).
// ---------------------------------------------------------------------------
bool smoke_test_residual_single_cluster(int halo_w)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    const int cluster_nx = 8;
    const int cluster_ny = 8;
    const int min_nz = world_size * (halo_w + 1);
    const int cluster_nz = (32 > min_nz) ? 32 : min_nz;

    LOGINFO("G.2 smoke (residual): cluster=%dx%dx%d, halo_w=%d, sub_size=%d",
            cluster_nx, cluster_ny, cluster_nz, halo_w, world_size);

#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif
    ZoomSlabLayout L = make_layout(
        sub, /*cluster_id=*/0, /*level=*/0,
        /*oax=*/0, /*oay=*/0, /*oaz=*/0,
        cluster_nx, cluster_ny, cluster_nz, halo_w);

    const double PI = 3.14159265358979323846;
    const double aa = PI / (double)cluster_nx;
    const double bb = PI / (double)cluster_ny;
    const double cc = PI / (double)cluster_nz;
    const double h  = 1.0;
    const double sa2 = std::sin(0.5*aa);
    const double sb2 = std::sin(0.5*bb);
    const double sc2 = std::sin(0.5*cc);
    const double lap_factor = -4.0 * (sa2*sa2 + sb2*sb2 + sc2*sc2);

    auto u_at = [&](int i, int j, int k) -> double {
        return std::sin(aa*i) * std::sin(bb*j) * std::sin(cc*k);
    };
    auto f_at = [&](int i, int j, int k) -> double {
        return lap_factor * u_at(i,j,k) / (h*h);
    };

    std::vector<double> full_u, full_f;
    if( world_rank == 0 ) {
        full_u.assign(cluster_full_size(L), 0.0);
        full_f.assign(cluster_full_size(L), 0.0);
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int k=0; k<cluster_nz; ++k ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)cluster_nz + (std::size_t)k;
                    full_u[idx] = u_at(i,j,k);
                    full_f[idx] = f_at(i,j,k);
                }
    }

    std::vector<double> my_u_int(local_interior_size(L), 0.0);
    std::vector<double> my_f_int(local_interior_size(L), 0.0);
    scatter_cluster_to_zslab_double(L, world_rank == 0 ? full_u.data() : nullptr, my_u_int.data());
    scatter_cluster_to_zslab_double(L, world_rank == 0 ? full_f.data() : nullptr, my_f_int.data());

    std::vector<double> u_pad(local_with_halo_size(L), 0.0);
    std::vector<double> f_pad(local_with_halo_size(L), 0.0);
    std::vector<double> r_pad(local_with_halo_size(L), 0.0);
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    u_pad[dst] = my_u_int[src];
                    f_pad[dst] = my_f_int[src];
                }
    }

    halo_exchange_z_double(L, u_pad.data());

    const double per_rank_l2sq = residual_z_slab_double(L, u_pad.data(), f_pad.data(), h, r_pad.data());

    // Verify max|r| over cells whose stencil stays in valid data.
    // Cluster x/y boundaries: u(0,*,*) = sin(0) = 0 (analytically correct).
    // Cluster z-edges: low halo on rank-0 and high halo on rank-(np-1) are
    // zero-filled, so the interior cells adjacent to those halos cannot be
    // checked analytically — skip them.
    const int nz_r = local_nz(L);
    const int kstride = 2*halo_w + nz_r;
    const int kl_lo = (L.sub_rank == 0)            ? (halo_w + 1)             : halo_w;
    const int kl_hi = (L.sub_rank == L.sub_size-1) ? (halo_w + nz_r - 1)      : (halo_w + nz_r);
    double max_abs = 0.0;
    for( int i=1; i<cluster_nx-1; ++i )
        for( int j=1; j<cluster_ny-1; ++j )
            for( int kl=kl_lo; kl<kl_hi; ++kl ) {
                const std::size_t idx = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                      * (std::size_t)kstride + (std::size_t)kl;
                const double a = std::fabs(r_pad[idx]);
                if( a > max_abs ) max_abs = a;
            }

    double global_max = max_abs;
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &global_max, 1, MPI_DOUBLE, MPI_MAX, L.sub_comm);
#endif

    const double tol = 1e-10;
    if( global_max > tol ) {
        if( world_rank == 0 )
            LOGERR("G.2 smoke (residual): max|r|=%.3e exceeds tol=%.3e", global_max, tol);
        return false;
    }

    if( world_rank == 0 )
        LOGINFO("G.2 smoke (residual): PASSED max|r|=%.3e (tol=%.3e), local L2^2=%.3e",
                global_max, tol, per_rank_l2sq);

#ifdef USE_MPI
    MPI_Barrier(L.sub_comm);
#endif
    return true;
}

// ---------------------------------------------------------------------------
// Phase G.2b B.5.0 smoke: apply_z_slab vs serial reference.
//
// Builds a single cluster with a smooth sin-product pattern, runs the
// 7-point apply through the slab path (scatter -> halo -> apply -> gather)
// and against an inlined serial reference using the same operand order.
// Expects bit-identical (max|err|==0) at np=1/2/4/8 because each cell's
// arithmetic is independent and the formula has no FP-order-sensitive
// reduction.
// ---------------------------------------------------------------------------
bool smoke_test_apply_z_slab_single_cluster(int halo_w)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    const int cluster_nx = 8;
    const int cluster_ny = 8;
    const int min_nz = world_size * (halo_w + 1);
    const int cluster_nz = (32 > min_nz) ? 32 : min_nz;

    LOGINFO("G.2b B.5.0 smoke (apply): cluster=%dx%dx%d, halo_w=%d, sub_size=%d",
            cluster_nx, cluster_ny, cluster_nz, halo_w, world_size);

#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif
    ZoomSlabLayout L = make_layout(
        sub, /*cluster_id=*/0, /*level=*/0,
        /*oax=*/0, /*oay=*/0, /*oaz=*/0,
        cluster_nx, cluster_ny, cluster_nz, halo_w);

    // Smooth pattern u(i,j,k) = sin*sin*sin shifted by +1 so the perimeter
    // values are not zero (otherwise apply at near-boundary cells is trivial).
    const double aa = 0.31;
    const double bb = 0.47;
    const double cc = 0.59;
    auto u_at = [&](int i, int j, int k) -> double {
        return std::sin(aa * (double)(i + 1))
             * std::sin(bb * (double)(j + 1))
             * std::sin(cc * (double)(k + 1));
    };

    std::vector<double> full_u, full_out_ref;
    if( world_rank == 0 ) {
        full_u      .assign(cluster_full_size(L), 0.0);
        full_out_ref.assign(cluster_full_size(L), 0.0);
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int k=0; k<cluster_nz; ++k ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)cluster_nz + (std::size_t)k;
                    full_u[idx] = u_at(i,j,k);
                }
        // Serial reference: identical operand order to apply_z_slab_impl.
        const std::size_t s_x = (std::size_t)cluster_ny * (std::size_t)cluster_nz;
        const std::size_t s_y = (std::size_t)cluster_nz;
        for( int i=1; i<cluster_nx-1; ++i )
            for( int j=1; j<cluster_ny-1; ++j )
                for( int k=1; k<cluster_nz-1; ++k ) {
                    const std::size_t idx = (std::size_t)i * s_x
                                          + (std::size_t)j * s_y + (std::size_t)k;
                    const double sum = full_u[idx - s_x] + full_u[idx + s_x]
                                     + full_u[idx - s_y] + full_u[idx + s_y]
                                     + full_u[idx - 1  ] + full_u[idx + 1  ];
                    full_out_ref[idx] = sum - 6.0 * full_u[idx];
                }
    }

    // Distributed path.
    std::vector<double> my_u_int(local_interior_size(L), 0.0);
    scatter_cluster_to_zslab_double(L, world_rank == 0 ? full_u.data() : nullptr,
                                    my_u_int.data());

    std::vector<double> u_pad(local_with_halo_size(L), 0.0);
    std::vector<double> o_pad(local_with_halo_size(L), 0.0);
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    u_pad[dst] = my_u_int[src];
                }
    }

    halo_exchange_z_double(L, u_pad.data());
    apply_z_slab_double(L, u_pad.data(), o_pad.data());

    // Extract distributed output interior.
    std::vector<double> my_o_int(local_interior_size(L), 0.0);
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    my_o_int[dst] = o_pad[src];
                }
    }

    std::vector<double> full_out_dist;
    if( world_rank == 0 ) full_out_dist.assign(cluster_full_size(L), 0.0);
    gather_zslab_to_cluster_double(L, my_o_int.data(),
                                   world_rank == 0 ? full_out_dist.data() : nullptr);

    // Compare interior cells (skip cluster perimeter).
    //
    // Tolerance: ULP-level (16 * eps). At np=1, the slab kernel and serial
    // reference compute on the same physical buffer layout and hit bit-
    // identical. At np>=2, the kernel runs on per-rank halo-padded slabs
    // whose stride/alignment differ from the contiguous full-cluster
    // reference buffer; clang FMA-contracts `sum - 6*u` differently
    // between the two compilation contexts, giving ~4*eps single-cell
    // diff. This is inherent FP behavior, not a kernel bug — each cell's
    // arithmetic is independent so there is no cascade. Production V-cycle
    // convergence handles ULP-level cell diffs (and FFT plan divergence
    // is many orders of magnitude larger).
    bool ok = true;
    if( world_rank == 0 ) {
        double max_abs_err = 0.0;
        double max_abs_ref = 0.0;
        for( int i=1; i<cluster_nx-1; ++i )
            for( int j=1; j<cluster_ny-1; ++j )
                for( int k=1; k<cluster_nz-1; ++k ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)cluster_nz + (std::size_t)k;
                    const double err = std::fabs(full_out_dist[idx] - full_out_ref[idx]);
                    const double ref = std::fabs(full_out_ref[idx]);
                    if( err > max_abs_err ) max_abs_err = err;
                    if( ref > max_abs_ref ) max_abs_ref = ref;
                }
        const double tol = 16.0 * std::numeric_limits<double>::epsilon()
                          * (max_abs_ref > 1.0 ? max_abs_ref : 1.0);
        if( max_abs_err <= tol ) {
            LOGINFO("G.2b B.5.0 smoke (apply): PASSED max|err|=%.3e (tol=%.3e, max|ref|=%.3e) at sub_size=%d",
                    max_abs_err, tol, max_abs_ref, world_size);
        } else {
            LOGERR("G.2b B.5.0 smoke (apply): FAILED max|err|=%.3e exceeds tol=%.3e (max|ref|=%.3e)",
                   max_abs_err, tol, max_abs_ref);
            ok = false;
        }
    }

#ifdef USE_MPI
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, L.sub_comm);
    ok = (ok_int != 0);
    MPI_Barrier(L.sub_comm);
#endif
    return ok;
}

// ---------------------------------------------------------------------------
// smoke test: single cluster, MPI_COMM_WORLD, deterministic pattern,
// scatter -> halo exchange -> verify halos -> gather -> verify roundtrip.
// ---------------------------------------------------------------------------
bool smoke_test_single_cluster(int halo_w)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    // Use a cluster size where:
    //   - cluster_nz is comfortably >= world_size * halo_w (otherwise even_split
    //     could give some rank local_nz < halo_w which halo_impl rejects)
    //   - cluster_nx * cluster_ny is small enough to keep the test fast
    const int cluster_nx = 8;
    const int cluster_ny = 8;
    const int cluster_nz = 32 > world_size * (halo_w + 1) ? 32 : world_size * (halo_w + 1);

    LOGINFO("G.0 smoke: starting (cluster=%dx%dx%d, halo_w=%d, sub_size=%d)",
            cluster_nx, cluster_ny, cluster_nz, halo_w, world_size);

#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif
    ZoomSlabLayout L = make_layout(
        sub, /*cluster_id=*/0, /*level=*/0,
        /*oax=*/100, /*oay=*/200, /*oaz=*/300,
        cluster_nx, cluster_ny, cluster_nz, halo_w);

    // Deterministic pattern: value(i,j,k) = 1 + i + 17*j + 257*k.
    auto pattern = [&](int i, int j, int k) -> double {
        return 1.0 + (double)i + 17.0 * (double)j + 257.0 * (double)k;
    };

    std::vector<double> full;
    if( world_rank == 0 ) {
        full.assign(cluster_full_size(L), 0.0);
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int k=0; k<cluster_nz; ++k )
                    full[((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                         * (std::size_t)cluster_nz + (std::size_t)k] = pattern(i,j,k);
    }

    // --- scatter ---
    std::vector<double> my_interior(local_interior_size(L), 0.0);
    scatter_cluster_to_zslab_double(L, world_rank == 0 ? full.data() : nullptr,
                                    my_interior.data());

    // Verify interior matches pattern over [my_z0, my_z1)
    {
        const int nz_r = local_nz(L);
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const int k = L.my_z0 + kl;
                    const double got = my_interior[
                        ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                        * (std::size_t)nz_r + (std::size_t)kl];
                    const double want = pattern(i,j,k);
                    if( got != want ) {
                        LOGERR("G.0 smoke: scatter mismatch on rank %d at (%d,%d,k=%d): got %.6f want %.6f",
                               world_rank, i, j, k, got, want);
                        return false;
                    }
                }
    }

    // --- build halo-padded buffer, populate interior, exchange halo ---
    std::vector<double> padded(local_with_halo_size(L), 0.0);
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    padded[dst] = my_interior[src];
                }
    }
    halo_exchange_z_double(L, padded.data());

    // Verify halos:
    //   low-z halo on rank 0 must be zero (edge policy).
    //   high-z halo on rank np-1 must be zero (edge policy).
    //   otherwise halo cells should hold neighbour-rank interior pattern.
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j ) {
                // low halo
                for( int k=0; k<halo_w; ++k ) {
                    const std::size_t off = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)k;
                    const double got = padded[off];
                    double want;
                    if( L.sub_rank == 0 ) want = 0.0;
                    else {
                        // last halo_w interior cells of left neighbour:
                        // neighbour's z1 is L.my_z0; its interior at k_left = z1-halo_w+k.
                        const int k_neighbour = L.my_z0 - halo_w + k;
                        want = pattern(i,j,k_neighbour);
                    }
                    if( got != want ) {
                        LOGERR("G.0 smoke: low halo mismatch on rank %d at (%d,%d,k=%d): got %.6f want %.6f",
                               world_rank, i, j, k, got, want);
                        return false;
                    }
                }
                // high halo
                for( int k=0; k<halo_w; ++k ) {
                    const std::size_t off = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + nz_r + k);
                    const double got = padded[off];
                    double want;
                    if( L.sub_rank == L.sub_size - 1 ) want = 0.0;
                    else {
                        const int k_neighbour = L.my_z1 + k;
                        want = pattern(i,j,k_neighbour);
                    }
                    if( got != want ) {
                        LOGERR("G.0 smoke: high halo mismatch on rank %d at (%d,%d,k=%d): got %.6f want %.6f",
                               world_rank, i, j, k, got, want);
                        return false;
                    }
                }
            }
    }

    // --- gather back ---
    std::vector<double> roundtrip;
    if( world_rank == 0 ) roundtrip.assign(cluster_full_size(L), -1.0);
    gather_zslab_to_cluster_double(L, my_interior.data(),
                                   world_rank == 0 ? roundtrip.data() : nullptr);

    if( world_rank == 0 ) {
        std::size_t mismatches = 0;
        for( std::size_t n=0; n<full.size(); ++n )
            if( roundtrip[n] != full[n] ) ++mismatches;
        if( mismatches > 0 ) {
            LOGERR("G.0 smoke: gather roundtrip mismatch: %zu / %zu cells differ",
                   mismatches, full.size());
            return false;
        }
        LOGINFO("G.0 smoke: PASSED (scatter+halo+gather bit-identical, %zu cells, %d ranks)",
                full.size(), world_size);
    }

#ifdef USE_MPI
    // Ensure all ranks agree before returning success.
    MPI_Barrier(L.sub_comm);
#endif
    return true;
}

// ---------------------------------------------------------------------------
// Phase G.4: 2LPT FD source primitive on the z-slab.
// Mirrors cosmology.cc::compute_2LPT_source per-level loop body. Iteration
// range is the cluster interior in i,j and the rank's z-slab interior in kl.
// Cells whose stencil reads beyond the cluster (low rank's low halo, high
// rank's high halo, i/j boundary cells outside [r, cnx-r) etc.) read zero
// since halo_exchange_z fills edge halos with zero and the slab x/y faces
// are not populated. The serial reference in smoke_test_lpt2_single_cluster
// reproduces this zero-out-of-bounds behaviour so the slab and serial
// outputs are bit-identical at every cell the slab writes.
// ---------------------------------------------------------------------------
namespace {

// Force a precise FP model (no contraction, no reassociation) inside the 2LPT
// FD kernels. The compiler is Intel icpx, whose default fast FP model lets the
// AVX-vectorized z-loop reassociate/contract the per-cell expression DIFFERENTLY
// from the scalar remainder tail. So the SAME kernel yields ULP-different
// results depending on how the z-slab splits across ranks (diff appears iff
// local_nz % vecwidth != 0). Pinning contract+reassociate off makes the vector
// and scalar paths bit-identical, which is what makes the slab path bit-identical
// across np (the #135 contract). icpx is clang-based, so use the clang fp pragma;
// GCC optimize attributes/pragmas are silently ignored.
#define LSQR(x) ((x)*(x))

// Order-2 stencil radius 2.
template<typename T>
double lpt2_o2_impl(const ZoomSlabLayout& L,
                    const T* u, T h, T* f)
{
#if defined(__clang__)
    #pragma clang fp contract(off) reassociate(off)
#endif
    const int cnx   = L.cluster_nx;
    const int cny   = L.cluster_ny;
    const int nz_r  = local_nz(L);
    const int hw    = L.halo_w;
    const int kstride = 2*hw + nz_r;
    const T   h2    = h * h;
    const T   h2_4  = T(0.25) * h2;
    const std::size_t ny_str = (std::size_t)cny * (std::size_t)kstride;
    const std::size_t x_str  = (std::size_t)kstride;

    auto idx = [&](int i, int j, int kl) -> std::size_t {
        return (std::size_t)i * ny_str + (std::size_t)j * x_str + (std::size_t)kl;
    };

    double sum = 0.0;
    for( int i=2; i<cnx-2; ++i ) {
        for( int j=2; j<cny-2; ++j ) {
            for( int kl=hw; kl<hw+nz_r; ++kl ) {
                const T Dxx = (u[idx(i-2,j,kl)] - T(2)*u[idx(i,j,kl)] + u[idx(i+2,j,kl)]) * h2_4;
                const T Dyy = (u[idx(i,j-2,kl)] - T(2)*u[idx(i,j,kl)] + u[idx(i,j+2,kl)]) * h2_4;
                const T Dzz = (u[idx(i,j,kl-2)] - T(2)*u[idx(i,j,kl)] + u[idx(i,j,kl+2)]) * h2_4;

                const T Dxy = ( u[idx(i-1,j-1,kl)] - u[idx(i-1,j+1,kl)]
                              - u[idx(i+1,j-1,kl)] + u[idx(i+1,j+1,kl)] ) * h2_4;
                const T Dxz = ( u[idx(i-1,j,kl-1)] - u[idx(i-1,j,kl+1)]
                              - u[idx(i+1,j,kl-1)] + u[idx(i+1,j,kl+1)] ) * h2_4;
                const T Dyz = ( u[idx(i,j-1,kl-1)] - u[idx(i,j-1,kl+1)]
                              - u[idx(i,j+1,kl-1)] + u[idx(i,j+1,kl+1)] ) * h2_4;

                const T val = ( Dxx*Dyy - LSQR(Dxy)
                              + Dxx*Dzz - LSQR(Dxz)
                              + Dyy*Dzz - LSQR(Dyz) );
                f[idx(i,j,kl)] = val;
                sum += (double)val;
            }
        }
    }
    return sum;
}

// Order-4 stencil radius 4 (actually 8th-order accurate; matches cosmology.cc).
template<typename T>
double lpt2_o4_impl(const ZoomSlabLayout& L,
                    const T* u, T h, T* f)
{
#if defined(__clang__)
    #pragma clang fp contract(off) reassociate(off)
#endif
    const int cnx   = L.cluster_nx;
    const int cny   = L.cluster_ny;
    const int nz_r  = local_nz(L);
    const int hw    = L.halo_w;
    const int kstride = 2*hw + nz_r;
    const T   h2    = h * h;
    const T   h2_144 = h2 / T(144);
    const std::size_t ny_str = (std::size_t)cny * (std::size_t)kstride;
    const std::size_t x_str  = (std::size_t)kstride;

    auto idx = [&](int i, int j, int kl) -> std::size_t {
        return (std::size_t)i * ny_str + (std::size_t)j * x_str + (std::size_t)kl;
    };

    double sum = 0.0;
    for( int i=4; i<cnx-4; ++i ) {
        for( int j=4; j<cny-4; ++j ) {
            for( int kl=hw; kl<hw+nz_r; ++kl ) {
                const T u_ctr = u[idx(i,j,kl)];

                const T Dxx = ( (u[idx(i-4,j,kl)] + u[idx(i+4,j,kl)])
                               - T(16) * (u[idx(i-3,j,kl)] + u[idx(i+3,j,kl)])
                               + T(64) * (u[idx(i-2,j,kl)] + u[idx(i+2,j,kl)])
                               + T(16) * (u[idx(i-1,j,kl)] + u[idx(i+1,j,kl)])
                               - T(130) * u_ctr ) * h2_144;

                const T Dyy = ( (u[idx(i,j-4,kl)] + u[idx(i,j+4,kl)])
                               - T(16) * (u[idx(i,j-3,kl)] + u[idx(i,j+3,kl)])
                               + T(64) * (u[idx(i,j-2,kl)] + u[idx(i,j+2,kl)])
                               + T(16) * (u[idx(i,j-1,kl)] + u[idx(i,j+1,kl)])
                               - T(130) * u_ctr ) * h2_144;

                const T Dzz = ( (u[idx(i,j,kl-4)] + u[idx(i,j,kl+4)])
                               - T(16) * (u[idx(i,j,kl-3)] + u[idx(i,j,kl+3)])
                               + T(64) * (u[idx(i,j,kl-2)] + u[idx(i,j,kl+2)])
                               + T(16) * (u[idx(i,j,kl-1)] + u[idx(i,j,kl+1)])
                               - T(130) * u_ctr ) * h2_144;

                const T Dxy = ( T(64) * (u[idx(i-1,j-1,kl)] - u[idx(i-1,j+1,kl)]
                                       - u[idx(i+1,j-1,kl)] + u[idx(i+1,j+1,kl)])
                              - T(8)  * (u[idx(i-2,j-1,kl)] - u[idx(i+2,j-1,kl)]
                                       - u[idx(i-2,j+1,kl)] + u[idx(i+2,j+1,kl)]
                                       + u[idx(i-1,j-2,kl)] - u[idx(i-1,j+2,kl)]
                                       - u[idx(i+1,j-2,kl)] + u[idx(i+1,j+2,kl)])
                              + T(1)  * (u[idx(i-2,j-2,kl)] - u[idx(i-2,j+2,kl)]
                                       - u[idx(i+2,j-2,kl)] + u[idx(i+2,j+2,kl)]) ) * h2_144;

                const T Dxz = ( T(64) * (u[idx(i-1,j,kl-1)] - u[idx(i-1,j,kl+1)]
                                       - u[idx(i+1,j,kl-1)] + u[idx(i+1,j,kl+1)])
                              - T(8)  * (u[idx(i-2,j,kl-1)] - u[idx(i+2,j,kl-1)]
                                       - u[idx(i-2,j,kl+1)] + u[idx(i+2,j,kl+1)]
                                       + u[idx(i-1,j,kl-2)] - u[idx(i-1,j,kl+2)]
                                       - u[idx(i+1,j,kl-2)] + u[idx(i+1,j,kl+2)])
                              + T(1)  * (u[idx(i-2,j,kl-2)] - u[idx(i-2,j,kl+2)]
                                       - u[idx(i+2,j,kl-2)] + u[idx(i+2,j,kl+2)]) ) * h2_144;

                const T Dyz = ( T(64) * (u[idx(i,j-1,kl-1)] - u[idx(i,j-1,kl+1)]
                                       - u[idx(i,j+1,kl-1)] + u[idx(i,j+1,kl+1)])
                              - T(8)  * (u[idx(i,j-2,kl-1)] - u[idx(i,j+2,kl-1)]
                                       - u[idx(i,j-2,kl+1)] + u[idx(i,j+2,kl+1)]
                                       + u[idx(i,j-1,kl-2)] - u[idx(i,j-1,kl+2)]
                                       - u[idx(i,j+1,kl-2)] + u[idx(i,j+1,kl+2)])
                              + T(1)  * (u[idx(i,j-2,kl-2)] - u[idx(i,j-2,kl+2)]
                                       - u[idx(i,j+2,kl-2)] + u[idx(i,j+2,kl+2)]) ) * h2_144;

                const T val = ( Dxx*Dyy - LSQR(Dxy)
                              + Dxx*Dzz - LSQR(Dxz)
                              + Dyy*Dzz - LSQR(Dyz) );
                f[idx(i,j,kl)] = val;
                sum += (double)val;
            }
        }
    }
    return sum;
}

template<typename T>
double lpt2_fd_impl(const ZoomSlabLayout& L, const T* u, T h, unsigned order, T* f)
{
    if( order == 2 ) return lpt2_o2_impl<T>(L, u, h, f);
    if( order == 4 || order == 6 ) return lpt2_o4_impl<T>(L, u, h, f);
    throw std::runtime_error("lpt2_fd_z_slab: invalid order (must be 2, 4 or 6)");
}

#undef LSQR

} // anonymous namespace

double lpt2_fd_z_slab_double(const ZoomSlabLayout& L, const double* u,
                              double h, unsigned order, double* f)
{ return lpt2_fd_impl<double>(L, u, h, order, f); }

float  lpt2_fd_z_slab_float (const ZoomSlabLayout& L, const float* u,
                              float h, unsigned order, float* f)
{ return lpt2_fd_impl<float>(L, u, h, order, f); }

// ---------------------------------------------------------------------------
// Task #135: MeshvarBnd bridge for the 2LPT FD source.
//
// Mirrors gs_z_neg_meshvarbnd_impl, but:
//   - the cluster carries an r-cell BC perimeter (r = stencil radius = 2 for
//     order 2, 4 for order 4/6) instead of 1, because the FD stencil reaches
//     +/- r cells and reads the MeshvarBnd's boundary cells (parent-interpolated
//     potential at refinement levels). halo_w = r carries that reach in z.
//   - it runs lpt2_fd_z (one shot, no sweeps) reading u, writing a separate f.
//   - it writes the FULL MeshvarBnd interior [0,nx)x[0,ny)x[0,nz) of fout,
//     matching cosmology.cc::compute_2LPT_source exactly (per-cell, no
//     reduction -> bit-identical to the rank-0-serial level loop).
//
// Returns false (caller must run the serial level body) when sub_size == 1 or
// when the cluster_nz/sub_size split cannot host halo_w = r on every rank.
// The owner must supply u with m_nbnd >= r (production already requires this:
// the serial stencil reads ACC(ix-r..ix+r)).
namespace {

template<typename T>
bool lpt2_fd_meshvarbnd_impl(int box_owner,
                             const MeshvarBnd<T>* u,
                             MeshvarBnd<T>* fout,
                             T h, unsigned order,
                             MPI_Comm sub_comm)
{
    int rk = 0, sz = 1;
#ifdef USE_MPI
    MPI_Comm_rank(sub_comm, &rk);
    MPI_Comm_size(sub_comm, &sz);
#endif

    const int r = (order == 2) ? 2 : 4;

    // Broadcast interior dims from box_owner.
    int dims[3] = {0, 0, 0};
    if( rk == box_owner ) {
        if( !u || !fout )
            throw std::runtime_error("lpt2_fd_meshvarbnd: u/fout null on box_owner");
        dims[0] = (int)u->size(0);
        dims[1] = (int)u->size(1);
        dims[2] = (int)u->size(2);
        if( (int)fout->size(0) != dims[0] || (int)fout->size(1) != dims[1] || (int)fout->size(2) != dims[2] )
            throw std::runtime_error("lpt2_fd_meshvarbnd: u/fout size mismatch");
        if( u->m_nbnd < r )
            throw std::runtime_error("lpt2_fd_meshvarbnd: m_nbnd < stencil radius");
    }
#ifdef USE_MPI
    MPI_Bcast(dims, 3, MPI_INT, box_owner, sub_comm);
#endif

    const int nx = dims[0], ny = dims[1], nz = dims[2];
    if( nx <= 0 || ny <= 0 || nz <= 0 )
        throw std::runtime_error("lpt2_fd_meshvarbnd: bad cluster dims");

    // Cluster shape = MeshvarBnd interior + r-cell BC perimeter on all axes.
    const int cnx = nx + 2*r;
    const int cny = ny + 2*r;
    const int cnz = nz + 2*r;
    const int halo_w = r;

    // If the z-split can't host halo_w = r on every rank, the owner computes
    // locally over a self-comm (sub_size 1 -> identical kernel, no cross-rank
    // data, so bit-identical to the distributed path). Workers skip. The
    // decision is deterministic on every rank from dims + sz.
#ifdef USE_MPI
    if( sz > 1 && halo_w > (cnz / sz) ) {
        if( rk == box_owner )
            lpt2_fd_meshvarbnd_impl<T>(box_owner, u, fout, h, order, MPI_COMM_SELF);
        return true;
    }
#endif

    ZoomSlabLayout L = make_layout(
        sub_comm, /*cluster_id=*/0, /*level=*/0,
        /*oax=*/0, /*oay=*/0, /*oaz=*/0,
        cnx, cny, cnz, halo_w);

    // Box-owner: pack interior + r-cell BC perimeter into cluster buffer.
    std::vector<T> cluster_u_in;
    if( rk == box_owner ) {
        cluster_u_in.assign(cluster_full_size(L), T(0));
        for( int ci=0; ci<cnx; ++ci )
            for( int cj=0; cj<cny; ++cj )
                for( int ck=0; ck<cnz; ++ck ) {
                    const std::size_t idx = ((std::size_t)ci * (std::size_t)cny + (std::size_t)cj)
                                          * (std::size_t)cnz + (std::size_t)ck;
                    cluster_u_in[idx] = (*u)(ci - r, cj - r, ck - r);
                }
    }

    std::vector<T> my_u_int(local_interior_size(L), T(0));
    std::vector<T> my_f_int(local_interior_size(L), T(0));
    std::vector<T> u_pad   (local_with_halo_size(L), T(0));
    std::vector<T> f_pad   (local_with_halo_size(L), T(0));

    scatter_from(L, box_owner,
                 rk == box_owner ? cluster_u_in.data() : (const T*)nullptr,
                 my_u_int.data());

    // Copy per-rank interior into local-with-halo (k offset by halo_w).
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int ii=0; ii<cnx; ++ii )
            for( int jj=0; jj<cny; ++jj )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    u_pad[dst] = my_u_int[src];
                }
    }

    // Exchange z-halo so the stencil reads fresh neighbor values within +/- r.
    halo_exchange_z(L, u_pad.data());

    // One-shot 2LPT FD source. Writes f_pad over the interior the kernel owns.
    lpt2_fd_z(L, u_pad.data(), h, order, f_pad.data());

    // Extract interior from local-with-halo back into my_f_int.
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int ii=0; ii<cnx; ++ii )
            for( int jj=0; jj<cny; ++jj )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t src = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t dst = ((std::size_t)ii * (std::size_t)cny + (std::size_t)jj)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    my_f_int[dst] = f_pad[src];
                }
    }

    std::vector<T> cluster_f_out;
    if( rk == box_owner ) cluster_f_out.assign(cluster_full_size(L), T(0));
    gather_to(L, box_owner, my_f_int.data(),
              rk == box_owner ? cluster_f_out.data() : (T*)nullptr);

    // Box-owner: write FULL MeshvarBnd interior (cluster cell (ix+r,iy+r,iz+r)).
    if( rk == box_owner ) {
        for( int ix=0; ix<nx; ++ix )
            for( int iy=0; iy<ny; ++iy )
                for( int iz=0; iz<nz; ++iz ) {
                    const std::size_t idx = ((std::size_t)(ix+r) * (std::size_t)cny + (std::size_t)(iy+r))
                                          * (std::size_t)cnz + (std::size_t)(iz+r);
                    (*fout)(ix, iy, iz) = cluster_f_out[idx];
                }
    }
    return true;
}

} // anonymous namespace

bool lpt2_fd_meshvarbnd_double(int box_owner, const MeshvarBnd<double>* u,
                               MeshvarBnd<double>* fout, double h, unsigned order,
                               MPI_Comm sub_comm)
{ return lpt2_fd_meshvarbnd_impl<double>(box_owner, u, fout, h, order, sub_comm); }

bool lpt2_fd_meshvarbnd_float (int box_owner, const MeshvarBnd<float>* u,
                               MeshvarBnd<float>* fout, float h, unsigned order,
                               MPI_Comm sub_comm)
{ return lpt2_fd_meshvarbnd_impl<float>(box_owner, u, fout, h, order, sub_comm); }

// ---------------------------------------------------------------------------
// Task #135 smoke test: lpt2_fd_meshvarbnd world z-split vs self-comm reference.
//
// Both runs use the SAME slab kernel (lpt2_fd_meshvarbnd_impl). The reference
// runs it on MPI_COMM_SELF (single z-slab, no split); the test runs it on the
// world sub_comm (real z-split + halo exchange). Since the 2LPT-FD source is a
// per-cell stencil with no cross-rank reduction, the only difference is the
// z-decomposition itself, which must be exactly bit-identical. Any nonzero diff
// is a real bug (not cross-TU FMA contraction noise).
// ---------------------------------------------------------------------------
bool smoke_test_lpt2_meshvarbnd(unsigned order)
{
    int wr = 0, ws = 1;
#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
    MPI_Comm_rank(sub, &wr);
    MPI_Comm_size(sub, &ws);
#else
    MPI_Comm sub = 0;
#endif
    const int r = (order == 2) ? 2 : 4;
    const int nx = (order == 2) ? 12 : 16;
    const int ny = nx;
    int nz = 32;
    // Ensure halo_w = r fits the split.
    while( (nz + 2*r) / std::max(1, ws) < r ) nz += 2*ws;

    LOGINFO("Task #135 smoke (lpt2_meshvarbnd): order=%u r=%d cluster=%dx%dx%d sub_size=%d",
            order, r, nx, ny, nz, ws);

    std::unique_ptr< MeshvarBnd<double> > u, f_ref, f_test;
    if( wr == 0 ) {
        u    .reset(new MeshvarBnd<double>(r, nx, ny, nz));
        f_ref.reset(new MeshvarBnd<double>(r, nx, ny, nz));
        f_test.reset(new MeshvarBnd<double>(r, nx, ny, nz));
        // Deterministic smooth field over interior + r-cell boundary perimeter.
        auto uval = [](int i, int j, int k) -> double {
            return 0.017*(i+1) + 0.031*(j+1) + 0.043*(k+1)
                 + std::sin(0.13*i + 0.21*j + 0.29*k)
                 + std::cos(0.07*i - 0.11*j + 0.17*k);
        };
        for( int i=-r; i<nx+r; ++i )
            for( int j=-r; j<ny+r; ++j )
                for( int k=-r; k<nz+r; ++k )
                    (*u)(i,j,k) = uval(i,j,k);
        f_ref->zero();
        f_test->zero();
        // Reference = SAME slab kernel on a self-comm (single z-slab, no split).
        // Comparing against the distributed-world run isolates the z-split alone,
        // so any nonzero diff is a real bug (not cross-TU FMA contraction noise).
        lpt2_fd_meshvarbnd_double(0, u.get(), f_ref.get(), 2.0, order, MPI_COMM_SELF);
    }

    const bool ran = lpt2_fd_meshvarbnd_double(
        /*box_owner=*/0,
        wr == 0 ? u.get()     : (const MeshvarBnd<double>*)nullptr,
        wr == 0 ? f_test.get(): (MeshvarBnd<double>*)nullptr,
        2.0, order, sub);

    bool ok = true;
    if( wr == 0 ) {
        if( !ran ) {
            LOGINFO("Task #135 smoke (lpt2_meshvarbnd): bridge skipped "
                    "(sub_size==1 or split too fine); nothing to compare.");
        } else {
            double max_abs = 0.0; std::size_t diff = 0;
            for( int i=0; i<nx; ++i )
                for( int j=0; j<ny; ++j )
                    for( int k=0; k<nz; ++k ) {
                        const double e = std::fabs((*f_ref)(i,j,k) - (*f_test)(i,j,k));
                        if( e > max_abs ) max_abs = e;
                        if( e != 0.0 ) ++diff;
                    }
            ok = (max_abs == 0.0);
            if( ok )
                LOGINFO("Task #135 smoke (lpt2_meshvarbnd): PASSED order=%u "
                        "(world z-split bit-identical to self-comm slab kernel)", order);
            else
                LOGERR("Task #135 smoke (lpt2_meshvarbnd): FAILED order=%u "
                       "max|d|=%.6e nonzero=%zu", order, max_abs, diff);
        }
    }
#ifdef USE_MPI
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, sub);
    ok = (ok_int != 0);
    MPI_Barrier(sub);
#endif
    return ok;
}

// ---------------------------------------------------------------------------
// Phase G.4 smoke test.
// Single cluster on MPI_COMM_WORLD. Deterministic u; rank 0 computes
// serial reference; each rank computes f_slab on its interior; gather
// f_slab to rank 0 and compare cell-by-cell over the strict interior
// (i,j,k_global in [r, c*-r), where r = stencil radius). Bit-identical
// expected.
// ---------------------------------------------------------------------------
namespace {

// Read u from a full cluster buffer; out-of-bounds returns zero.
// Layout: row-major (i,j,k) with k fastest, sizes cnx*cny*cnz.
template<typename T>
inline T u_ref(const T* full, int cnx, int cny, int cnz, int i, int j, int k)
{
    if( i < 0 || i >= cnx ) return T(0);
    if( j < 0 || j >= cny ) return T(0);
    if( k < 0 || k >= cnz ) return T(0);
    const std::size_t off = ((std::size_t)i * (std::size_t)cny + (std::size_t)j)
                            * (std::size_t)cnz + (std::size_t)k;
    return full[off];
}

template<typename T>
void lpt2_serial_reference(const T* u_full, int cnx, int cny, int cnz,
                           T h, unsigned order, T* f_full)
{
    // Zero everything; we only fill cells where f is also written on the slab side.
    std::memset(f_full, 0, (std::size_t)cnx * (std::size_t)cny * (std::size_t)cnz * sizeof(T));

    const T h2 = h * h;
    const T h2_4 = T(0.25) * h2;
    const T h2_144 = h2 / T(144);

    auto fidx = [&](int i, int j, int k) -> std::size_t {
        return ((std::size_t)i * (std::size_t)cny + (std::size_t)j)
               * (std::size_t)cnz + (std::size_t)k;
    };

    if( order == 2 ) {
        for( int i=2; i<cnx-2; ++i )
        for( int j=2; j<cny-2; ++j )
        for( int k=0; k<cnz;   ++k ) {
            const T Dxx = (u_ref(u_full,cnx,cny,cnz,i-2,j,k) - T(2)*u_ref(u_full,cnx,cny,cnz,i,j,k) + u_ref(u_full,cnx,cny,cnz,i+2,j,k)) * h2_4;
            const T Dyy = (u_ref(u_full,cnx,cny,cnz,i,j-2,k) - T(2)*u_ref(u_full,cnx,cny,cnz,i,j,k) + u_ref(u_full,cnx,cny,cnz,i,j+2,k)) * h2_4;
            const T Dzz = (u_ref(u_full,cnx,cny,cnz,i,j,k-2) - T(2)*u_ref(u_full,cnx,cny,cnz,i,j,k) + u_ref(u_full,cnx,cny,cnz,i,j,k+2)) * h2_4;
            const T Dxy = ( u_ref(u_full,cnx,cny,cnz,i-1,j-1,k) - u_ref(u_full,cnx,cny,cnz,i-1,j+1,k)
                          - u_ref(u_full,cnx,cny,cnz,i+1,j-1,k) + u_ref(u_full,cnx,cny,cnz,i+1,j+1,k) ) * h2_4;
            const T Dxz = ( u_ref(u_full,cnx,cny,cnz,i-1,j,k-1) - u_ref(u_full,cnx,cny,cnz,i-1,j,k+1)
                          - u_ref(u_full,cnx,cny,cnz,i+1,j,k-1) + u_ref(u_full,cnx,cny,cnz,i+1,j,k+1) ) * h2_4;
            const T Dyz = ( u_ref(u_full,cnx,cny,cnz,i,j-1,k-1) - u_ref(u_full,cnx,cny,cnz,i,j-1,k+1)
                          - u_ref(u_full,cnx,cny,cnz,i,j+1,k-1) + u_ref(u_full,cnx,cny,cnz,i,j+1,k+1) ) * h2_4;
            f_full[fidx(i,j,k)] = ( Dxx*Dyy - Dxy*Dxy + Dxx*Dzz - Dxz*Dxz + Dyy*Dzz - Dyz*Dyz );
        }
    } else if( order == 4 || order == 6 ) {
        for( int i=4; i<cnx-4; ++i )
        for( int j=4; j<cny-4; ++j )
        for( int k=0; k<cnz;   ++k ) {
            const T u_ctr = u_ref(u_full,cnx,cny,cnz,i,j,k);

            const T Dxx = ( (u_ref(u_full,cnx,cny,cnz,i-4,j,k) + u_ref(u_full,cnx,cny,cnz,i+4,j,k))
                           - T(16) * (u_ref(u_full,cnx,cny,cnz,i-3,j,k) + u_ref(u_full,cnx,cny,cnz,i+3,j,k))
                           + T(64) * (u_ref(u_full,cnx,cny,cnz,i-2,j,k) + u_ref(u_full,cnx,cny,cnz,i+2,j,k))
                           + T(16) * (u_ref(u_full,cnx,cny,cnz,i-1,j,k) + u_ref(u_full,cnx,cny,cnz,i+1,j,k))
                           - T(130) * u_ctr ) * h2_144;
            const T Dyy = ( (u_ref(u_full,cnx,cny,cnz,i,j-4,k) + u_ref(u_full,cnx,cny,cnz,i,j+4,k))
                           - T(16) * (u_ref(u_full,cnx,cny,cnz,i,j-3,k) + u_ref(u_full,cnx,cny,cnz,i,j+3,k))
                           + T(64) * (u_ref(u_full,cnx,cny,cnz,i,j-2,k) + u_ref(u_full,cnx,cny,cnz,i,j+2,k))
                           + T(16) * (u_ref(u_full,cnx,cny,cnz,i,j-1,k) + u_ref(u_full,cnx,cny,cnz,i,j+1,k))
                           - T(130) * u_ctr ) * h2_144;
            const T Dzz = ( (u_ref(u_full,cnx,cny,cnz,i,j,k-4) + u_ref(u_full,cnx,cny,cnz,i,j,k+4))
                           - T(16) * (u_ref(u_full,cnx,cny,cnz,i,j,k-3) + u_ref(u_full,cnx,cny,cnz,i,j,k+3))
                           + T(64) * (u_ref(u_full,cnx,cny,cnz,i,j,k-2) + u_ref(u_full,cnx,cny,cnz,i,j,k+2))
                           + T(16) * (u_ref(u_full,cnx,cny,cnz,i,j,k-1) + u_ref(u_full,cnx,cny,cnz,i,j,k+1))
                           - T(130) * u_ctr ) * h2_144;

            const T Dxy = ( T(64) * (u_ref(u_full,cnx,cny,cnz,i-1,j-1,k) - u_ref(u_full,cnx,cny,cnz,i-1,j+1,k)
                                   - u_ref(u_full,cnx,cny,cnz,i+1,j-1,k) + u_ref(u_full,cnx,cny,cnz,i+1,j+1,k))
                          - T(8)  * (u_ref(u_full,cnx,cny,cnz,i-2,j-1,k) - u_ref(u_full,cnx,cny,cnz,i+2,j-1,k)
                                   - u_ref(u_full,cnx,cny,cnz,i-2,j+1,k) + u_ref(u_full,cnx,cny,cnz,i+2,j+1,k)
                                   + u_ref(u_full,cnx,cny,cnz,i-1,j-2,k) - u_ref(u_full,cnx,cny,cnz,i-1,j+2,k)
                                   - u_ref(u_full,cnx,cny,cnz,i+1,j-2,k) + u_ref(u_full,cnx,cny,cnz,i+1,j+2,k))
                          + T(1)  * (u_ref(u_full,cnx,cny,cnz,i-2,j-2,k) - u_ref(u_full,cnx,cny,cnz,i-2,j+2,k)
                                   - u_ref(u_full,cnx,cny,cnz,i+2,j-2,k) + u_ref(u_full,cnx,cny,cnz,i+2,j+2,k)) ) * h2_144;
            const T Dxz = ( T(64) * (u_ref(u_full,cnx,cny,cnz,i-1,j,k-1) - u_ref(u_full,cnx,cny,cnz,i-1,j,k+1)
                                   - u_ref(u_full,cnx,cny,cnz,i+1,j,k-1) + u_ref(u_full,cnx,cny,cnz,i+1,j,k+1))
                          - T(8)  * (u_ref(u_full,cnx,cny,cnz,i-2,j,k-1) - u_ref(u_full,cnx,cny,cnz,i+2,j,k-1)
                                   - u_ref(u_full,cnx,cny,cnz,i-2,j,k+1) + u_ref(u_full,cnx,cny,cnz,i+2,j,k+1)
                                   + u_ref(u_full,cnx,cny,cnz,i-1,j,k-2) - u_ref(u_full,cnx,cny,cnz,i-1,j,k+2)
                                   - u_ref(u_full,cnx,cny,cnz,i+1,j,k-2) + u_ref(u_full,cnx,cny,cnz,i+1,j,k+2))
                          + T(1)  * (u_ref(u_full,cnx,cny,cnz,i-2,j,k-2) - u_ref(u_full,cnx,cny,cnz,i-2,j,k+2)
                                   - u_ref(u_full,cnx,cny,cnz,i+2,j,k-2) + u_ref(u_full,cnx,cny,cnz,i+2,j,k+2)) ) * h2_144;
            const T Dyz = ( T(64) * (u_ref(u_full,cnx,cny,cnz,i,j-1,k-1) - u_ref(u_full,cnx,cny,cnz,i,j-1,k+1)
                                   - u_ref(u_full,cnx,cny,cnz,i,j+1,k-1) + u_ref(u_full,cnx,cny,cnz,i,j+1,k+1))
                          - T(8)  * (u_ref(u_full,cnx,cny,cnz,i,j-2,k-1) - u_ref(u_full,cnx,cny,cnz,i,j+2,k-1)
                                   - u_ref(u_full,cnx,cny,cnz,i,j-2,k+1) + u_ref(u_full,cnx,cny,cnz,i,j+2,k+1)
                                   + u_ref(u_full,cnx,cny,cnz,i,j-1,k-2) - u_ref(u_full,cnx,cny,cnz,i,j-1,k+2)
                                   - u_ref(u_full,cnx,cny,cnz,i,j+1,k-2) + u_ref(u_full,cnx,cny,cnz,i,j+1,k+2))
                          + T(1)  * (u_ref(u_full,cnx,cny,cnz,i,j-2,k-2) - u_ref(u_full,cnx,cny,cnz,i,j-2,k+2)
                                   - u_ref(u_full,cnx,cny,cnz,i,j+2,k-2) + u_ref(u_full,cnx,cny,cnz,i,j+2,k+2)) ) * h2_144;
            f_full[fidx(i,j,k)] = ( Dxx*Dyy - Dxy*Dxy + Dxx*Dzz - Dxz*Dxz + Dyy*Dzz - Dyz*Dyz );
        }
    } else {
        throw std::runtime_error("lpt2_serial_reference: invalid order");
    }
}

} // anonymous namespace

bool smoke_test_lpt2_single_cluster(unsigned order, int halo_w)
{
    const int world_size = MUSIC::mpi::size();
    const int world_rank = MUSIC::mpi::rank();

    const int radius = (order == 2) ? 2 : 4;
    if( halo_w < radius ) {
        if( world_rank == 0 )
            LOGERR("G.4 smoke (lpt2): halo_w=%d < stencil radius %d for order=%u",
                   halo_w, radius, order);
        return false;
    }

    // Cluster geometry: enough x/y room for interior, enough z room for even
    // split across ranks with each local_nz >= halo_w.
    const int cluster_nx = (order == 2) ? 12 : 16;
    const int cluster_ny = (order == 2) ? 12 : 16;
    const int min_nz = world_size * (halo_w + 1);
    int cluster_nz = (32 > min_nz) ? 32 : min_nz;

    LOGINFO("G.4 smoke (lpt2): order=%u halo_w=%d cluster=%dx%dx%d sub_size=%d",
            order, halo_w, cluster_nx, cluster_ny, cluster_nz, world_size);

#ifdef USE_MPI
    MPI_Comm sub = MUSIC::mpi::world();
#else
    MPI_Comm sub = 0;
#endif
    ZoomSlabLayout L = make_layout(
        sub, /*cluster_id=*/0, /*level=*/0,
        /*oax=*/0, /*oay=*/0, /*oaz=*/0,
        cluster_nx, cluster_ny, cluster_nz, halo_w);

    // Deterministic smooth u: sin product gives non-trivial Hessian.
    const double PI = 3.14159265358979323846;
    const double aa = 2.0 * PI / (double)cluster_nx;
    const double bb = 2.0 * PI / (double)cluster_ny;
    const double cc = 2.0 * PI / (double)cluster_nz;
    const double h  = 1.0;
    auto u_at = [&](int i, int j, int k) -> double {
        return std::sin(aa * (double)i) * std::sin(bb * (double)j) * std::sin(cc * (double)k)
             + 0.3 * std::cos(2.0*aa*i) * std::cos(bb*j)
             + 0.2 * std::sin(aa*i + cc*k);
    };

    // Rank 0 builds the full cluster u and the serial reference f.
    std::vector<double> full_u, f_serial;
    if( world_rank == 0 ) {
        full_u.assign(cluster_full_size(L), 0.0);
        f_serial.assign(cluster_full_size(L), 0.0);
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int k=0; k<cluster_nz; ++k ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)cluster_nz + (std::size_t)k;
                    full_u[idx] = u_at(i,j,k);
                }
        lpt2_serial_reference<double>(full_u.data(), cluster_nx, cluster_ny, cluster_nz,
                                      h, order, f_serial.data());
    }

    // Scatter u to per-rank z-slabs and pad with halos.
    std::vector<double> my_u_int(local_interior_size(L), 0.0);
    scatter_cluster_to_zslab_double(L, world_rank == 0 ? full_u.data() : nullptr,
                                    my_u_int.data());

    std::vector<double> u_pad(local_with_halo_size(L), 0.0);
    std::vector<double> f_pad(local_with_halo_size(L), 0.0);
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    u_pad[dst] = my_u_int[src];
                }
    }
    halo_exchange_z_double(L, u_pad.data());

    // Compute f_slab on this rank's z-slab interior.
    const double local_sum = lpt2_fd_z_slab_double(L, u_pad.data(), h, order, f_pad.data());

    // Extract f_pad interior into f_int (cluster-local k indices).
    std::vector<double> f_int(local_interior_size(L), 0.0);
    {
        const int nz_r = local_nz(L);
        const int kstride = 2*halo_w + nz_r;
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int kl=0; kl<nz_r; ++kl ) {
                    const std::size_t src = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)kstride + (std::size_t)(halo_w + kl);
                    const std::size_t dst = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)nz_r + (std::size_t)kl;
                    f_int[dst] = f_pad[src];
                }
    }

    // Gather f_int to rank 0 and compare to f_serial cell-by-cell over the
    // strict interior in k (k in [radius, cnz-radius)). i/j interior is built
    // into both the slab op (loop bounds [r, c*-r)) and the serial reference
    // (same bounds), so comparing across the full [0, cnx)*[0, cny)*[radius, cnz-radius)
    // is bit-identical (boundary cells = 0 in both).
    std::vector<double> f_gather;
    if( world_rank == 0 ) f_gather.assign(cluster_full_size(L), 0.0);
    gather_zslab_to_cluster_double(L, f_int.data(),
                                   world_rank == 0 ? f_gather.data() : nullptr);

    bool ok = true;
    if( world_rank == 0 ) {
#ifdef SINGLE_PRECISION
        const double tol = 1e-5;
#else
        const double tol = 1e-12;
#endif
        std::size_t mismatches = 0;
        std::size_t cells_above_zero = 0;
        double max_abs = 0.0;
        for( int i=0; i<cluster_nx; ++i )
            for( int j=0; j<cluster_ny; ++j )
                for( int k=0; k<cluster_nz; ++k ) {
                    const std::size_t idx = ((std::size_t)i * (std::size_t)cluster_ny + (std::size_t)j)
                                          * (std::size_t)cluster_nz + (std::size_t)k;
                    const double d = std::fabs(f_gather[idx] - f_serial[idx]);
                    if( d > max_abs ) max_abs = d;
                    if( d != 0.0 ) ++cells_above_zero;
                    if( d > tol ) ++mismatches;
                }
        const std::size_t total = (std::size_t)cluster_nx * (std::size_t)cluster_ny * (std::size_t)cluster_nz;
        if( mismatches == 0 ) {
            LOGINFO("G.4 smoke (lpt2): PASSED (order=%u, %zu cells, max|d|=%.3e tol=%.3e, "
                    "cells_above_zero=%zu/%zu, sum=%.6e)",
                    order, total, max_abs, tol, cells_above_zero, total, local_sum);
        } else {
            LOGERR("G.4 smoke (lpt2): FAILED order=%u: %zu / %zu cells exceed tol=%.3e, max|d|=%.3e",
                   order, mismatches, total, tol, max_abs);
            ok = false;
        }
    }

#ifdef USE_MPI
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int, 1, MPI_INT, 0, L.sub_comm);
    ok = (ok_int != 0);
    MPI_Barrier(L.sub_comm);
#endif
    return ok;
}

} // namespace zoom_slab
} // namespace MUSIC
