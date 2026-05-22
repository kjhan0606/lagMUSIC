#ifndef __MG_DIST_HH
#define __MG_DIST_HH

// Distributed multigrid V-cycle helpers (task #11).
//
// Used only at the finest level (ilevel == levelmax) in the FAS V-cycle of
// multigrid::solver<S,I,O,T>::twoGrid(). Coarser levels stay serial on rank 0:
// they are too small for distribution to pay off and the recursive descent is
// irreducibly serial anyway.
//
// Master/worker dispatch model (mirrors mpi_poisson):
//   * Workers sit in MUSIC::poisson::worker_pump() between density phases.
//   * Rank 0 broadcasts an op code in the existing 6-int meta envelope; ops
//     >= OP_MG_BEGIN are forwarded to MUSIC::mg::worker_handle which performs
//     any additional collective communications and returns to the pump.
//   * Worker-side V-cycle state (slab buffers) persists across worker_handle
//     calls so a single V-cycle's sequence of ops doesn't reallocate.
//
// Phase 2 (current): perf-negative gather/scatter for interp_cf, distributed
// GS sweep. Phase 3 will replace gather/scatter with distributed interp_cf.
//
// In single-rank or non-MPI builds every helper is a no-op and the serial
// twoGrid path runs unmodified.

#include "general.hh"
#include "mesh.hh"
#include "mpi_helper.hh"

namespace MUSIC { namespace mg {

// Op codes broadcast in meta[0] from rank 0 to workers (see mpi_poisson::worker_pump).
// Distinct from MUSIC::poisson ops (0=DONE, 1=SOLVE, 2=GRADIENT).
enum Op {
	OP_MG_BEGIN      = 10,  // alloc worker state, scatter uf+ff initial slabs
	OP_MG_GS_SWEEP   = 11,  // one Gauss-Seidel sweep on slab + halo_exchange_x between colors
	OP_MG_SCATTER_UF = 12,  // re-scatter uf from rank 0 to workers (after rank-0 interp_cf)
	OP_MG_GATHER_UF  = 13,  // gather worker uf interior to rank 0
	OP_MG_END        = 14,  // gather + free worker state
	// Phase 3 ops (reserved):
	OP_MG_APPLY_R    = 15,
	OP_MG_BCAST_UC   = 16,
	OP_MG_PROLONG    = 17,
	OP_MG_INTERP_CF  = 18,
};

// Worker entry point. Called from MUSIC::poisson::worker_pump after the 6-int
// meta has been broadcast and op was identified as an mg op. The worker is
// responsible for any further collective communication tied to this op.
void worker_handle( int op );

// True iff the distributed path should run for this V-cycle level.
bool should_distribute( unsigned ilevel, unsigned levelmax );

// ---------------------------------------------------------------------------
// Rank-0 entry points. Each broadcasts the op code + meta to workers, then
// participates in the collective work. Templated on real_t to support either
// the build's single or double precision.
// ---------------------------------------------------------------------------

// Allocate worker state (uf_slab, ff_slab buffers sized to slab+ghosts), do
// the initial scatter of uf and ff from rank 0 to worker slabs. Stencil order
// determines ghost width (1/2/3 for O2/O4/O6).
template<typename real_t>
void mg_begin( int order, int ilevel,
               const MeshvarBnd<real_t>& uf_root,
               const MeshvarBnd<real_t>& ff_root );

// One distributed Gauss-Seidel sweep: 2-color red-black with halo_exchange_x
// between colors. Uses worker-cached uf_slab, ff_slab.
void mg_gs_sweep();

// Re-scatter uf from rank-0 buffer to worker slabs. Called by the driver after
// rank-0 interp_coarse_fine has updated rank-0 uf's ghosts (so workers receive
// fresh y/z ghosts and x ghosts at global boundaries via the overlap scatter).
template<typename real_t>
void mg_scatter_uf( const MeshvarBnd<real_t>& uf_root );

// Gather worker uf interiors back to rank-0 uf buffer.
template<typename real_t>
void mg_gather_uf( MeshvarBnd<real_t>& uf_root );

// Final gather + free worker state.
template<typename real_t>
void mg_end( MeshvarBnd<real_t>& uf_root );

// Phase A.2: distributed Laplace apply + 8:1 coarse restriction.
//
// Computes tLu(ix+oxp, iy+oyp, iz+ozp) = 0.125 * sum_{8 fine corners}(
//     scheme_stencil_O{order}.apply(uf_slab, fine_ix, fine_iy, fine_iz) ) / h^2
// for ix in [0, gnx/2), iy in [0, gny/2), iz in [0, gnz/2). Result is gathered
// to rank-0's tLu_root buffer at offset (oxp, oyp, ozp).
//
// Caller must ensure uf has already been scattered (uf_slab on workers reflects
// the current rank-0 uf). The scheme order is taken from the cached worker
// state (set by mg_begin).
template<typename real_t>
void mg_apply_restrict( const MeshvarBnd<real_t>& uf_root,
                        MeshvarBnd<real_t>& tLu_root,
                        int oxp, int oyp, int ozp );

}} // namespace MUSIC::mg

#endif // __MG_DIST_HH
