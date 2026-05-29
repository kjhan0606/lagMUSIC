#ifndef __MPI_POISSON_HH
#define __MPI_POISSON_HH

// Worker-pump + distributed FFT-Poisson helpers.
//
// fft_poisson_plugin's existing solve()/gradient() are invoked from main.cc
// inside if(MUSIC::mpi::is_root()) blocks (because the surrounding grid_hierarchy
// data lives only on rank 0). To MPI-distribute the FFTs without restructuring
// every is_root() block in main.cc, we use a master/worker dispatch:
//
//   * Workers, between SPMD density-generation phases, call worker_pump().
//   * Rank 0 stays inside the existing is_root() blocks and calls solve()/
//     gradient() as before. Those calls broadcast OP_SOLVE / OP_GRADIENT to
//     the workers in the pump and then run the collective FFT.
//   * At the end of each rank-0 Poisson phase, rank 0 calls broadcast_done()
//     to release the workers from the pump.
//
// Single-rank builds (USE_MPI undefined or size==1) bypass the pump entirely;
// fft_poisson_plugin falls through to its serial code path.

#include "general.hh"
#include "mpi_helper.hh"
#include <cstdio>
#include <cstdlib>

template<typename T> class MeshvarBnd;  // declared in mesh.hh (global scope)
// config_file pulled in transitively via general.hh

namespace MUSIC { namespace poisson {

// Diagnostic trace — see definition in mpi_poisson.cc. Inline here so the
// header callers can use the same gating.
inline bool with_pbox_trace_enabled_()
{
	static int cached = -1;
	if( cached < 0 ){
		const char * e = std::getenv("MUSIC_TRACE_PHASE");
		cached = (e && e[0] && e[0] != '0') ? 1 : 0;
	}
	return cached != 0;
}
inline unsigned long with_pbox_next_id_()
{
	static unsigned long s = 0;
	return ++s;
}

// Op codes broadcast from rank 0 to workers in worker_pump().
enum Op {
	OP_DONE          = 0,
	OP_SOLVE         = 1,
	OP_GRADIENT      = 2,
	OP_SOLVE_SLAB    = 3,  // Phase E.2.2a: solve with per-rank slab in/out registered via set_slab_solve_inout
	OP_GRADIENT_SLAB = 4,  // Phase E.2.3: gradient with per-rank slab in/out (reuses set_slab_solve_inout plumbing)
	OP_LPT2_FFT      = 5,  // Phase F2LPT.1: compute_2LPT_source_FFT SPMD (1 r2c + 6 c2r + composite)
	OP_POISSON_HYBRID= 6   // Phase H.1: poisson_hybrid SPMD on slab-distributed MeshvarBnd (periodic only)
};

// Workers (non-root) call this between SPMD phases to wait for fft_poisson
// requests broadcast from rank 0. Returns when rank 0 broadcasts OP_DONE.
// No-op in single-rank or non-MPI builds.
void worker_pump();

// Rank 0 calls this at the end of a Poisson phase to release workers from
// worker_pump(). No-op in single-rank or non-MPI builds.
void broadcast_done();

// RAII helper used in main.cc to bracket a rank-0 Poisson block:
//   { MUSIC::poisson::phase_scope _p; if(MUSIC::mpi::is_root()){ ...solve... } }
// In MPI runs with size>1, non-root ranks enter worker_pump() in the
// constructor and stay there until the rank-0 destructor broadcasts OP_DONE.
// Single-rank / non-MPI builds: both ctor and dtor are no-ops, so the
// bracketed if(is_root()) block runs unmodified on the sole rank.
struct phase_scope {
	phase_scope();
	~phase_scope();
};

// Rank-0 entry points used by fft_poisson_plugin. Each broadcasts the
// matching OP code + grid dims to workers, then performs the collective FFT
// in-place on root_buf (rank 0's padded real buffer of size gnx*gny*2*(gnz/2+1)).
// Workers (already in worker_pump) participate via the same OP dispatch.
template<typename real_t>
void rank0_dist_solve( real_t* root_buf, size_t gnx, size_t gny, size_t gnz );

template<typename real_t>
void rank0_dist_gradient( int dir, real_t* root_buf, size_t gnx, size_t gny, size_t gnz,
                          bool deconvolve_cic );

// ----- Phase E.2.2a: SPMD slab-direct solve --------------------------------
// Caller (all ranks) supplies their own per-rank slab in/out MeshvarBnd<real_t>*
// via set_slab_solve_inout BEFORE entering phase_scope. Workers in worker_pump
// dispatch OP_SOLVE_SLAB by reading the registered pointers; rank 0 broadcasts
// OP_SOLVE_SLAB inside rank0_dist_solve_slab. Both src and dst slabs must
// already exist on every rank with extent (local_nx, gny, gnz) matching FFTW
// MPI decomposition for (gnx,gny,gnz). (Type ::MeshvarBnd is forward-declared
// at file scope below to avoid namespace lookup pulling MUSIC::poisson::...)
template<typename real_t>
void set_slab_solve_inout( ::MeshvarBnd<real_t>* src, ::MeshvarBnd<real_t>* dst );

template<typename real_t>
void rank0_dist_solve_slab( size_t gnx, size_t gny, size_t gnz );
// ----- end Phase E.2.2a ----------------------------------------------------

// ----- Phase E.2.3: SPMD slab-direct gradient ------------------------------
// Same plumbing as the slab solve (set_slab_solve_inout registers src/dst
// pointers before phase_scope). rank 0 broadcasts OP_GRADIENT_SLAB; both
// ranks 0 and workers run run_dist_fft_op_slab via the worker dispatch.
template<typename real_t>
void rank0_dist_gradient_slab( int dir, size_t gnx, size_t gny, size_t gnz,
                                bool deconvolve_cic );
// ----- end Phase E.2.3 -----------------------------------------------------

// ----- Phase H.1: SPMD slab poisson_hybrid ---------------------------------
// Drop-in slab replacement for poisson_hybrid<MeshvarBnd<real_t>> (the
// rank-0-only finest-grid hybrid Poisson step in src/poisson.cc:1217).
// Caller (all ranks) supplies a per-rank slab MeshvarBnd<real_t>* (in-place
// in/out; poisson_hybrid overwrites its input). PERIODIC ONLY for first ship
// (non-periodic 2x-padded path is H.1.4). Periodic branch fires when the
// caller's hierarchy has levelmin==levelmax at the affected level, matching
// the existing serial poisson_hybrid argument shape.
template<typename real_t>
void set_slab_hybrid_buf( ::MeshvarBnd<real_t>* buf );

template<typename real_t>
void rank0_dist_poisson_hybrid_slab( int idir, int order, bool periodic,
                                     bool deconvolve_cic,
                                     size_t gnx, size_t gny, size_t gnz );
// ----- end Phase H.1 -------------------------------------------------------

// Phase E.1b helper. Brackets a rank-0 Poisson compute block with collective
// per-box gather (before) and scatter (after) so the rank-0 body sees full
// per-box meshes for every hierarchy in `hs`. Workers participate in the
// gather/scatter and then idle inside the phase_scope's worker_pump until
// rank 0 broadcasts OP_DONE. The callable runs ONLY on rank 0.
//
// Usage:
//   MUSIC::poisson::with_pbox_distributed([&]{
//       err = the_poisson_solver->solve(f, u);
//   }, f, u);
//
// Under default_owner_of_box==0 the gather/scatter calls are no-ops, so
// existing single-rank/owner=0 runs stay bit-identical.
template<typename Callable, typename... Hs>
inline void with_pbox_distributed( Callable&& c, Hs&... hs )
{
    const bool trace = with_pbox_trace_enabled_();
    const unsigned long wpd_id = trace ? with_pbox_next_id_() : 0UL;
    if( trace ){
        std::fprintf(stderr, "[trace rk=%d] wpd #%lu enter gather (n_hs=%zu)\n",
                     MUSIC::mpi::rank(), wpd_id, sizeof...(Hs));
        std::fflush(stderr);
    }
    using swallow = int[];
    (void)swallow{ 0, (hs.gather_per_box_to_root(), 0)... };
    if( trace ){
        std::fprintf(stderr, "[trace rk=%d] wpd #%lu gather done; enter phase_scope\n",
                     MUSIC::mpi::rank(), wpd_id);
        std::fflush(stderr);
    }
    {
        phase_scope _ps;
        if( MUSIC::mpi::is_root() ) c();
    }
    if( trace ){
        std::fprintf(stderr, "[trace rk=%d] wpd #%lu phase_scope done; enter scatter\n",
                     MUSIC::mpi::rank(), wpd_id);
        std::fflush(stderr);
    }
    (void)swallow{ 0, (hs.scatter_per_box_from_root(), 0)... };
    if( trace ){
        std::fprintf(stderr, "[trace rk=%d] wpd #%lu scatter done; exit\n",
                     MUSIC::mpi::rank(), wpd_id);
        std::fflush(stderr);
    }
}

// ----- Phase G.2b B.2b.0: SPMD wpd variant --------------------------------
// All ranks invoke c() concurrently. NO gather, NO scatter, NO phase_scope.
// The hierarchies in `hs` are passed for ABI symmetry with with_pbox_distributed
// but their per-box layout is assumed post-scatter (each rank holds only its
// E.1b-owned boxes). The callable is responsible for owner-gating any per-box
// op it performs (see B.2b.2 design memo).
//
// Why no MPI here: workers must remain free to engage in collectives that the
// callable initiates (e.g. gs_z_neg_meshvarbnd on a sub_comm, fetch_parent_halo
// on COMM_WORLD). Entering phase_scope would trap them in worker_pump.
//
// Standalone smoke test: call with an empty lambda from all ranks → returns
// without deadlock; trace output (if enabled) shows enter/exit on each rank.

// B.2b.2.2.1.d: solver compute_error/compute_RMS_resid sums across ranks
// when SPMD MG is active; under plain wpd (rank-0-only callable) the Allreduce
// would deadlock since workers never enter solve. This flag is set by
// with_pbox_distributed_spmd and read by the solver.
inline bool& spmd_mg_active_(){
    static thread_local bool flag = false;
    return flag;
}
inline bool spmd_mg_is_active(){ return spmd_mg_active_(); }

template<typename Callable, typename... Hs>
inline void with_pbox_distributed_spmd( Callable&& c, Hs&... /*hs*/ )
{
    const bool trace = with_pbox_trace_enabled_();
    const unsigned long wpd_id = trace ? with_pbox_next_id_() : 0UL;
    if( trace ){
        std::fprintf(stderr, "[trace rk=%d] wpd_spmd #%lu enter callable\n",
                     MUSIC::mpi::rank(), wpd_id);
        std::fflush(stderr);
    }
    bool& flag = spmd_mg_active_();
    const bool prev = flag;
    flag = true;
    c();
    flag = prev;
    if( trace ){
        std::fprintf(stderr, "[trace rk=%d] wpd_spmd #%lu exit\n",
                     MUSIC::mpi::rank(), wpd_id);
        std::fflush(stderr);
    }
}
// ----- end B.2b.0 ----------------------------------------------------------

// ----- D.6: flag-gated dispatch between wpd and wpd_spmd ------------------
// Read setup.zoom_slab_spmd_multigrid from `cf` and route to either
// with_pbox_distributed_spmd (all ranks enter the callable, no gather/scatter)
// or the classic with_pbox_distributed (gather → rank-0 callable → scatter).
//
// Use this at solve() call sites whose lambda body is SPMD-safe (per-box ops
// only — no normalize_density, no compute_LLA_density, no output writes that
// dereference the union mesh on workers). Sites that touch the union mesh
// must keep classic with_pbox_distributed.
//
// At np==1 wpd_spmd is structurally equivalent to wpd, so this helper is
// bit-identical to wpd whenever the flag is off or np==1.
//
// Pulled out of main.cc:1422 (B.2b.2.2.1.c) so the swap can be applied
// uniformly across the ~3 SPMD-safe solve sites without duplicating the
// flag-read boilerplate.
template<typename Callable, typename... Hs>
inline void with_pbox_distributed_maybe_spmd( const ::config_file& cf,
                                              Callable&& c, Hs&... hs )
{
    const bool use_spmd = cf.getValueSafe<bool>(
        "setup", "zoom_slab_spmd_multigrid", false );
    if( use_spmd ){
        with_pbox_distributed_spmd( std::forward<Callable>(c), hs... );
    } else {
        with_pbox_distributed( std::forward<Callable>(c), hs... );
    }
}
// ----- end D.6 -------------------------------------------------------------

// ----- Phase E.2.2b: production slab-solve wrapper for unigrid FFT path ----
// All-ranks SPMD wrapper around the E.2.2a slab-solve primitive. Bypasses
// fft_poisson_plugin::solve entirely (the rank-0 plugin call is replaced by
// the SPMD slab solve, which gives the same answer bit-identical).
//
// Preconditions:
//   - levelmin==levelmax (k-space FFT poisson is unigrid-only)
//   - f has the density populated on rank 0 (full storage); workers' f may
//     be an empty shell.
//   - u was constructed from f and zero'd on rank 0.
//
// Postconditions on rank 0:
//   - u.get_grid(ilevel) holds the Poisson solution as a full MeshvarBnd
//     with periodic halos applied. f is restored to full storage with the
//     original density. Downstream code (gradient_add / write_dm_potential)
//     sees the same shape as the rank-0-only fft_poisson_plugin::solve path.
//
// Must run OUTSIDE phase_scope (the convert helpers and gather are collective
// and cannot run while workers are parked in worker_pump).
template<typename real_t>
void apply_periodic_bc_unigrid_rank0( ::MeshvarBnd<real_t>& u, int nx, int ny, int nz );

template<typename GH>
inline void slab_solve_unigrid( GH& f, GH& u,
                                 unsigned ilevel,
                                 size_t gnx, size_t gny, size_t gnz )
{
#ifdef USE_MPI
    if( MUSIC::mpi::size() <= 1 ) return; // caller falls through to serial path

    typedef typename GH::real_t real_t;

    f.convert_level_full_to_slab( ilevel, gnx, gny, gnz );
    u.convert_level_full_to_slab( ilevel, gnx, gny, gnz );

    set_slab_solve_inout<real_t>( f.get_grid(ilevel), u.get_grid(ilevel) );
    {
        phase_scope _ps;
        if( MUSIC::mpi::is_root() )
            rank0_dist_solve_slab<real_t>( gnx, gny, gnz );
    }
    set_slab_solve_inout<real_t>( (::MeshvarBnd<real_t>*)NULL,
                                   (::MeshvarBnd<real_t>*)NULL );

    u.convert_level_slab_to_full( ilevel );
    f.convert_level_slab_to_full( ilevel ); // restore f for downstream gradient_add

    if( MUSIC::mpi::is_root() )
        apply_periodic_bc_unigrid_rank0<real_t>(
            *u.get_grid(ilevel), (int)gnx, (int)gny, (int)gnz );
#else
    (void)f; (void)u; (void)ilevel; (void)gnx; (void)gny; (void)gnz;
#endif
}
// ----- end Phase E.2.2b ----------------------------------------------------

// ----- Phase E.2.4: combined solve + per-coord gradient span ---------------
// Like slab_solve_unigrid but does NOT convert u back to full at the end.
// Caller must subsequently call slab_gradient_unigrid_existing_u (which also
// leaves u in slab) for each direction, then convert u back via
// u.convert_level_slab_to_full(ilevel) + apply_periodic_bc_unigrid_rank0.
//
// f is restored to full at the end (downstream baryon path may need it).
// Use this to widen the slab span over solve + all three gradients so rank-0
// no longer holds a full u between the calls.
template<typename GH>
inline void slab_solve_unigrid_keep_u_slab( GH& f, GH& u,
                                             unsigned ilevel,
                                             size_t gnx, size_t gny, size_t gnz )
{
#ifdef USE_MPI
    if( MUSIC::mpi::size() <= 1 ) return;

    typedef typename GH::real_t real_t;

    f.convert_level_full_to_slab( ilevel, gnx, gny, gnz );
    u.convert_level_full_to_slab( ilevel, gnx, gny, gnz );

    set_slab_solve_inout<real_t>( f.get_grid(ilevel), u.get_grid(ilevel) );
    {
        phase_scope _ps;
        if( MUSIC::mpi::is_root() )
            rank0_dist_solve_slab<real_t>( gnx, gny, gnz );
    }
    set_slab_solve_inout<real_t>( (::MeshvarBnd<real_t>*)NULL,
                                   (::MeshvarBnd<real_t>*)NULL );

    f.convert_level_slab_to_full( ilevel ); // restore f for downstream gradient_add
    // u stays in slab; caller converts back after gradient loop.
#else
    (void)f; (void)u; (void)ilevel; (void)gnx; (void)gny; (void)gnz;
#endif
}

// Like slab_gradient_unigrid but assumes u is ALREADY in slab form (caller
// invoked slab_solve_unigrid_keep_u_slab earlier). Du goes full→slab→full
// per call. u remains in slab.
template<typename GH>
inline void slab_gradient_unigrid_existing_u( int dir, GH& u, GH& Du,
                                               unsigned ilevel,
                                               size_t gnx, size_t gny, size_t gnz,
                                               bool deconvolve_cic )
{
#ifdef USE_MPI
    if( MUSIC::mpi::size() <= 1 ) return;

    typedef typename GH::real_t real_t;

    // u already slab; only convert Du.
    Du.convert_level_full_to_slab( ilevel, gnx, gny, gnz );

    set_slab_solve_inout<real_t>( u.get_grid(ilevel), Du.get_grid(ilevel) );
    {
        phase_scope _ps;
        if( MUSIC::mpi::is_root() )
            rank0_dist_gradient_slab<real_t>( dir, gnx, gny, gnz, deconvolve_cic );
    }
    set_slab_solve_inout<real_t>( (::MeshvarBnd<real_t>*)NULL,
                                   (::MeshvarBnd<real_t>*)NULL );

    Du.convert_level_slab_to_full( ilevel );
    // u still slab; do not convert.
#else
    (void)dir; (void)u; (void)Du; (void)ilevel; (void)gnx; (void)gny; (void)gnz;
    (void)deconvolve_cic;
#endif
}

// Convert a slab-form u back to full on rank 0 and re-apply periodic BC.
// Use this after the gradient loop completes.
template<typename GH>
inline void slab_restore_u_full( GH& u, unsigned ilevel,
                                  size_t gnx, size_t gny, size_t gnz )
{
#ifdef USE_MPI
    if( MUSIC::mpi::size() <= 1 ) return;
    typedef typename GH::real_t real_t;
    u.convert_level_slab_to_full( ilevel );
    if( MUSIC::mpi::is_root() )
        apply_periodic_bc_unigrid_rank0<real_t>(
            *u.get_grid(ilevel), (int)gnx, (int)gny, (int)gnz );
#else
    (void)u; (void)ilevel; (void)gnx; (void)gny; (void)gnz;
#endif
}
// ----- end Phase E.2.4 -----------------------------------------------------

// ----- Phase E.2.3: production slab-gradient wrapper for unigrid FFT path --
// All-ranks SPMD wrapper around the E.2.3 slab-gradient primitive. Bypasses
// fft_poisson_plugin::gradient (rank 0 plugin call is replaced by the SPMD
// slab gradient, which is bit-identical to the full-path serial gradient on
// the interior cells).
//
// Preconditions:
//   - levelmin==levelmax (k-space FFT poisson/gradient is unigrid-only)
//   - u (potential) has full storage populated on rank 0; workers' u may be
//     an empty shell.
//   - Du is constructed from u (e.g. data_forIO(u)) so it has matching shape;
//     interior cells will be overwritten with the gradient.
//
// Postconditions on rank 0:
//   - Du.get_grid(ilevel)'s INTERIOR cells hold the displacement field; halos
//     are whatever convert_level_full_to_slab+slab_to_full produces (not BC'd,
//     matching the serial fft_poisson_plugin::gradient which also leaves
//     halos as the initial Du=u copy).
//
// Must run OUTSIDE phase_scope (the convert helpers are collective).
template<typename GH>
inline void slab_gradient_unigrid( int dir, GH& u, GH& Du,
                                    unsigned ilevel,
                                    size_t gnx, size_t gny, size_t gnz,
                                    bool deconvolve_cic )
{
#ifdef USE_MPI
    if( MUSIC::mpi::size() <= 1 ) return; // caller falls through to serial path

    typedef typename GH::real_t real_t;

    u.convert_level_full_to_slab( ilevel, gnx, gny, gnz );
    Du.convert_level_full_to_slab( ilevel, gnx, gny, gnz );

    set_slab_solve_inout<real_t>( u.get_grid(ilevel), Du.get_grid(ilevel) );
    {
        phase_scope _ps;
        if( MUSIC::mpi::is_root() )
            rank0_dist_gradient_slab<real_t>( dir, gnx, gny, gnz, deconvolve_cic );
    }
    set_slab_solve_inout<real_t>( (::MeshvarBnd<real_t>*)NULL,
                                   (::MeshvarBnd<real_t>*)NULL );

    Du.convert_level_slab_to_full( ilevel );
    u.convert_level_slab_to_full( ilevel );
    // No periodic BC on Du: serial fft_poisson_plugin::gradient leaves Du
    // halos as the initial Du=u copy and downstream code only reads interior.
#else
    (void)dir; (void)u; (void)Du; (void)ilevel; (void)gnx; (void)gny; (void)gnz;
    (void)deconvolve_cic;
#endif
}
// ----- end Phase E.2.3 -----------------------------------------------------

// ----- Phase F2LPT.1: distributed compute_2LPT_source_FFT ------------------
// Scatterv/Gatherv-style entry called from within a rank-0 is_root() block.
// Workers must be parked in worker_pump (via phase_scope) so they can pick up
// OP_LPT2_FFT and participate in the FFT. Rank 0 supplies padded buffers:
//   phi_root — input padded buffer (nx * ny * 2*(nz/2+1)) with the potential
//   dst_root — output padded buffer of the same shape; receives the composite
//              (sum of 2x2 minors of the Hessian).
// Workers internally pass NULL for both; only rank 0's buffers are touched.
// Internally allocates 7 per-rank slabs (1 phi + 6 Hessian components) of
// size local_nx * gny * 2*(gnz/2+1). Mirrors rank0_dist_solve plumbing.
template<typename real_t>
void rank0_dist_lpt2( real_t* phi_root, real_t* dst_root,
                      size_t gnx, size_t gny, size_t gnz );
// ----- end Phase F2LPT.1 ---------------------------------------------------

// ----- Phase H.1: production slab_poisson_hybrid wrapper -------------------
// All-ranks SPMD wrapper around the H.1 slab hybrid Poisson primitive.
// In-place on f.get_grid(ilevel). Bypasses serial poisson_hybrid<T>(f, ...)
// which allocates ~8x gnx^3 bytes on rank 0 for the FFT scratch (or ~1x for
// periodic) and runs single-threaded.
//
// PERIODIC ONLY for first ship. Non-periodic 2x-padded branch is H.1.4.
//
// Preconditions:
//   - f.get_grid(ilevel) populated on owner of box 0 at this level (typical
//     for the multibox hybrid sites where the finest level has a single union
//     mesh holding the gravity correction source).
//   - periodic must match what serial poisson_hybrid would take (caller passes
//     levelmin==levelmax test result).
//   - f's union mesh is in full storage; will be flipped to slab and back.
//
// Postconditions:
//   - f.get_grid(ilevel) overwritten in place with the hybrid correction
//     (matches serial poisson_hybrid output bit-by-bit modulo FFTW3-MPI plan
//     divergence, identical to E.2.2b/E.2.3 ULP-level behavior).
//
// Must run OUTSIDE phase_scope (convert helpers are collective).
template<typename GH>
inline void slab_poisson_hybrid( GH& f, int idir, int order, bool periodic,
                                  bool deconvolve_cic, unsigned ilevel,
                                  size_t gnx, size_t gny, size_t gnz )
{
#ifdef USE_MPI
    if( MUSIC::mpi::size() <= 1 ) return; // caller falls through to serial path
    if( !periodic ){
        LOGERR("slab_poisson_hybrid: non-periodic 2x-padded path not implemented yet (H.1.4)");
        throw std::runtime_error("slab_poisson_hybrid: non-periodic not implemented");
    }

    typedef typename GH::real_t real_t;

    f.convert_level_full_to_slab( ilevel, gnx, gny, gnz );

    set_slab_hybrid_buf<real_t>( f.get_grid(ilevel) );
    {
        phase_scope _ps;
        if( MUSIC::mpi::is_root() )
            rank0_dist_poisson_hybrid_slab<real_t>( idir, order, periodic,
                                                     deconvolve_cic,
                                                     gnx, gny, gnz );
    }
    set_slab_hybrid_buf<real_t>( (::MeshvarBnd<real_t>*)NULL );

    f.convert_level_slab_to_full( ilevel );
#else
    (void)f; (void)idir; (void)order; (void)periodic; (void)deconvolve_cic;
    (void)ilevel; (void)gnx; (void)gny; (void)gnz;
#endif
}
// ----- end Phase H.1 -------------------------------------------------------

}} // namespace MUSIC::poisson

#endif // __MPI_POISSON_HH
