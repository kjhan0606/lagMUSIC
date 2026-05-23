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

template<typename T> class MeshvarBnd;  // declared in mesh.hh (global scope)

namespace MUSIC { namespace poisson {

// Op codes broadcast from rank 0 to workers in worker_pump().
enum Op {
	OP_DONE          = 0,
	OP_SOLVE         = 1,
	OP_GRADIENT      = 2,
	OP_SOLVE_SLAB    = 3,  // Phase E.2.2a: solve with per-rank slab in/out registered via set_slab_solve_inout
	OP_GRADIENT_SLAB = 4   // Phase E.2.3: gradient with per-rank slab in/out (reuses set_slab_solve_inout plumbing)
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
    using swallow = int[];
    (void)swallow{ 0, (hs.gather_per_box_to_root(), 0)... };
    {
        phase_scope _ps;
        if( MUSIC::mpi::is_root() ) c();
    }
    (void)swallow{ 0, (hs.scatter_per_box_from_root(), 0)... };
}

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

}} // namespace MUSIC::poisson

#endif // __MPI_POISSON_HH
