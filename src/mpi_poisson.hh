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

namespace MUSIC { namespace poisson {

// Op codes broadcast from rank 0 to workers in worker_pump().
enum Op {
	OP_DONE     = 0,
	OP_SOLVE    = 1,
	OP_GRADIENT = 2
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

}} // namespace MUSIC::poisson

#endif // __MPI_POISSON_HH
