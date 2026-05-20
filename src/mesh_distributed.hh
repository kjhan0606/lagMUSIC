#ifndef __MESH_DISTRIBUTED_HH
#define __MESH_DISTRIBUTED_HH

// Slab-distributed factories and halo-exchange helpers for Meshvar/MeshvarBnd.
//
// Design notes:
//   * Decomposition is along the x-axis (slowest stride), matching FFTW3-MPI's
//     default 3D layout. This lets us hand a distributed Meshvar's m_pdata
//     buffer directly to fftw_mpi_plan_dft_r2c_3d without an extra transpose.
//   * The slab boundaries are computed via fftw_mpi_local_size_3d so the
//     Meshvar layout is exactly what FFTW wants for an in-place r2c plan.
//   * In serial builds (USE_MPI undefined) every helper here falls back to
//     the legacy single-rank behaviour: m_local_x_start=0, m_local_nx=global_nx.
//
// Padding: FFTW r2c in-place requires the inner dimension to be padded to
// 2*(global_nz/2+1). The helpers here allocate that padding when the
// caller passes `fftw_inplace_pad=true`.

#include "general.hh"
#include "mesh.hh"
#include "mpi_helper.hh"

#ifdef USE_MPI
#include <fftw3-mpi.h>
#endif

#include <cstring>
#include <stdexcept>

namespace MUSIC { namespace dist {

//! local-slab descriptor in the FFTW MPI layout
struct slab_layout {
	size_t global_nx;
	size_t global_ny;
	size_t global_nz;        //!< logical z-extent (NOT padded)
	size_t local_nx;
	size_t local_x_start;
	size_t alloc_real_count; //!< number of real elements to allocate per buffer
	                         //!< (== local_nx * global_ny * 2*(global_nz/2+1) for in-place r2c,
	                         //!<  or local_nx * global_ny * global_nz for non-FFT buffers).
	size_t padded_nz;        //!< inner dim including FFTW r2c padding (2*(nz/2+1)); equal to nz when fftw_inplace_pad=false
};

//! Query FFTW for the slab decomposition this rank should own.
//! When USE_MPI is undefined, returns a single-rank "decomposition" (rank owns everything).
inline slab_layout compute_slab_layout( size_t gnx, size_t gny, size_t gnz,
                                        bool fftw_inplace_pad )
{
	slab_layout s;
	s.global_nx = gnx;
	s.global_ny = gny;
	s.global_nz = gnz;
	s.padded_nz = fftw_inplace_pad ? 2*(gnz/2+1) : gnz;

#ifdef USE_MPI
	ptrdiff_t local_n0, local_0_start;
	ptrdiff_t alloc_local;
#ifdef SINGLE_PRECISION
	alloc_local = fftwf_mpi_local_size_3d(
#else
	alloc_local = fftw_mpi_local_size_3d(
#endif
		(ptrdiff_t)gnx, (ptrdiff_t)gny, (ptrdiff_t)(gnz/2+1),
		MUSIC::mpi::world(), &local_n0, &local_0_start);

	s.local_nx       = (size_t)local_n0;
	s.local_x_start  = (size_t)local_0_start;
	if( fftw_inplace_pad )
		s.alloc_real_count = (size_t)(2 * alloc_local);   // r2c complex count -> real count
	else
		s.alloc_real_count = s.local_nx * s.global_ny * s.global_nz;
#else
	s.local_nx        = gnx;
	s.local_x_start   = 0;
	s.alloc_real_count = s.local_nx * s.global_ny * s.padded_nz;
#endif
	return s;
}

//! Allocate a Meshvar<real_t> as a slab of a (gnx,gny,gnz) global grid.
//! The caller owns the returned pointer.
template<typename real_t>
inline Meshvar<real_t>* make_slab_meshvar( size_t gnx, size_t gny, size_t gnz,
                                           bool fftw_inplace_pad,
                                           int offx=0, int offy=0, int offz=0 )
{
	slab_layout s = compute_slab_layout(gnx, gny, gnz, fftw_inplace_pad);
	Meshvar<real_t>* m = new Meshvar<real_t>(
		s.local_nx == 0 ? 1 : s.local_nx,
		s.global_ny,
		s.padded_nz,
		offx, offy, offz);
	m->mark_as_distributed(gnx, gny, gnz, s.local_x_start, s.local_nx);
	// FFTW MPI may need MORE reals than the naive local_nx*gny*padded_nz product
	// (it allocates extra workspace for the implicit all-to-all transpose when
	// the grid does not divide evenly across ranks). When that's the case,
	// reallocate to alloc_real_count so the FFT plan can run in-place safely.
	const size_t naive_count = m->m_nx * m->m_ny * m->m_nz;
	if( fftw_inplace_pad && s.alloc_real_count > naive_count ){
		delete[] m->m_pdata;
		m->m_pdata = new real_t[s.alloc_real_count];
		std::memset(m->m_pdata, 0, s.alloc_real_count * sizeof(real_t));
	} else {
		std::memset(m->m_pdata, 0, naive_count * sizeof(real_t));
	}
	return m;
}

//! Allocate a MeshvarBnd<real_t> as a slab of a (gnx,gny,gnz) global grid with
//! nbnd ghost cells on every face (including the +/- x faces between slabs).
template<typename real_t>
inline MeshvarBnd<real_t>* make_slab_meshvarbnd( int nbnd,
                                                 size_t gnx, size_t gny, size_t gnz,
                                                 int offx=0, int offy=0, int offz=0 )
{
	// MeshvarBnd uses no FFTW padding (caller does FFTs via a separate Meshvar).
	slab_layout s = compute_slab_layout(gnx, gny, gnz, /*fftw_inplace_pad=*/false);
	MeshvarBnd<real_t>* m = new MeshvarBnd<real_t>(
		nbnd,
		s.local_nx == 0 ? 1 : s.local_nx,
		s.global_ny,
		s.global_nz,
		offx, offy, offz);
	m->mark_as_distributed(gnx, gny, gnz, s.local_x_start, s.local_nx);
	m->zero();
	return m;
}

#ifdef USE_MPI
//! ring neighbours along the x-axis (periodic)
inline void x_neighbours( int& left, int& right )
{
	int rk = MUSIC::mpi::rank();
	int sz = MUSIC::mpi::size();
	left  = (rk - 1 + sz) % sz;
	right = (rk + 1) % sz;
}
#endif

//! Exchange the nbnd-cell halo ring along the x-axis with the +/- neighbours.
//! Periodic in x by construction. Assumes m is slab-distributed.
//! Y and Z halos are local copies (the slab spans the full y and z extent),
//! so they are handled by the existing (single-rank) ghost logic.
template<typename real_t>
inline void halo_exchange_x( MeshvarBnd<real_t>& m )
{
	if( !m.is_distributed() ) return;  // nothing to do in serial layout

#ifdef USE_MPI
	const int nbnd = m.m_nbnd;
	const size_t ny   = m.m_ny; // padded extent including y ghosts
	const size_t nz   = m.m_nz; // padded extent including z ghosts
	const size_t face = (size_t)nbnd * ny * nz;
	// local-storage x indices [0 .. m_nx) include nbnd ghosts on each side.
	// interior local x range is [nbnd .. m_nx-nbnd).
	const size_t left_interior_off  = (size_t)nbnd                * ny * nz; // first interior row
	const size_t right_interior_off = (m.m_nx - 2*(size_t)nbnd)   * ny * nz; // last nbnd interior rows
	const size_t left_ghost_off     = 0;
	const size_t right_ghost_off    = (m.m_nx - (size_t)nbnd)     * ny * nz;

	int lneigh, rneigh; x_neighbours(lneigh, rneigh);

#ifdef SINGLE_PRECISION
	MPI_Datatype dtype = MPI_FLOAT;
#else
	MPI_Datatype dtype = MPI_DOUBLE;
#endif

	// send right interior, recv left ghost
	MPI_Sendrecv(m.m_pdata + right_interior_off, (int)face, dtype, rneigh, 1001,
	             m.m_pdata + left_ghost_off,     (int)face, dtype, lneigh, 1001,
	             MUSIC::mpi::world(), MPI_STATUS_IGNORE);
	// send left interior, recv right ghost
	MPI_Sendrecv(m.m_pdata + left_interior_off,  (int)face, dtype, lneigh, 1002,
	             m.m_pdata + right_ghost_off,    (int)face, dtype, rneigh, 1002,
	             MUSIC::mpi::world(), MPI_STATUS_IGNORE);
#else
	(void)m;
#endif
}

}} // namespace MUSIC::dist

#endif // __MESH_DISTRIBUTED_HH
