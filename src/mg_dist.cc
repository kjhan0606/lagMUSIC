// Distributed multigrid V-cycle helpers. See mg_dist.hh.
//
// Phase 2 (current): perf-negative gather/scatter for interp_coarse_fine and
// FAS body; distributed GS sweep with halo_exchange_x between colors.
// Phase 3 will add distributed interp_coarse_fine + distributed apply/restrict
// for perf-positive behavior.

#include "mg_dist.hh"
#include "mpi_helper.hh"
#include "mesh.hh"
#include "fd_schemes.hh"

#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace MUSIC { namespace mg {

#ifdef USE_MPI

// ---------------------------------------------------------------------------
// Per-rank persistent V-cycle state. Lives across worker_handle calls within
// one V-cycle: mg_begin allocates, mg_end frees.
// ---------------------------------------------------------------------------
namespace {

template<typename real_t>
struct WorkerState {
	bool active = false;
	int  order  = 0;      // 2/4/6 → nbnd 1/2/3
	int  nbnd   = 0;
	int  ilevel = 0;

	// Global uf dims (without ghosts)
	int  gnx = 0, gny = 0, gnz = 0;

	// Local slab: x ∈ [local_x_start, local_x_start+local_nx).
	// Buffer extents (with ghosts): m_nx = local_nx+2*nbnd, m_ny = gny+2*nbnd, m_nz = gnz+2*nbnd.
	int  local_x_start = 0;
	int  local_nx      = 0;

	// Persistent slabs (own data). Allocated by mg_begin, freed by mg_end.
	MeshvarBnd<real_t>* uf_slab = nullptr;
	MeshvarBnd<real_t>* ff_slab = nullptr;
};

// One state per precision. The build instantiates exactly one (fftw_real).
WorkerState<float>  g_state_f;
WorkerState<double> g_state_d;

template<typename real_t> WorkerState<real_t>& state();
template<> WorkerState<float >& state<float >() { return g_state_f; }
template<> WorkerState<double>& state<double>() { return g_state_d; }

// Which precision is the build using?
inline bool is_double_precision()
{
#ifdef SINGLE_PRECISION
	return false;
#else
	return true;
#endif
}

// Meta layout for mg ops (broadcast in two pieces):
//   First: existing 6-int meta from mpi_poisson::worker_pump (we only use [0]=op here).
//   Second (broadcast inside worker_handle/rank-0 entries): mg_meta[8] =
//     [0] order, [1] ilevel, [2] gnx, [3] gny, [4] gnz, [5] local_x_start (for ref), [6/7] reserved.
//
// For ops that don't need full mg_meta (e.g. GS_SWEEP after BEGIN cached the state),
// we still bcast a small follow-up int to keep collective patterns symmetric.

// Compute the slab decomposition along x for a global grid of size gnx.
// Always produces an EVEN local_nx per rank so that the 8:1 FAS restriction in
// mg_apply_restrict (which iterates cix in [0, local_nx/2)) covers every fine
// cell. Distributes the gnx/2 coarse cells evenly; rank r's local_nx is
// 2 * coarse_per_rank[r].
//   nx_per_rank[r] = local_nx for rank r (always even)
//   x_start[r]     = first global ix on rank r (always even)
inline void compute_x_decomposition( int gnx, int size,
                                     std::vector<int>& nx_per_rank,
                                     std::vector<int>& x_start )
{
	nx_per_rank.assign(size, 0);
	x_start.assign(size, 0);
	const int gnx_c = gnx / 2;     // gnx is even (multigrid invariant)
	const int base_c = gnx_c / size;
	const int rem_c  = gnx_c % size;
	for( int r=0; r<size; ++r ){
		nx_per_rank[r] = 2 * (base_c + (r < rem_c ? 1 : 0));
	}
	x_start[0] = 0;
	for( int r=1; r<size; ++r )
		x_start[r] = x_start[r-1] + nx_per_rank[r-1];
}

// ---------------------------------------------------------------------------
// MPI helpers
// ---------------------------------------------------------------------------

template<typename real_t> MPI_Datatype mpi_dtype();
template<> MPI_Datatype mpi_dtype<float >() { return MPI_FLOAT;  }
template<> MPI_Datatype mpi_dtype<double>() { return MPI_DOUBLE; }

// Non-periodic halo exchange along x. Each rank sends its `nbnd` leftmost and
// rightmost interior slices to its x-neighbours. The leftmost rank does not
// receive into its left ghost (interp_coarse_fine fills it instead), and the
// rightmost rank likewise for its right ghost. We achieve this by using
// MPI_PROC_NULL for the off-end neighbour.
template<typename real_t>
void halo_exchange_x_nonperiodic( MeshvarBnd<real_t>& m, int nbnd )
{
	const int rk = MUSIC::mpi::rank();
	const int sz = MUSIC::mpi::size();
	const int lneigh = (rk == 0)      ? MPI_PROC_NULL : rk-1;
	const int rneigh = (rk == sz-1)   ? MPI_PROC_NULL : rk+1;

	const size_t m_nx = (size_t)m.size(0) + 2*(size_t)nbnd;
	const size_t m_ny = (size_t)m.size(1) + 2*(size_t)nbnd;
	const size_t m_nz = (size_t)m.size(2) + 2*(size_t)nbnd;
	const size_t face = (size_t)nbnd * m_ny * m_nz;
	// First interior slice begins at iix = nbnd; the leftmost `nbnd` interior
	// slices occupy iix in [nbnd, 2*nbnd). The rightmost `nbnd` interior
	// slices occupy iix in [m_nx-2*nbnd, m_nx-nbnd).
	const size_t left_interior_off  = (size_t)nbnd               * m_ny * m_nz;
	const size_t right_interior_off = (m_nx - 2*(size_t)nbnd)    * m_ny * m_nz;
	const size_t left_ghost_off     = 0;
	const size_t right_ghost_off    = (m_nx - (size_t)nbnd)      * m_ny * m_nz;

	MPI_Datatype dtype = mpi_dtype<real_t>();
	MPI_Comm comm = MUSIC::mpi::world();

	// Send right interior → right neighbour; recv from left neighbour into left ghost.
	MPI_Sendrecv(m.get_ptr() + right_interior_off, (int)face, dtype, rneigh, 7001,
	             m.get_ptr() + left_ghost_off,     (int)face, dtype, lneigh, 7001,
	             comm, MPI_STATUS_IGNORE);
	// Send left interior → left neighbour; recv from right neighbour into right ghost.
	MPI_Sendrecv(m.get_ptr() + left_interior_off,  (int)face, dtype, lneigh, 7002,
	             m.get_ptr() + right_ghost_off,    (int)face, dtype, rneigh, 7002,
	             comm, MPI_STATUS_IGNORE);
}

// ---------------------------------------------------------------------------
// Scatter / gather utilities. Operate on MeshvarBnd buffers laid out as
//   m_pdata[ ((iix)*m_ny + iiy)*m_nz + iiz ]
// with iix in [0, m_nx). The x-slabs are contiguous in memory (size m_ny*m_nz per slice).
//
// Scatter pattern (rank 0 -> all): each rank r receives x-slices
//   rank-0 iix range [x_start[r], x_start[r] + local_nx[r] + 2*nbnd)
// (i.e., interior PLUS both ghost regions; adjacent ranks' regions overlap by 2*nbnd).
//
// Gather pattern (all -> rank 0): each rank sends INTERIOR ONLY
//   worker iix range [nbnd, nbnd + local_nx)
// → deposited at rank-0 iix range [nbnd + x_start[r], nbnd + x_start[r] + local_nx[r]).
// ---------------------------------------------------------------------------

template<typename real_t>
void scatter_uf_or_ff( const real_t* root_buf, real_t* local_buf,
                       int gnx, int gny, int gnz, int nbnd,
                       int local_x_start, int local_nx )
{
	const int rk = MUSIC::mpi::rank();
	const int sz = MUSIC::mpi::size();
	const size_t m_ny = (size_t)gny + 2*(size_t)nbnd;
	const size_t m_nz = (size_t)gnz + 2*(size_t)nbnd;
	const size_t slice_count = m_ny * m_nz;

	std::vector<int> nx_per_rank, x_start;
	compute_x_decomposition(gnx, sz, nx_per_rank, x_start);

	std::vector<int> sendcounts(sz, 0), senddispls(sz, 0);
	for( int r=0; r<sz; ++r ){
		sendcounts[r] = (int)((size_t)(nx_per_rank[r] + 2*nbnd) * slice_count);
		senddispls[r] = (int)((size_t)x_start[r] * slice_count);
	}

	const int recvcount = (int)((size_t)(local_nx + 2*nbnd) * slice_count);
	(void)local_x_start; // already encoded in senddispls

	MPI_Scatterv( (rk==0) ? root_buf : NULL, sendcounts.data(), senddispls.data(),
	              mpi_dtype<real_t>(),
	              local_buf, recvcount, mpi_dtype<real_t>(),
	              0, MUSIC::mpi::world() );
}

template<typename real_t>
void gather_uf_interior( real_t* root_buf, const real_t* local_buf,
                         int gnx, int gny, int gnz, int nbnd,
                         int local_x_start, int local_nx )
{
	const int rk = MUSIC::mpi::rank();
	const int sz = MUSIC::mpi::size();
	const size_t m_ny = (size_t)gny + 2*(size_t)nbnd;
	const size_t m_nz = (size_t)gnz + 2*(size_t)nbnd;
	const size_t slice_count = m_ny * m_nz;

	std::vector<int> nx_per_rank, x_start;
	compute_x_decomposition(gnx, sz, nx_per_rank, x_start);

	std::vector<int> recvcounts(sz, 0), recvdispls(sz, 0);
	for( int r=0; r<sz; ++r ){
		recvcounts[r] = (int)((size_t)nx_per_rank[r] * slice_count);
		recvdispls[r] = (int)((size_t)(nbnd + x_start[r]) * slice_count);
	}

	// Each worker sends its interior [nbnd, nbnd+local_nx).
	const real_t* send_ptr = local_buf + (size_t)nbnd * slice_count;
	const int     sendcount = (int)((size_t)local_nx * slice_count);
	(void)local_x_start;

	MPI_Gatherv( send_ptr, sendcount, mpi_dtype<real_t>(),
	             (rk==0) ? root_buf : NULL, recvcounts.data(), recvdispls.data(),
	             mpi_dtype<real_t>(),
	             0, MUSIC::mpi::world() );
}

// ---------------------------------------------------------------------------
// Distributed GS sweep (templated on stencil scheme S).
// One sweep does both colours (red then black), with halo_exchange_x in between.
//
// Coloring uses the GLOBAL x-index gix = local_x_start + lix so that the
// red-black pattern is consistent across slab boundaries.
// ---------------------------------------------------------------------------

template<class S, typename real_t>
void do_gs_sweep_local( WorkerState<real_t>& st )
{
	S scheme;
	const real_t c0 = (real_t)(-1.0/scheme.ccoeff());
	const double h  = 1.0/(double)(1<<st.ilevel);
	const real_t h2 = (real_t)(h*h);

	const int nx = st.local_nx;
	const int ny = st.gny;
	const int nz = st.gnz;

	MeshvarBnd<real_t>& u = *st.uf_slab;
	MeshvarBnd<real_t>& f = *st.ff_slab;

	// halo_exchange is required after BOTH colors:
	//   * after color 0 so color 1 reads fresh color-0 ghosts (within this sweep)
	//   * after color 1 so the next sweep's color 0 reads fresh color-1 ghosts
	for( int color=0; color<2; ++color ){
		#pragma omp parallel for
		for( int lix=0; lix<nx; ++lix ){
			const int gix = st.local_x_start + lix;
			for( int iy=0; iy<ny; ++iy ){
				for( int iz=0; iz<nz; ++iz ){
					if( ((gix + iy + iz) & 1) == color ){
						u(lix, iy, iz) = ( scheme.rhs(u, lix, iy, iz)
						                   + h2 * f(lix, iy, iz) ) * c0;
					}
				}
			}
		}
		halo_exchange_x_nonperiodic(u, st.nbnd);
	}
}

template<typename real_t>
void do_gs_sweep_dispatch( WorkerState<real_t>& st )
{
	switch( st.order ){
		case 2: do_gs_sweep_local<stencil_7P <real_t>, real_t>(st); break;
		case 4: do_gs_sweep_local<stencil_13P<real_t>, real_t>(st); break;
		case 6: do_gs_sweep_local<stencil_19P<real_t>, real_t>(st); break;
		default: throw std::runtime_error("mg_dist: unknown stencil order");
	}
}

// ---------------------------------------------------------------------------
// Worker-side handlers
// ---------------------------------------------------------------------------

template<typename real_t>
void on_begin( WorkerState<real_t>& st )
{
	// Receive mg_meta[6]: [order, ilevel, gnx, gny, gnz, root_nbnd]
	// Use root_nbnd (uf_root's actual ghost width) — NOT order_to_nbnd. With the
	// MUSIC multigrid the ghost width is set by the INTERPOLATION operator
	// (interp_O{3,5,7}_fluxcorr → nbnd 2/3/4), which exceeds the FD stencil's
	// own requirement. Using order_to_nbnd would size the slab wrong relative to
	// uf_root's buffer layout, corrupting the scatter strides.
	int mg_meta[6] = {0,0,0,0,0,0};
	MPI_Bcast(mg_meta, 6, MPI_INT, 0, MUSIC::mpi::world());
	st.order  = mg_meta[0];
	st.ilevel = mg_meta[1];
	st.gnx    = mg_meta[2];
	st.gny    = mg_meta[3];
	st.gnz    = mg_meta[4];
	st.nbnd   = mg_meta[5];

	std::vector<int> nx_per_rank, x_start;
	compute_x_decomposition(st.gnx, MUSIC::mpi::size(), nx_per_rank, x_start);
	const int rk = MUSIC::mpi::rank();
	st.local_x_start = x_start[rk];
	st.local_nx      = nx_per_rank[rk];

	// Allocate slabs sized to (local_nx + 2*nbnd, gny + 2*nbnd, gnz + 2*nbnd)
	// — MeshvarBnd ctor adds the 2*nbnd ghost cells automatically.
	if( st.uf_slab ){ delete st.uf_slab; st.uf_slab = nullptr; }
	if( st.ff_slab ){ delete st.ff_slab; st.ff_slab = nullptr; }
	st.uf_slab = new MeshvarBnd<real_t>(st.nbnd, st.local_nx, st.gny, st.gnz, 0, 0, 0);
	st.ff_slab = new MeshvarBnd<real_t>(st.nbnd, st.local_nx, st.gny, st.gnz, 0, 0, 0);
	st.uf_slab->mark_as_distributed(st.gnx, st.gny, st.gnz, st.local_x_start, st.local_nx);
	st.ff_slab->mark_as_distributed(st.gnx, st.gny, st.gnz, st.local_x_start, st.local_nx);

	// Initial scatter of uf and ff.
	scatter_uf_or_ff<real_t>(NULL, st.uf_slab->get_ptr(),
	                         st.gnx, st.gny, st.gnz, st.nbnd,
	                         st.local_x_start, st.local_nx);
	scatter_uf_or_ff<real_t>(NULL, st.ff_slab->get_ptr(),
	                         st.gnx, st.gny, st.gnz, st.nbnd,
	                         st.local_x_start, st.local_nx);
	st.active = true;
}

template<typename real_t>
void on_scatter_uf( WorkerState<real_t>& st )
{
	scatter_uf_or_ff<real_t>(NULL, st.uf_slab->get_ptr(),
	                         st.gnx, st.gny, st.gnz, st.nbnd,
	                         st.local_x_start, st.local_nx);
}

template<typename real_t>
void on_gather_uf( WorkerState<real_t>& st )
{
	gather_uf_interior<real_t>(NULL, st.uf_slab->get_ptr(),
	                           st.gnx, st.gny, st.gnz, st.nbnd,
	                           st.local_x_start, st.local_nx);
}

template<typename real_t>
void on_gs_sweep( WorkerState<real_t>& st )
{
	if( !st.active ) return;
	do_gs_sweep_dispatch(st);
}

// ---------------------------------------------------------------------------
// Distributed apply+restrict (tLu) — Phase A.2
//
// Each worker computes the coarse-cell tLu values for the coarse x-range it
// owns. A coarse cell (ix_c, iy_c, iz_c) consumes 8 fine cells centered at
// (2*ix_c, 2*iy_c, 2*iz_c) and reads stencil neighbours up to ±nbnd cells —
// which fit entirely within the slab+ghosts as long as local_x_start and
// local_nx are both EVEN.
//
// Output layout: each rank produces a packed buffer of size
//   (local_nx/2) * (gny/2) * (gnz/2) reals (no ghosts), x-major.
// Rank-0 gathers them and writes into tLu_root at offset (oxp, oyp, ozp).
// ---------------------------------------------------------------------------

template<class S, typename real_t>
void do_apply_restrict_local( WorkerState<real_t>& st,
                              real_t* out_packed )
{
	S scheme;
	const double h  = 1.0/(double)(1<<st.ilevel);
	const real_t h2 = (real_t)(h*h);

	MeshvarBnd<real_t>& u = *st.uf_slab;

	const int nx = st.local_nx;
	const int ny = st.gny;
	const int nz = st.gnz;
	const int cnx = nx/2;
	const int cny = ny/2;
	const int cnz = nz/2;

	#pragma omp parallel for
	for( int cix=0; cix<cnx; ++cix ){
		const int iix = 2*cix;
		for( int ciy=0, iiy=0; ciy<cny; ++ciy, iiy+=2 ){
			for( int ciz=0, iiz=0; ciz<cnz; ++ciz, iiz+=2 ){
				const real_t v = (real_t)(0.125 * (
					  scheme.apply(u, iix,   iiy,   iiz  )
					+ scheme.apply(u, iix,   iiy,   iiz+1)
					+ scheme.apply(u, iix,   iiy+1, iiz  )
					+ scheme.apply(u, iix,   iiy+1, iiz+1)
					+ scheme.apply(u, iix+1, iiy,   iiz  )
					+ scheme.apply(u, iix+1, iiy,   iiz+1)
					+ scheme.apply(u, iix+1, iiy+1, iiz  )
					+ scheme.apply(u, iix+1, iiy+1, iiz+1)
				)) / h2;
				out_packed[ ((size_t)cix*cny + ciy)*cnz + ciz ] = v;
			}
		}
	}
}

template<typename real_t>
void do_apply_restrict_dispatch( WorkerState<real_t>& st, real_t* out_packed )
{
	switch( st.order ){
		case 2: do_apply_restrict_local<stencil_7P <real_t>, real_t>(st, out_packed); break;
		case 4: do_apply_restrict_local<stencil_13P<real_t>, real_t>(st, out_packed); break;
		case 6: do_apply_restrict_local<stencil_19P<real_t>, real_t>(st, out_packed); break;
		default: throw std::runtime_error("mg_dist::apply_restrict: unknown stencil order");
	}
}

// Worker side: compute local packed buffer, send via Gatherv to rank-0.
template<typename real_t>
void on_apply_restrict( WorkerState<real_t>& st )
{
	if( !st.active ) return;
	const int cnx = st.local_nx/2;
	const int cny = st.gny/2;
	const int cnz = st.gnz/2;
	std::vector<real_t> packed( (size_t)cnx * cny * cnz );
	do_apply_restrict_dispatch(st, packed.data());

	// Rank-0 handles its own assembly directly in the entry point; workers
	// just send. Counts/displacements are recomputed identically on both
	// sides so a Gatherv is well-defined.
	const int sz = MUSIC::mpi::size();
	std::vector<int> nx_per_rank, x_start;
	compute_x_decomposition(st.gnx, sz, nx_per_rank, x_start);
	std::vector<int> cnts(sz, 0), disps(sz, 0);
	int run = 0;
	for( int r=0; r<sz; ++r ){
		cnts[r]  = (nx_per_rank[r]/2) * cny * cnz;
		disps[r] = run;
		run += cnts[r];
	}
	MPI_Gatherv( packed.data(), (int)packed.size(), mpi_dtype<real_t>(),
	             NULL, cnts.data(), disps.data(), mpi_dtype<real_t>(),
	             0, MUSIC::mpi::world() );
}

template<typename real_t>
void on_end( WorkerState<real_t>& st )
{
	// No gather here: the driver's per-iteration mg_gather_uf already left
	// rank-0 uf in a fresh state. Gathering again would overwrite rank-0
	// changes (e.g. prolong_add when npostsmooth == 0).
	delete st.uf_slab; st.uf_slab = nullptr;
	delete st.ff_slab; st.ff_slab = nullptr;
	st.active = false;
}

} // anonymous namespace
#endif // USE_MPI

// ---------------------------------------------------------------------------
// Public dispatcher + helpers
// ---------------------------------------------------------------------------

bool should_distribute( unsigned ilevel, unsigned levelmax )
{
#ifdef USE_MPI
	return MUSIC::mpi::size() > 1 && ilevel == levelmax;
#else
	(void)ilevel; (void)levelmax;
	return false;
#endif
}

void worker_handle( int op )
{
#ifdef USE_MPI
	if( is_double_precision() ){
		auto& st = state<double>();
		switch( op ){
			case OP_MG_BEGIN:        on_begin        (st); break;
			case OP_MG_GS_SWEEP:     on_gs_sweep     (st); break;
			case OP_MG_SCATTER_UF:   on_scatter_uf   (st); break;
			case OP_MG_GATHER_UF:    on_gather_uf    (st); break;
			case OP_MG_APPLY_R:      on_apply_restrict(st); break;
			case OP_MG_END:          on_end          (st); break;
			default: throw std::runtime_error("mg_dist::worker_handle: unknown op");
		}
	} else {
		auto& st = state<float>();
		switch( op ){
			case OP_MG_BEGIN:        on_begin        (st); break;
			case OP_MG_GS_SWEEP:     on_gs_sweep     (st); break;
			case OP_MG_SCATTER_UF:   on_scatter_uf   (st); break;
			case OP_MG_GATHER_UF:    on_gather_uf    (st); break;
			case OP_MG_APPLY_R:      on_apply_restrict(st); break;
			case OP_MG_END:          on_end          (st); break;
			default: throw std::runtime_error("mg_dist::worker_handle: unknown op");
		}
	}
#else
	(void)op;
#endif
}

// ---------------------------------------------------------------------------
// Rank-0 entry points
// ---------------------------------------------------------------------------

#ifdef USE_MPI
namespace {

inline void bcast_op_meta( int op )
{
	int poisson_meta[6] = { op, 0,0,0,0,0 };
	MPI_Bcast(poisson_meta, 6, MPI_INT, 0, MUSIC::mpi::world());
}

} // anonymous
#endif

template<typename real_t>
void mg_begin( int order, int ilevel,
               const MeshvarBnd<real_t>& uf_root,
               const MeshvarBnd<real_t>& ff_root )
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() < 1 ) return;
	auto& st = state<real_t>();
	st.order = order;
	// Use uf_root's actual ghost width (set by the interpolation operator), not
	// order_to_nbnd(order). See on_begin for the rationale.
	st.nbnd  = uf_root.m_nbnd;
	st.ilevel = ilevel;
	st.gnx = (int)uf_root.size(0);
	st.gny = (int)uf_root.size(1);
	st.gnz = (int)uf_root.size(2);

	std::vector<int> nx_per_rank, x_start;
	compute_x_decomposition(st.gnx, MUSIC::mpi::size(), nx_per_rank, x_start);
	st.local_x_start = x_start[0];
	st.local_nx      = nx_per_rank[0];

	// Rank-0 also keeps its own slab in the WorkerState — the rank-0 slab will
	// receive its share of the scatter. The driver continues to use uf_root /
	// ff_root for the FAS body, then mg_scatter_uf pushes rank-0 changes back.
	if( st.uf_slab ){ delete st.uf_slab; st.uf_slab = nullptr; }
	if( st.ff_slab ){ delete st.ff_slab; st.ff_slab = nullptr; }
	st.uf_slab = new MeshvarBnd<real_t>(st.nbnd, st.local_nx, st.gny, st.gnz, 0, 0, 0);
	st.ff_slab = new MeshvarBnd<real_t>(st.nbnd, st.local_nx, st.gny, st.gnz, 0, 0, 0);
	st.uf_slab->mark_as_distributed(st.gnx, st.gny, st.gnz, st.local_x_start, st.local_nx);
	st.ff_slab->mark_as_distributed(st.gnx, st.gny, st.gnz, st.local_x_start, st.local_nx);

	bcast_op_meta(OP_MG_BEGIN);
	int mg_meta[6] = { order, ilevel, st.gnx, st.gny, st.gnz, st.nbnd };
	MPI_Bcast(mg_meta, 6, MPI_INT, 0, MUSIC::mpi::world());

	scatter_uf_or_ff<real_t>(uf_root.get_ptr(), st.uf_slab->get_ptr(),
	                         st.gnx, st.gny, st.gnz, st.nbnd,
	                         st.local_x_start, st.local_nx);
	scatter_uf_or_ff<real_t>(ff_root.get_ptr(), st.ff_slab->get_ptr(),
	                         st.gnx, st.gny, st.gnz, st.nbnd,
	                         st.local_x_start, st.local_nx);
	st.active = true;
#else
	(void)order; (void)ilevel; (void)uf_root; (void)ff_root;
#endif
}

void mg_gs_sweep()
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() < 1 ) return;
	bcast_op_meta(OP_MG_GS_SWEEP);
	if( is_double_precision() ) do_gs_sweep_dispatch<double>(state<double>());
	else                        do_gs_sweep_dispatch<float >(state<float >());
#endif
}

template<typename real_t>
void mg_scatter_uf( const MeshvarBnd<real_t>& uf_root )
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() < 1 ) return;
	auto& st = state<real_t>();
	bcast_op_meta(OP_MG_SCATTER_UF);
	scatter_uf_or_ff<real_t>(uf_root.get_ptr(), st.uf_slab->get_ptr(),
	                         st.gnx, st.gny, st.gnz, st.nbnd,
	                         st.local_x_start, st.local_nx);
#else
	(void)uf_root;
#endif
}

template<typename real_t>
void mg_gather_uf( MeshvarBnd<real_t>& uf_root )
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() < 1 ) return;
	auto& st = state<real_t>();
	bcast_op_meta(OP_MG_GATHER_UF);
	gather_uf_interior<real_t>(uf_root.get_ptr(), st.uf_slab->get_ptr(),
	                           st.gnx, st.gny, st.gnz, st.nbnd,
	                           st.local_x_start, st.local_nx);
#else
	(void)uf_root;
#endif
}

template<typename real_t>
void mg_apply_restrict( const MeshvarBnd<real_t>& uf_root,
                        MeshvarBnd<real_t>& tLu_root,
                        int oxp, int oyp, int ozp )
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() < 1 ) return;
	auto& st = state<real_t>();

	// Push current uf to all worker slabs first (reuses OP_MG_SCATTER_UF
	// dispatch + on_scatter_uf on workers — symmetric collective).
	mg_scatter_uf<real_t>(uf_root);

	// Now broadcast the apply+gather op. Workers' on_apply_restrict only
	// computes locally and joins the Gatherv — no scatter inside it.
	bcast_op_meta(OP_MG_APPLY_R);

	const int sz  = MUSIC::mpi::size();
	const int cny = st.gny/2;
	const int cnz = st.gnz/2;

	std::vector<int> nx_per_rank, x_start;
	compute_x_decomposition(st.gnx, sz, nx_per_rank, x_start);

	std::vector<int> cnts(sz, 0), disps(sz, 0);
	int total = 0;
	for( int r=0; r<sz; ++r ){
		cnts[r]  = (nx_per_rank[r]/2) * cny * cnz;
		disps[r] = total;
		total   += cnts[r];
	}

	// Rank-0 computes its own local packed buffer.
	std::vector<real_t> local_packed( (size_t)(nx_per_rank[0]/2) * cny * cnz );
	do_apply_restrict_dispatch(st, local_packed.data());

	// Gather all worker buffers into one contiguous staging buffer on rank-0.
	std::vector<real_t> all_packed(total);
	MPI_Gatherv( local_packed.data(), (int)local_packed.size(), mpi_dtype<real_t>(),
	             all_packed.data(), cnts.data(), disps.data(), mpi_dtype<real_t>(),
	             0, MUSIC::mpi::world() );

	// Scatter the gathered values into tLu_root at offset (oxp, oyp, ozp).
	// Rank-r's packed block covers coarse-cell x in
	//   [x_start[r]/2, (x_start[r]+nx_per_rank[r])/2) (global coarse x),
	// which maps into tLu_root at uc-coords
	//   ix = oxp + x_start[r]/2 + cix,  iy = oyp + ciy,  iz = ozp + ciz.
	#pragma omp parallel for
	for( int r=0; r<sz; ++r ){
		const int cnx_r = nx_per_rank[r]/2;
		const int cx0   = x_start[r]/2;
		const real_t* src = all_packed.data() + disps[r];
		for( int cix=0; cix<cnx_r; ++cix ){
			for( int ciy=0; ciy<cny; ++ciy ){
				for( int ciz=0; ciz<cnz; ++ciz ){
					tLu_root(oxp + cx0 + cix, oyp + ciy, ozp + ciz) =
						src[ ((size_t)cix*cny + ciy)*cnz + ciz ];
				}
			}
		}
	}
#else
	(void)uf_root; (void)tLu_root; (void)oxp; (void)oyp; (void)ozp;
#endif
}

template<typename real_t>
void mg_end( MeshvarBnd<real_t>& uf_root )
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() < 1 ) return;
	auto& st = state<real_t>();
	bcast_op_meta(OP_MG_END);
	delete st.uf_slab; st.uf_slab = nullptr;
	delete st.ff_slab; st.ff_slab = nullptr;
	st.active = false;
	(void)uf_root;
#else
	(void)uf_root;
#endif
}

// Explicit instantiations for both precisions (only one is used at runtime,
// but both are usable from the templated solver class).
template void mg_begin<float >(int, int, const MeshvarBnd<float >&, const MeshvarBnd<float >&);
template void mg_begin<double>(int, int, const MeshvarBnd<double>&, const MeshvarBnd<double>&);
template void mg_scatter_uf<float >(const MeshvarBnd<float >&);
template void mg_scatter_uf<double>(const MeshvarBnd<double>&);
template void mg_gather_uf <float >(MeshvarBnd<float >&);
template void mg_gather_uf <double>(MeshvarBnd<double>&);
template void mg_end       <float >(MeshvarBnd<float >&);
template void mg_end       <double>(MeshvarBnd<double>&);
template void mg_apply_restrict<float >(const MeshvarBnd<float >&, MeshvarBnd<float >&, int, int, int);
template void mg_apply_restrict<double>(const MeshvarBnd<double>&, MeshvarBnd<double>&, int, int, int);

}} // namespace MUSIC::mg
