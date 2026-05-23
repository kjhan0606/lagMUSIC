/*
 
 mg_solver.hh - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010  Oliver Hahn
 
*/

#ifndef __MG_SOLVER_HH
#define __MG_SOLVER_HH

#include <cmath>
#include <iostream>

#include "mg_operators.hh"
#include "mg_interp.hh"

#include "mesh.hh"
#include "mg_dist.hh"
#include "mg_profile.hh"

#define BEGIN_MULTIGRID_NAMESPACE namespace multigrid {
#define END_MULTIGRID_NAMESPACE }

BEGIN_MULTIGRID_NAMESPACE
	
//! options for multigrid smoothing operation
namespace opt {
	enum smtype { sm_jacobi, sm_gauss_seidel, sm_sor };
}


//! actual implementation of FAS adaptive multigrid solver
template< class S, class I, class O, typename T=double >
class solver
{
public:
	typedef S scheme;
	typedef O mgop;
	typedef I interp;

protected:
	scheme				m_scheme;				//!< finite difference scheme
	mgop				m_gridop;				//!< grid prolongation and restriction operator
	unsigned			m_npresmooth,			//!< number of pre sweeps
						m_npostsmooth;			//!< number of post sweeps
	opt::smtype			m_smoother;				//!< smoothing method to be applied
	unsigned			m_ilevelmin;			//!< index of the top grid level
	
	const static bool	m_bperiodic = true;		//!< flag whether top grid is periodic
	
	std::vector<double> m_residu_ini;			//!< vector of initial residuals for each level
	bool m_is_ini;								//!< bool that is true for first iteration

	GridHierarchy<T>	*m_pu,					//!< pointer to GridHierarchy for solution u
						*m_pf,					//!< pointer to GridHierarchy for right-hand-side
						*m_pfsave;				//!< pointer to saved state of right-hand-side (unused)
	
	const MeshvarBnd<T> *m_pubnd;
	
	//! compute residual for a level
  double compute_error( const MeshvarBnd<T>& u, const MeshvarBnd<T>& unew, int ilevel );
	
	//! compute residuals for entire grid hierarchy
	double compute_error( const GridHierarchy<T>& uh, const GridHierarchy<T>& uhnew, bool verbose );
	
	//! compute residuals for entire grid hierarchy
	double compute_RMS_resid( const GridHierarchy<T>& uh, const GridHierarchy<T>& fh, bool verbose );

protected:
	
	//! Jacobi smoothing 
	void Jacobi( T h, MeshvarBnd<T>* u, const MeshvarBnd<T>* f );
	
	//! Gauss-Seidel smoothing
	void GaussSeidel( T h, MeshvarBnd<T>* u, const MeshvarBnd<T>* f );
	
	//! Successive-Overrelaxation smoothing
	void SOR( T h, MeshvarBnd<T>* u, const MeshvarBnd<T>* f );
	
	//! main two-grid (V-cycle) for multi-grid iterations
	void twoGrid( unsigned ilevel );

	//! Phase D.3.2: multi-box variant of twoGrid. At fine level L with
	//! num_boxes(L)>1 each per-box mesh is pre/post-smoothed independently,
	//! restricts into the still-existing union parent uc at L-1 (which gives
	//! correct apply(uc) at gap cells), recurses through standard twoGrid
	//! on the single-box parent, and receives the correction back from
	//! the union uc into each per-box uf. Requires num_boxes(L-1)==1 in
	//! this phase (D.3.2). General multi-box parent is D.3.3.
	void twoGrid_multibox( unsigned ilevel );
	
	//! apply boundary conditions
	void setBC( unsigned ilevel );
	
	//! make top grid periodic boundary conditions
	void make_periodic( MeshvarBnd<T> *u );
	
	//void interp_coarse_fine_cubic( unsigned ilevel, MeshvarBnd<T>& coarse, MeshvarBnd<T>& fine );
		
public:
	
	//! constructor
	solver( GridHierarchy<T>& f, opt::smtype smoother, unsigned npresmooth, unsigned npostsmooth );
	
	//! destructor
	~solver()
	{  }
	
	//! solve Poisson's equation 
	double solve( GridHierarchy<T>& u, double accuracy, double h=-1.0, bool verbose=false );
	
	//! solve Poisson's equation 
	double solve( GridHierarchy<T>& u, double accuracy, bool verbose=false )
	{
		return this->solve ( u, accuracy, -1.0, verbose );
	}
	
	
	
};


template< class S, class I, class O, typename T >
solver<S,I,O,T>::solver( GridHierarchy<T>& f, opt::smtype smoother, unsigned npresmooth, unsigned npostsmooth )
:	m_scheme(), m_gridop(), m_npresmooth( npresmooth ), m_npostsmooth( npostsmooth ), 
m_smoother( smoother ), m_ilevelmin( f.levelmin() ), m_is_ini( true ), m_pf( &f )
{ 
	m_is_ini = true;
}


template< class S, class I, class O, typename T >
void solver<S,I,O,T>::Jacobi( T h, MeshvarBnd<T> *u, const MeshvarBnd<T>* f )
{
	int
		nx = u->size(0), 
		ny = u->size(1), 
		nz = u->size(2);
	
	double 
		c0 = -1.0/m_scheme.ccoeff(),
		h2 = h*h; 
	
	MeshvarBnd<T> uold(*u);
	
	double alpha = 0.95, ialpha = 1.0-alpha;
	
	#pragma omp parallel for
	for( int ix=0; ix<nx; ++ix )
		for( int iy=0; iy<ny; ++iy )
			for( int iz=0; iz<nz; ++iz )
				(*u)(ix,iy,iz) = ialpha * uold(ix,iy,iz) + alpha * (m_scheme.rhs( uold, ix, iy, iz ) + h2 * (*f)(ix,iy,iz))*c0;
	
}

template< class S, class I, class O, typename T >
void solver<S,I,O,T>::SOR( T h, MeshvarBnd<T> *u, const MeshvarBnd<T>* f )
{
	int
		nx = u->size(0), 
		ny = u->size(1), 
		nz = u->size(2);

	double 
		c0 = -1.0/m_scheme.ccoeff(),
		h2 = h*h; 
		
	MeshvarBnd<T> uold(*u);
	
	double 
		alpha = 1.2, 
	//alpha = 2 / (1 + 4 * atan(1.0) / double(u->size(0)))-1.0, //.. ideal alpha
		ialpha = 1.0-alpha;
	
	#pragma omp parallel for
	for( int ix=0; ix<nx; ++ix )
		for( int iy=0; iy<ny; ++iy )
			for( int iz=0; iz<nz; ++iz )
				if( (ix+iy+iz)%2==0 )
					(*u)(ix,iy,iz) = ialpha * uold(ix,iy,iz) + alpha * (m_scheme.rhs( uold, ix, iy, iz ) + h2 * (*f)(ix,iy,iz))*c0;
	
	
	#pragma omp parallel for
	for( int ix=0; ix<nx; ++ix )
		for( int iy=0; iy<ny; ++iy )
			for( int iz=0; iz<nz; ++iz )
				if( (ix+iy+iz)%2!=0 )
					(*u)(ix,iy,iz) = ialpha * uold(ix,iy,iz) + alpha * (m_scheme.rhs( *u, ix, iy, iz ) + h2 * (*f)(ix,iy,iz))*c0;
	
	
	
}

template< class S, class I, class O, typename T >
void solver<S,I,O,T>::GaussSeidel( T h, MeshvarBnd<T>* u, const MeshvarBnd<T>* f )
{
	int 
		nx = u->size(0), 
		ny = u->size(1), 
		nz = u->size(2);
	
	T
		c0 = -1.0/m_scheme.ccoeff(),
		h2 = h*h; 
	
	for( int color=0; color < 2; ++color )
		#pragma omp parallel for
		for( int ix=0; ix<nx; ++ix )
			for( int iy=0; iy<ny; ++iy )
				for( int iz=0; iz<nz; ++iz )
					if( (ix+iy+iz)%2 == color )
						(*u)(ix,iy,iz) = (m_scheme.rhs( *u, ix, iy, iz ) + h2 * (*f)(ix,iy,iz))*c0;
	
}


template< class S, class I, class O, typename T >
void solver<S,I,O,T>::twoGrid( unsigned ilevel )
{
	// Phase D.3.3 (proper, composite multigrid): at multi-box levels,
	// dispatch to per-cluster V-cycle. compute_error/compute_RMS_resid,
	// gradient operator, and final FAS-restrict loop are now all per-box
	// aware so the union mesh at multi-box levels is unused on the compute
	// path — per-box is the source of truth.
	if( m_pu->num_boxes(ilevel) > 1 ){
		twoGrid_multibox(ilevel);
		return;
	}

	MeshvarBnd<T> *uf, *uc, *ff, *fc;


	double
		h = 1.0/(1<<ilevel),
		c0 = -1.0/m_scheme.ccoeff(),
		h2 = h*h;

	uf = m_pu->get_grid(ilevel);
	ff = m_pf->get_grid(ilevel);
	
	uc = m_pu->get_grid(ilevel-1);
	fc = m_pf->get_grid(ilevel-1);	
	
	
	int
		nx = uf->size(0),
		ny = uf->size(1),
		nz = uf->size(2);

	if( m_bperiodic && ilevel <= m_ilevelmin) {
		double _tp = MUSIC::mg_profile::tic();
		make_periodic( uf );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_PERIODIC, _tp);
	}
	else if(!m_bperiodic)
		setBC( ilevel );

	// MPI-distributed path at the finest level only (Phase 2).
	// Coarser levels stay serial on rank 0; the recursive descent is
	// irreducibly serial in this V-cycle.
	const bool mg_dist_active = MUSIC::mg::should_distribute(ilevel, m_pu->levelmax())
	                            && (m_smoother == opt::sm_gauss_seidel);
	const int  mg_dist_order  = S::order;

	if( mg_dist_active ){
		double _tb = MUSIC::mg_profile::tic();
		MUSIC::mg::mg_begin<T>( mg_dist_order, (int)ilevel, *uf, *ff );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_MG_BEGIN_END, _tb);
	}

	//... do smoothing sweeps with specified solver
	for( unsigned i=0; i<m_npresmooth; ++i ){

		if( ilevel > m_ilevelmin ){
			double _ti = MUSIC::mg_profile::tic();
			interp().interp_coarse_fine(ilevel,*uc,*uf);
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_INTERP_CF, _ti);
		}

		if( mg_dist_active ){
			double _ts = MUSIC::mg_profile::tic();
			MUSIC::mg::mg_scatter_uf<T>( *uf );
			MUSIC::mg::mg_gs_sweep();
			MUSIC::mg::mg_gather_uf<T>( *uf );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_DIST, _ts);
		}
		else if( m_smoother == opt::sm_gauss_seidel ){
			double _ts = MUSIC::mg_profile::tic();
			GaussSeidel( h, uf, ff );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_LOCAL, _ts);
		}

		else if( m_smoother == opt::sm_jacobi ){
			double _ts = MUSIC::mg_profile::tic();
			Jacobi( h, uf, ff);
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_LOCAL, _ts);
		}

		else if( m_smoother == opt::sm_sor ){
			double _ts = MUSIC::mg_profile::tic();
			SOR( h, uf, ff );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_LOCAL, _ts);
		}

		if( m_bperiodic && ilevel <= m_ilevelmin ){
			double _tp = MUSIC::mg_profile::tic();
			make_periodic( uf );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_PERIODIC, _tp);
		}
	}


	{
		double _tr = MUSIC::mg_profile::tic();
		m_gridop.restrict( *uf, *uc );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_RESTRICT_U, _tr);
	}
	
	//... essential!!
	if( m_bperiodic && ilevel <= m_ilevelmin ){
		double _tp = MUSIC::mg_profile::tic();
		make_periodic( uc );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_PERIODIC, _tp);
	}
	else if( ilevel > m_ilevelmin ){
		double _ti = MUSIC::mg_profile::tic();
		interp().interp_coarse_fine(ilevel,*uc,*uf);
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_INTERP_CF, _ti);
	}

	
	//....................................................................
	//... we now use hard-coded restriction+operatore app, see below
	/*meshvar_bnd Lu(*uf,false);
	Lu.zero();

	#pragma omp parallel for
	for( int ix=0; ix<nx; ++ix )
		for( int iy=0; iy<ny; ++iy )
			for( int iz=0; iz<nz; ++iz )
				Lu(ix,iy,iz) = m_scheme.apply( (*uf), ix, iy, iz )/h2;
	
	meshvar_bnd tLu(*uc,false);
	
	
	//... restrict Lu
	m_gridop.restrict( Lu, tLu );
	Lu.deallocate();*/
	//.................................................................... 
	
	int 
		oxp = uf->offset(0),
		oyp = uf->offset(1),
		ozp = uf->offset(2);
	
	meshvar_bnd tLu(*uc,false);
	if( mg_dist_active ){
		// Distributed apply+restrict (Phase A.2). Workers compute their slab's
		// share of tLu (via on_apply_restrict) and rank-0 gathers + writes into
		// tLu at offset (oxp, oyp, ozp). The required pre-scatter of *uf is
		// done inside mg_apply_restrict via mg_scatter_uf.
		double _ta = MUSIC::mg_profile::tic();
		MUSIC::mg::mg_apply_restrict<T>( *uf, tLu, oxp, oyp, ozp );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_APPLY_REST_DIST, _ta);
	}
	else
	{
	double _ta = MUSIC::mg_profile::tic();
	#pragma omp parallel for
	for( int ix=0; ix<nx/2; ++ix )
	{
		int iix=2*ix;
		for( int iy=0,iiy=0; iy<ny/2; ++iy,iiy+=2 )


			for( int iz=0,iiz=0; iz<nz/2; ++iz,iiz+=2 )
				tLu(ix+oxp,iy+oyp,iz+ozp) = 0.125 * (
							 m_scheme.apply( (*uf), iix, iiy, iiz )
							+m_scheme.apply( (*uf), iix, iiy, iiz+1 )
							+m_scheme.apply( (*uf), iix, iiy+1, iiz )
							+m_scheme.apply( (*uf), iix, iiy+1, iiz+1 )
							+m_scheme.apply( (*uf), iix+1, iiy, iiz )
							+m_scheme.apply( (*uf), iix+1, iiy, iiz+1 )
							+m_scheme.apply( (*uf), iix+1, iiy+1, iiz )
							+m_scheme.apply( (*uf), iix+1, iiy+1, iiz+1 )
						)/h2;
	}
	MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_APPLY_REST_LOCAL, _ta);
	}

	//... restrict source term
	{
		double _trf = MUSIC::mg_profile::tic();
		m_gridop.restrict( *ff, *fc );

		int oi, oj, ok;
		oi = ff->offset(0);
		oj = ff->offset(1);
		ok = ff->offset(2);

		#pragma omp parallel for
		for( int ix=oi; ix<oi+(int)ff->size(0)/2; ++ix )
			for( int iy=oj; iy<oj+(int)ff->size(1)/2; ++iy )
				for( int iz=ok; iz<ok+(int)ff->size(2)/2; ++iz )
					(*fc)(ix,iy,iz) += ((tLu( ix, iy, iz ) - (m_scheme.apply( *uc, ix, iy, iz )/(4.0*h2))));
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_RESTRICT_F, _trf);
	}

	tLu.deallocate();
	
	meshvar_bnd ucsave(*uc,true);
						
	//... have we reached the end of the recursion or do we need to go up one level?
	if( ilevel == 1 )
		if( m_bperiodic )
			(*uc)(0,0,0) = 0.0;
		else 
			(*uc)(0,0,0) = (m_scheme.rhs( (*uc), 0, 0, 0 ) + 4.0 * h2 * (*fc)(0,0,0))*c0;
	else
		twoGrid( ilevel-1 );
	
	meshvar_bnd cc(*uc,false);

	{
		double _tc = MUSIC::mg_profile::tic();
		//... compute correction on coarse grid
		#pragma omp parallel for
		for( int ix=0; ix<(int)cc.size(0); ++ix )
			for( int iy=0; iy<(int)cc.size(1); ++iy )
				for( int iz=0; iz<(int)cc.size(2); ++iz )
					cc(ix,iy,iz) = (*uc)(ix,iy,iz) - ucsave(ix,iy,iz);

		ucsave.deallocate();

		if( m_bperiodic && ilevel <= m_ilevelmin )
			make_periodic( &cc );

		m_gridop.prolong_add( cc, *uf );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_PROLONG_CC, _tc);
	}

	//... interpolate and apply coarse-fine boundary conditions on fine level
	if( m_bperiodic && ilevel <= m_ilevelmin ){
		double _tp = MUSIC::mg_profile::tic();
		make_periodic( uf );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_PERIODIC, _tp);
	}
	else if(!m_bperiodic)
		setBC( ilevel );

	//... do smoothing sweeps with specified solver
	for( unsigned i=0; i<m_npostsmooth; ++i ){

		if( ilevel > m_ilevelmin ){
			double _ti = MUSIC::mg_profile::tic();
			interp().interp_coarse_fine(ilevel,*uc,*uf);
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_INTERP_CF, _ti);
		}

		if( mg_dist_active ){
			double _ts = MUSIC::mg_profile::tic();
			MUSIC::mg::mg_scatter_uf<T>( *uf );
			MUSIC::mg::mg_gs_sweep();
			MUSIC::mg::mg_gather_uf<T>( *uf );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_DIST, _ts);
		}
		else if( m_smoother == opt::sm_gauss_seidel ){
			double _ts = MUSIC::mg_profile::tic();
			GaussSeidel( h, uf, ff );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_LOCAL, _ts);
		}

		else if( m_smoother == opt::sm_jacobi ){
			double _ts = MUSIC::mg_profile::tic();
			Jacobi( h, uf, ff);
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_LOCAL, _ts);
		}

		else if( m_smoother == opt::sm_sor ){
			double _ts = MUSIC::mg_profile::tic();
			SOR( h, uf, ff );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_LOCAL, _ts);
		}

		if( m_bperiodic && ilevel <= m_ilevelmin ){
			double _tp = MUSIC::mg_profile::tic();
			make_periodic( uf );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_PERIODIC, _tp);
		}

	}

	if( mg_dist_active ){
		double _tb = MUSIC::mg_profile::tic();
		MUSIC::mg::mg_end<T>( *uf );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_MG_BEGIN_END, _tb);
	}

}

// --------------------------------------------------------------------------
// Phase D.3.2: multi-box V-cycle branch. At fine level `ilevel` each child
// box restricts/prolongs through its OWN parent box at level-1 (looked up
// via parent_box_index). The parent level then dispatches recursively —
// if num_boxes(L-1)>1 it re-enters twoGrid_multibox; if single it falls
// through to the standard twoGrid path.
//
// FAS source update + apply_restrict: when the parent is single-box, the
// union uc/fc carry data at "gap-between-clusters" cells so the apply(uc)
// stencil reads sensible values. When the parent is itself multi-box,
// each child's parent IS one specific per-box parent mesh, and we use the
// parent's own halo (filled by interp_coarse_fine from the grandparent in
// the previous frame of the recursion) as the gap source.
// --------------------------------------------------------------------------
template< class S, class I, class O, typename T >
void solver<S,I,O,T>::twoGrid_multibox( unsigned ilevel )
{
	size_t nb = m_pu->num_boxes(ilevel);
	if( nb <= 1 ){ twoGrid(ilevel); return; }

	const bool parent_multi = (m_pu->num_boxes(ilevel-1) > 1);

	double
		h  = 1.0/(1<<ilevel),
		c0 = -1.0/m_scheme.ccoeff(),
		h2 = h*h;
	(void)c0;

	// Gather child pointers and resolve each child's parent.
	std::vector< MeshvarBnd<T>* > ufs(nb), ffs(nb);
	std::vector< MeshvarBnd<T>* > ucs(nb), fcs(nb);
	std::vector< size_t > pidx(nb);
	for( size_t b=0; b<nb; ++b ){
		ufs[b] = m_pu->get_grid(ilevel, b);
		ffs[b] = m_pf->get_grid(ilevel, b);
		if( parent_multi ){
			pidx[b] = m_pu->parent_box_index(ilevel, b);
			ucs[b]  = m_pu->get_grid(ilevel-1, pidx[b]);
			fcs[b]  = m_pf->get_grid(ilevel-1, pidx[b]);
		} else {
			pidx[b] = 0;
			ucs[b]  = m_pu->get_grid(ilevel-1);
			fcs[b]  = m_pf->get_grid(ilevel-1);
		}
	}

	// --- pre-smoothing sweeps (per box) -------------------------------
	for( unsigned i=0; i<m_npresmooth; ++i ){
		for( size_t b=0; b<nb; ++b ){
			interp().interp_coarse_fine(ilevel, *(ucs[b]), *(ufs[b]));
			if( m_smoother == opt::sm_gauss_seidel )
				GaussSeidel( h, ufs[b], ffs[b] );
			else if( m_smoother == opt::sm_jacobi )
				Jacobi( h, ufs[b], ffs[b] );
			else if( m_smoother == opt::sm_sor )
				SOR( h, ufs[b], ffs[b] );
		}
	}

	// --- restrict each per-box uf into its parent uc ------------------
	for( size_t b=0; b<nb; ++b )
		m_gridop.restrict( *(ufs[b]), *(ucs[b]) );

	// Refresh halos so apply(uc) reads coherent values.
	if( !parent_multi && m_bperiodic && ilevel <= m_ilevelmin )
		make_periodic( ucs[0] );
	for( size_t b=0; b<nb; ++b )
		interp().interp_coarse_fine(ilevel, *(ucs[b]), *(ufs[b]));

	// --- apply Laplacian on fine + restrict (per-box tLu scratch) -----
	std::vector< meshvar_bnd* > tLus(nb, NULL);
	{
		// Per parent mesh, allocate tLu once and zero, then write child
		// contributions. Multiple children may share the same parent (when
		// parent is single-box) so we key by parent pointer.
		std::map< MeshvarBnd<T>*, meshvar_bnd* > tlu_for_parent;
		for( size_t b=0; b<nb; ++b ){
			MeshvarBnd<T> * uc = ucs[b];
			auto it = tlu_for_parent.find(uc);
			if( it == tlu_for_parent.end() ){
				meshvar_bnd * t = new meshvar_bnd(*uc, false);
				t->zero();
				tlu_for_parent[uc] = t;
				tLus[b] = t;
			} else {
				tLus[b] = it->second;
			}
		}

		for( size_t b=0; b<nb; ++b ){
			MeshvarBnd<T> * uf = ufs[b];
			meshvar_bnd & tLu = *(tLus[b]);
			int nx = uf->size(0), ny = uf->size(1), nz = uf->size(2);
			int oxp = uf->offset(0), oyp = uf->offset(1), ozp = uf->offset(2);
			#pragma omp parallel for
			for( int ix=0; ix<nx/2; ++ix ){
				int iix=2*ix;
				for( int iy=0,iiy=0; iy<ny/2; ++iy,iiy+=2 )
					for( int iz=0,iiz=0; iz<nz/2; ++iz,iiz+=2 )
						tLu(ix+oxp,iy+oyp,iz+ozp) = 0.125 * (
							 m_scheme.apply( *uf, iix,   iiy,   iiz   )
							+m_scheme.apply( *uf, iix,   iiy,   iiz+1 )
							+m_scheme.apply( *uf, iix,   iiy+1, iiz   )
							+m_scheme.apply( *uf, iix,   iiy+1, iiz+1 )
							+m_scheme.apply( *uf, iix+1, iiy,   iiz   )
							+m_scheme.apply( *uf, iix+1, iiy,   iiz+1 )
							+m_scheme.apply( *uf, iix+1, iiy+1, iiz   )
							+m_scheme.apply( *uf, iix+1, iiy+1, iiz+1 )
						)/h2;
			}
		}

		// --- restrict per-box source ff into its parent fc ------------
		for( size_t b=0; b<nb; ++b )
			m_gridop.restrict( *(ffs[b]), *(fcs[b]) );

		// --- FAS source update at child cells of fc -------------------
		for( size_t b=0; b<nb; ++b ){
			MeshvarBnd<T> * ff = ffs[b];
			MeshvarBnd<T> * uc = ucs[b];
			MeshvarBnd<T> * fc = fcs[b];
			meshvar_bnd & tLu = *(tLus[b]);
			int oi = ff->offset(0), oj = ff->offset(1), ok = ff->offset(2);
			int sx = (int)ff->size(0)/2, sy = (int)ff->size(1)/2, sz = (int)ff->size(2)/2;
			#pragma omp parallel for
			for( int ix=oi; ix<oi+sx; ++ix )
				for( int iy=oj; iy<oj+sy; ++iy )
					for( int iz=ok; iz<ok+sz; ++iz )
						(*fc)(ix,iy,iz) += ( tLu(ix,iy,iz)
						                    - (m_scheme.apply(*uc, ix,iy,iz)/(4.0*h2)) );
		}

		for( auto & kv : tlu_for_parent ){
			kv.second->deallocate();
			delete kv.second;
		}
	}

	// --- save parent uc state per parent mesh -------------------------
	std::map< MeshvarBnd<T>*, meshvar_bnd* > ucsave_for_parent;
	for( size_t b=0; b<nb; ++b ){
		MeshvarBnd<T> * uc = ucs[b];
		if( ucsave_for_parent.find(uc) == ucsave_for_parent.end() )
			ucsave_for_parent[uc] = new meshvar_bnd(*uc, true);
	}

	// --- recurse on the parent level ----------------------------------
	if( ilevel == 1 ){
		// Only single-box parent reachable at ilevel==1.
		(*(ucs[0]))(0,0,0) = 0.0;
	} else {
		// Standard descent: twoGrid() will itself dispatch to multibox
		// when num_boxes(ilevel-1) > 1.
		twoGrid( ilevel-1 );
	}

	// --- prolong correction back per-box ------------------------------
	std::map< MeshvarBnd<T>*, meshvar_bnd* > cc_for_parent;
	for( size_t b=0; b<nb; ++b ){
		MeshvarBnd<T> * uc = ucs[b];
		if( cc_for_parent.find(uc) != cc_for_parent.end() ) continue;
		meshvar_bnd * cc = new meshvar_bnd(*uc, false);
		meshvar_bnd & ucsave = *(ucsave_for_parent[uc]);
		#pragma omp parallel for
		for( int ix=0; ix<(int)cc->size(0); ++ix )
			for( int iy=0; iy<(int)cc->size(1); ++iy )
				for( int iz=0; iz<(int)cc->size(2); ++iz )
					(*cc)(ix,iy,iz) = (*uc)(ix,iy,iz) - ucsave(ix,iy,iz);
		if( !parent_multi && m_bperiodic && ilevel <= m_ilevelmin )
			make_periodic( cc );
		cc_for_parent[uc] = cc;
	}

	for( size_t b=0; b<nb; ++b ){
		meshvar_bnd & cc = *(cc_for_parent[ ucs[b] ]);
		m_gridop.prolong_add( cc, *(ufs[b]) );
	}

	for( auto & kv : ucsave_for_parent ){ kv.second->deallocate(); delete kv.second; }
	for( auto & kv : cc_for_parent     ){ kv.second->deallocate(); delete kv.second; }

	// --- post-smoothing sweeps (per box) ------------------------------
	for( unsigned i=0; i<m_npostsmooth; ++i ){
		for( size_t b=0; b<nb; ++b ){
			interp().interp_coarse_fine(ilevel, *(ucs[b]), *(ufs[b]));
			if( m_smoother == opt::sm_gauss_seidel )
				GaussSeidel( h, ufs[b], ffs[b] );
			else if( m_smoother == opt::sm_jacobi )
				Jacobi( h, ufs[b], ffs[b] );
			else if( m_smoother == opt::sm_sor )
				SOR( h, ufs[b], ffs[b] );
		}
	}
}

template< class S, class I, class O, typename T >
double solver<S,I,O,T>::compute_error( const MeshvarBnd<T>& u, const MeshvarBnd<T>& f, int ilevel )
{
	int 
		nx = u.size(0), 
		ny = u.size(1), 
		nz = u.size(2);
	
	double err = 0.0, err2 = 0.0;
	size_t count = 0;

	double h = 1.0/(1ul<<ilevel), h2=h*h;
	
	#pragma omp parallel for reduction(+:err,count)
	for( int ix=0; ix<nx; ++ix )
		for( int iy=0; iy<ny; ++iy )
			for( int iz=0; iz<nz; ++iz )
			  if( true )//fabs(unew(ix,iy,iz)) > 0.0 )//&& u(ix,iy,iz) != unew(ix,iy,iz) )
				{
				  //err += fabs(1.0 - (double)u(ix,iy,iz)/(double)unew(ix,iy,iz));
				  /*err += fabs(((double)m_scheme.apply( u, ix, iy, iz )/h2 + (double)(f(ix,iy,iz)) ));
				    err2 += fabs((double)f(ix,iy,iz));*/

				  err += fabs( (double)m_scheme.apply( u, ix, iy, iz )/h2/(double)(f(ix,iy,iz)) + 1.0 );
					++count;
				}
	
	  if( count != 0 )
	    err /= count; 
	  
	return err;
}

template< class S, class I, class O, typename T >
double solver<S,I,O,T>::compute_error( const GridHierarchy<T>& uh, const GridHierarchy<T>& fh, bool verbose )
{
	double maxerr = 0.0;

	for( unsigned ilevel=uh.levelmin(); ilevel <= uh.levelmax(); ++ilevel )
	{
		double h = 1.0/(1ul<<ilevel), h2=h*h;

		// D.3.3: at multi-box levels iterate per-box; at single-box levels use union.
		const size_t nb = uh.num_boxes(ilevel);
		const bool multi = (nb > 1);

		double err = 0.0, mean_res = 0.0;
		size_t count = 0;

		for( size_t b=0; b < (multi ? nb : 1); ++b ){
			const MeshvarBnd<T> * pu = multi ? uh.get_grid(ilevel, b) : uh.get_grid(ilevel);
			const MeshvarBnd<T> * pf = multi ? fh.get_grid(ilevel, b) : fh.get_grid(ilevel);
			int nx = pu->size(0), ny = pu->size(1), nz = pu->size(2);
			double berr = 0.0, bres = 0.0; size_t bcnt = 0;
			#pragma omp parallel for reduction(+:berr,bres,bcnt)
			for( int ix=0; ix<nx; ++ix )
			  for( int iy=0; iy<ny; ++iy )
			    for( int iz=0; iz<nz; ++iz )
				{
				  double res =  (double)m_scheme.apply( *pu, ix, iy, iz ) + h2 * (double)((*pf)(ix,iy,iz));
				  double val = (*pu)( ix, iy, iz );
				  if( fabs(val) > 0.0 ){
				      berr += fabs( res/val );
				      bres += fabs(res);
				      ++bcnt;
				    }
				}
			err += berr; mean_res += bres; count += bcnt;
		}

		if( count != 0 )
		  {
		    err /= count;
		    mean_res /= count;
		  }
		if( verbose )
			std::cout << "      Level " << std::setw(6) << ilevel << ",   Error = " << err << std::endl;

		LOGDEBUG("[mg]      level %3d,  residual %g,  rel. error %g",ilevel, mean_res, err);

		maxerr = std::max(maxerr,err);

	}
	return maxerr;
}

template< class S, class I, class O, typename T >
double solver<S,I,O,T>::compute_RMS_resid( const GridHierarchy<T>& uh, const GridHierarchy<T>& fh, bool verbose )
{
	if( m_is_ini )
		m_residu_ini.assign( uh.levelmax()+1, 0.0 );
	
	double maxerr=0.0;
	
	for( unsigned ilevel=uh.levelmin(); ilevel <= uh.levelmax(); ++ilevel )
	{
		double h = 1.0/(1<<ilevel), h2=h*h;

		// D.3.3: at multi-box levels iterate per-box; at single-box levels use union.
		const size_t nb = uh.num_boxes(ilevel);
		const bool multi = (nb > 1);

		double sum = 0.0, sumd2 = 0.0;
		size_t count = 0;

		for( size_t b=0; b < (multi ? nb : 1); ++b ){
			const MeshvarBnd<T> * pu = multi ? uh.get_grid(ilevel, b) : uh.get_grid(ilevel);
			const MeshvarBnd<T> * pf = multi ? fh.get_grid(ilevel, b) : fh.get_grid(ilevel);
			int nx = pu->size(0), ny = pu->size(1), nz = pu->size(2);
			double bsum=0.0, bsumd2=0.0; size_t bcnt=0;
			#pragma omp parallel for reduction(+:bsum,bsumd2,bcnt)
			for( int ix=0; ix<nx; ++ix )
				for( int iy=0; iy<ny; ++iy )
					for( int iz=0; iz<nz; ++iz )
					{
						double d = (double)(*pf)(ix,iy,iz);
						bsumd2 += d*d;
						double r = ((double)m_scheme.apply( *pu, ix, iy, iz )/h2 + d);
						bsum += r*r;
						++bcnt;
					}
			sum += bsum; sumd2 += bsumd2; count += bcnt;
		}
		
		if( m_is_ini )
			m_residu_ini[ilevel] =  sqrt(sum)/count;
		
		double err_abs = sqrt(sum/count);
		double err_rel = err_abs / sqrt(sumd2/count);
		
		if( verbose && !m_is_ini )
			std::cout << "      Level " << std::setw(6) << ilevel << ",   Error = " << err_rel << std::endl;		
		
		LOGDEBUG("[mg]      level %3d,  rms residual %g,  rel. error %g",ilevel, err_abs, err_rel);
		
		if( err_rel > maxerr )
			maxerr = err_rel;
		
	}
	
	if( m_is_ini )
		m_is_ini = false;
	
	return maxerr;
}


template< class S, class I, class O, typename T >
double solver<S,I,O,T>::solve( GridHierarchy<T>& uh, double acc, double h, bool verbose )
{

	double err, maxerr = 1e30;
	unsigned niter = 0;

	bool fullverbose = false;

	m_pu = &uh;

	// D.5: tell the user once what the multibox MPI parallelism profile is.
	// The per-cluster V-cycle (twoGrid_multibox) is communication-free and
	// runs on rank-0 only; workers idle during the multibox Poisson solve
	// (cluster-to-rank parallelization deferred). Single-box (zoom) levels
	// still distribute the finest-level GS smoother via mg_dist.
	{
		static bool s_logged_multibox_mpi = false;
		if( !s_logged_multibox_mpi ){
			s_logged_multibox_mpi = true;
			bool any_multi = false;
			for( unsigned L=uh.levelmin(); L<=uh.levelmax(); ++L )
				if( uh.num_boxes(L) > 1 ){ any_multi = true; break; }
			if( any_multi && MUSIC::mpi::size() > 1 ){
				LOGINFO("Multibox + MPI: per-cluster V-cycle runs on rank-0 only "
				        "(%d worker rank(s) idle during multibox Poisson solve). "
				        "Per-box meshes too small for slab distribute to amortize; "
				        "cluster-to-rank parallelization deferred to D.6.",
				        MUSIC::mpi::size()-1);
			}
		}
	}

	//err = compute_RMS_resid( *m_pu, *m_pf, fullverbose );

	MUSIC::mg_profile::reset();
	double _tsolve = MUSIC::mg_profile::tic();

	//... iterate ...//
	while (true)
	{

		LOGUSER("Performing multi-grid V-cycle...");
		twoGrid( uh.levelmax() );

		//err = compute_RMS_resid( *m_pu, *m_pf, fullverbose );
		err = compute_error( *m_pu, *m_pf, fullverbose );
		++niter;

		if( fullverbose ){
			LOGUSER("  multigrid iteration %3d, maximum RMS residual = %g", niter, err );
			std::cout << "   - Step No. " << std::setw(3) << niter << ", Max Err = " << err << std::endl;
			std::cout << "     ---------------------------------------------------\n";
		}

		if( err < maxerr )
			maxerr = err;

		if( (niter > 1) && ((err < acc) || (niter > 20)) )
			break;
	}

	{
		double _tw = MUSIC::mg_profile::tic() - _tsolve;
		char tag[64];
		snprintf(tag, sizeof(tag), "solve niter=%u wall=%.4fs", niter, _tw);
		MUSIC::mg_profile::report(tag, uh.levelmin(), uh.levelmax());
		MUSIC::mg_profile::reset();
	}
	
	if( err > acc )
	{
		std::cout << "Error : no convergence in Poisson solver" << std::endl;
		LOGERR("No convergence in Poisson solver, final error: %g.",err);
	}
	else if( verbose )
	{
		std::cout << " - Converged in " << niter << " steps to " << maxerr << std::endl;
		LOGUSER("Poisson solver converged to max. error of %g in %d steps.",err,niter);
	}

	//.. make sure that the RHS does not contain the FAS corrections any more
	// D.3.3: at multi-box levels, each per-box ff restricts into its own parent
	// fc (looked up via parent_box_index). At single-box levels, fall through
	// to union restrict.
	for( int i=m_pf->levelmax(); i>0; --i ){
		size_t nb = m_pf->num_boxes(i);
		if( nb > 1 ){
			bool parent_multi = (m_pf->num_boxes(i-1) > 1);
			for( size_t b=0; b<nb; ++b ){
				MeshvarBnd<T> * ff = m_pf->get_grid(i, b);
				MeshvarBnd<T> * fc = parent_multi
					? m_pf->get_grid(i-1, m_pf->parent_box_index(i, b))
					: m_pf->get_grid(i-1);
				m_gridop.restrict( *ff, *fc );
			}
		} else {
			m_gridop.restrict( *m_pf->get_grid(i), *m_pf->get_grid(i-1) );
		}
	}


	return err;
}



//TODO: this only works for 2nd order! (but actually not needed)
template< class S, class I, class O, typename T >
void solver<S,I,O,T>::setBC( unsigned ilevel )
{
	//... set only on level before additional refinement starts
	if( ilevel == m_ilevelmin )
	{
		MeshvarBnd<T> *u = m_pu->get_grid(ilevel);
		int
			nx = u->size(0), 
			ny = u->size(1), 
			nz = u->size(2);
			
		for( int iy=0; iy<ny; ++iy )
			for( int iz=0; iz<nz; ++iz )
			{
				(*u)(-1,iy,iz) = 2.0*(*m_pubnd)(-1,iy,iz) - (*u)(0,iy,iz);
				(*u)(nx,iy,iz) = 2.0*(*m_pubnd)(nx,iy,iz) - (*u)(nx-1,iy,iz);;
			}
		
		for( int ix=0; ix<nx; ++ix )
			for( int iz=0; iz<nz; ++iz )
			{
				(*u)(ix,-1,iz) = 2.0*(*m_pubnd)(ix,-1,iz) - (*u)(ix,0,iz);
				(*u)(ix,ny,iz) = 2.0*(*m_pubnd)(ix,ny,iz) - (*u)(ix,ny-1,iz);
			}
		
		for( int ix=0; ix<nx; ++ix )
			for( int iy=0; iy<ny; ++iy )
			{
				(*u)(ix,iy,-1) = 2.0*(*m_pubnd)(ix,iy,-1) - (*u)(ix,iy,0);
				(*u)(ix,iy,nz) = 2.0*(*m_pubnd)(ix,iy,nz) - (*u)(ix,iy,nz-1);
			}		
		
		
		
	}
}



//... enforce periodic boundary conditions
template< class S, class I, class O, typename T >
void solver<S,I,O,T>::make_periodic( MeshvarBnd<T> *u )
{
	

	int
		nx = u->size(0), 
		ny = u->size(1), 
		nz = u->size(2);
	int nb = u->m_nbnd;
	
		
	//if( u->offset(0) == 0 )
		for( int iy=-nb; iy<ny+nb; ++iy )
			for( int iz=-nb; iz<nz+nb; ++iz )
			{
				int iiy( (iy+ny)%ny ), iiz( (iz+nz)%nz );
				
				for( int i=-nb; i<0; ++i )
				{
					(*u)(i,iy,iz) = (*u)(nx+i,iiy,iiz);
					(*u)(nx-1-i,iy,iz) = (*u)(-1-i,iiy,iiz);	
				}
				
			}
	
	//if( u->offset(1) == 0 )
		for( int ix=-nb; ix<nx+nb; ++ix )
			for( int iz=-nb; iz<nz+nb; ++iz )
			{
				int iix( (ix+nx)%nx ), iiz( (iz+nz)%nz );
				
				for( int i=-nb; i<0; ++i )
				{
					(*u)(ix,i,iz) = (*u)(iix,ny+i,iiz);
					(*u)(ix,ny-1-i,iz) = (*u)(iix,-1-i,iiz);
				}
			}
	
	//if( u->offset(2) == 0 )
		for( int ix=-nb; ix<nx+nb; ++ix )
			for( int iy=-nb; iy<ny+nb; ++iy )
			{
				int iix( (ix+nx)%nx ), iiy( (iy+ny)%ny );
				
				for( int i=-nb; i<0; ++i )
				{
					(*u)(ix,iy,i) = (*u)(iix,iiy,nz+i);
					(*u)(ix,iy,nz-1-i) = (*u)(iix,iiy,-1-i);
				}
			}
	
}


END_MULTIGRID_NAMESPACE
 
#endif
