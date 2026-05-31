/*
 
 mesh.hh - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010  Oliver Hahn
 
*/

#ifndef __MESH_HH
#define __MESH_HH

#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <stdexcept>

#include <math.h>

#include "config_file.hh"
#include "log.hh"
#include "mpi_helper.hh"


#include "region_generator.hh"

// mpi_poisson.hh forward-declares MeshvarBnd (the full template is defined
// later in this file), so this include is non-circular. We need it for
// MUSIC::poisson::{phase_scope, rank0_dist_solve, rank0_dist_solve_slab,
// set_slab_solve_inout} used by test_slab_solve_at below.
#include "mpi_poisson.hh"
#include "zoom_slab.hh"

class refinement_mask
{
protected:
    std::vector<short> mask_;
    size_t nx_, ny_, nz_;
    
public:
    
    refinement_mask( void )
    : nx_( 0 ), ny_ ( 0 ), nz_( 0 )
    { }
    
    refinement_mask( size_t nx, size_t ny, size_t nz, short value = 0. )
    : nx_( nx ), ny_( ny ), nz_( nz )
    {
        mask_.assign( nx_*ny_*nz_, value );
    }
    
    refinement_mask( const refinement_mask& r )
    {
        nx_ = r.nx_;
        ny_ = r.ny_;
        nz_ = r.nz_;
        mask_ = r.mask_;
    }
    
    refinement_mask& operator=( const refinement_mask& r )
    {
        nx_ = r.nx_;
        ny_ = r.ny_;
        nz_ = r.nz_;
        mask_ = r.mask_;
        
        return *this;
    }
    
    void init( size_t nx, size_t ny, size_t nz, short value = 0. )
    {
        nx_ = nx;
        ny_ = ny;
        nz_ = nz;
        mask_.assign( nx_*ny_*nz_, value );
    }
    
    const short& operator()( size_t i, size_t j, size_t k ) const
    {
        return mask_[ (i*ny_+j)*nz_+k ];
    }
    
    short& operator()( size_t i, size_t j, size_t k )
    {
        return mask_[ (i*ny_+j)*nz_+k ];
    }
    
    size_t count_flagged( void )
    {
        size_t count = 0;
        for( size_t i=0; i<mask_.size(); ++i )
            if( mask_[i] )
                ++count;
        return count;
    }
    
    size_t count_notflagged( void )
    {
        size_t count = 0;
        for( size_t i=0; i<mask_.size(); ++i )
            if( !mask_[i] )
                ++count;
        return count;
    }
};

//! base class for all things that have rectangular mesh structure
template<typename T>
class Meshvar{
public:
	typedef T real_t;

	size_t
		m_nx,	//!< x-extent of the rectangular mesh (LOCAL, in distributed mode)
		m_ny,	//!< y-extent of the rectangular mesh
		m_nz;	//!< z-extent of the rectangular mesh

	int
		m_offx, //!< x-offset of the grid (just as a helper, not used inside the class)
		m_offy, //!< y-offset of the grid (just as a helper, not used inside the class)
		m_offz;	//!< z-offset of the grid (just as a helper, not used inside the class)

	// ---- slab-distribution metadata (opt-in; default = not distributed) ----
	// In distributed mode, the data array stores only the local x-slab of a
	// larger global grid. (m_nx, m_ny, m_nz) above refer to the LOCAL extent;
	// the fields below describe the global grid and the local-slab boundaries.
	bool m_is_distributed; //!< false: m_pdata is the full grid (legacy path).
	size_t m_global_nx;    //!< full x-extent of the global grid
	size_t m_global_ny;    //!< full y-extent (= m_ny in slab decomp)
	size_t m_global_nz;    //!< full z-extent (= m_nz in slab decomp)
	size_t m_local_x_start;//!< first global ix owned by this rank
	size_t m_local_nx;     //!< number of global ix owned by this rank (== m_nx in slab decomp)

	real_t * m_pdata; //!< pointer to the dynamic data array

	//! constructor for cubic mesh
	explicit Meshvar( size_t n, int offx, int offy, int offz )
	: m_nx( n ), m_ny( n ), m_nz( n ), m_offx( offx ), m_offy( offy ), m_offz( offz ),
	  m_is_distributed(false), m_global_nx(n), m_global_ny(n), m_global_nz(n),
	  m_local_x_start(0), m_local_nx(n)
	{
		m_pdata = new real_t[m_nx*m_ny*m_nz]();
	}

	//! constructor for rectangular mesh
	Meshvar( size_t nx, size_t ny, size_t nz, int offx, int offy, int offz )
	: m_nx( nx ), m_ny( ny ), m_nz( nz ), m_offx( offx ), m_offy( offy ), m_offz( offz ),
	  m_is_distributed(false), m_global_nx(nx), m_global_ny(ny), m_global_nz(nz),
	  m_local_x_start(0), m_local_nx(nx)
	{
		m_pdata = new real_t[m_nx*m_ny*m_nz]();
	}
	
	//! variant copy constructor with optional copying of the actual data
	Meshvar( const Meshvar<real_t>& m, bool copy_over=true )
	{
		m_nx = m.m_nx;
		m_ny = m.m_ny;
		m_nz = m.m_nz;

		m_offx = m.m_offx;
		m_offy = m.m_offy;
		m_offz = m.m_offz;

		m_is_distributed = m.m_is_distributed;
		m_global_nx = m.m_global_nx;
		m_global_ny = m.m_global_ny;
		m_global_nz = m.m_global_nz;
		m_local_x_start = m.m_local_x_start;
		m_local_nx = m.m_local_nx;

		if( copy_over ) {
			m_pdata = new real_t[m_nx*m_ny*m_nz];
			for( size_t i=0; i<m_nx*m_ny*m_nz; ++i )
				m_pdata[i] = m.m_pdata[i];
		} else {
			m_pdata = new real_t[m_nx*m_ny*m_nz]();
		}
	}

	//! standard copy constructor
	explicit Meshvar( const Meshvar<real_t>& m )
	{
		m_nx = m.m_nx;
		m_ny = m.m_ny;
		m_nz = m.m_nz;

		m_offx = m.m_offx;
		m_offy = m.m_offy;
		m_offz = m.m_offz;

		m_is_distributed = m.m_is_distributed;
		m_global_nx = m.m_global_nx;
		m_global_ny = m.m_global_ny;
		m_global_nz = m.m_global_nz;
		m_local_x_start = m.m_local_x_start;
		m_local_nx = m.m_local_nx;

		m_pdata = new real_t[m_nx*m_ny*m_nz];

		for( size_t i=0; i<m_nx*m_ny*m_nz; ++i )
			m_pdata[i] = m.m_pdata[i];
	}
	
	//! destructor
	~Meshvar()
	{
		if( m_pdata != NULL )
			delete[] m_pdata;
	}
	
	//! deallocate the data, but keep the structure
	inline void deallocate( void )
	{
		if( m_pdata != NULL )
			delete[] m_pdata;
		m_pdata = NULL;
	}
	
	//! get extent of the mesh along a specified dimension (const)
	inline size_t size( unsigned dim ) const
	{
		if( dim == 0 ) return m_nx;
		if( dim == 1 ) return m_ny;
		return m_nz;
	}
	
	//! get offset of the mesh along a specified dimension  (const)
	inline int offset( unsigned dim ) const
	{
		if( dim == 0 ) return m_offx;
		if( dim == 1 ) return m_offy;
		return m_offz;
	}
	
	//! get extent of the mesh along a specified dimension
	inline int& offset( unsigned dim )
	{
		if( dim == 0 ) return m_offx;
		if( dim == 1 ) return m_offy;
		return m_offz;
	}
	
	//! set all the data to zero values
	void zero( void )
	{
		for( size_t i=0; i<m_nx*m_ny*m_nz; ++i )
			m_pdata[i] = 0.0;
	}
	
	//! direct array random acces to the data block
	inline real_t * operator[]( const size_t i )
	{	return &m_pdata[i];	}

	//! direct array random acces to the data block (const)
	inline const real_t * operator[]( const size_t i ) const
	{	return &m_pdata[i];	}
	
	//! 3D random access to the data block via index 3-tuples
	inline real_t& operator()(const int ix, const int iy, const int iz )
	{
#ifdef DEBUG
        if( ix<0||ix>=(int)m_nx||iy<0||iy>=(int)m_ny||iz<0||iz>=(int)m_nz)
            LOGERR("Array index (%d,%d,%d) out of bounds",ix,iy,iz);
#endif
        
        return m_pdata[ ((size_t)ix*m_ny+(size_t)iy)*m_nz + (size_t)iz ];
    }
	
	//! 3D random access to the data block via index 3-tuples (const)
	inline const real_t& operator()(const int ix, const int iy, const int iz ) const
	{
#ifdef DEBUG
        if( ix<0||ix>=(int)m_nx||iy<0||iy>=(int)m_ny||iz<0||iz>=(int)m_nz)
            LOGERR("Array index (%d,%d,%d) out of bounds",ix,iy,iz);
#endif
        
        return m_pdata[ ((size_t)ix*m_ny+(size_t)iy)*m_nz + (size_t)iz ];
    }
	
	//! direct multiplication of the whole data block with a number
	Meshvar<real_t>& operator*=( real_t x )
	{
		for( size_t i=0; i<m_nx*m_ny*m_nz; ++i )
			m_pdata[i] *= x;
		return *this;
	}
	
	//! direct addition of a number to the whole data block
	Meshvar<real_t>& operator+=( real_t x )
	{
		for( size_t i=0; i<m_nx*m_ny*m_nz; ++i )
			m_pdata[i] += x;
		return *this;
	}

	//! direct element-wise division of the whole data block by a number
	Meshvar<real_t>& operator/=( real_t x )
	{
		for( size_t i=0; i<m_nx*m_ny*m_nz; ++i )
			m_pdata[i] /= x;
		return *this;
	}
	
	
	//! direct subtraction of a number from the whole data block
	Meshvar<real_t>& operator-=( real_t x )
	{
		for( size_t i=0; i<m_nx*m_ny*m_nz; ++i )
			m_pdata[i] -= x;
		return *this;
	}
	
	//! direct element-wise multiplication with another compatible mesh
	Meshvar<real_t>& operator*=( const Meshvar<real_t>& v )
	{
		if( v.m_nx*v.m_ny*v.m_nz != m_nx*m_ny*m_nz )
		{
            LOGERR("Meshvar::operator*= : attempt to operate on incompatible data");
            throw std::runtime_error("Meshvar::operator*= : attempt to operate on incompatible data");
		}
		for( size_t i=0; i<m_nx*m_ny*m_nz; ++i )
			m_pdata[i] *= v.m_pdata[i];
		
		return *this;
	}
	
	//! direct element-wise division with another compatible mesh
	Meshvar<real_t>& operator/=( const Meshvar<real_t>& v )
	{
		if( v.m_nx*v.m_ny*v.m_nz != m_nx*m_ny*m_nz )
		{
            LOGERR("Meshvar::operator/= : attempt to operate on incompatible data");
            throw std::runtime_error("Meshvar::operator/= : attempt to operate on incompatible data");
		}
        
		for( size_t i=0; i<m_nx*m_ny*m_nz; ++i )
			m_pdata[i] /= v.m_pdata[i];
		
		return *this;
	}
	
	//! direct element-wise addition of another compatible mesh
	Meshvar<real_t>& operator+=( const Meshvar<real_t>& v )
	{
		if( v.m_nx*v.m_ny*v.m_nz != m_nx*m_ny*m_nz )
		{
            LOGERR("Meshvar::operator+= : attempt to operate on incompatible data");
            throw std::runtime_error("Meshvar::operator+= : attempt to operate on incompatible data");
		}
		for( size_t i=0; i<m_nx*m_ny*m_nz; ++i )
			m_pdata[i] += v.m_pdata[i];
		
		return *this;
	}
	
	//! direct element-wise subtraction of another compatible mesh
	Meshvar<real_t>& operator-=( const Meshvar<real_t>& v )
	{
		if( v.m_nx*v.m_ny*v.m_nz != m_nx*m_ny*m_nz )
		{
            LOGERR("Meshvar::operator-= : attempt to operate on incompatible data");
            throw std::runtime_error("Meshvar::operator-= : attempt to operate on incompatible data");
		}
		for( size_t i=0; i<m_nx*m_ny*m_nz; ++i )
			m_pdata[i] -= v.m_pdata[i];
		
		return *this;
	}
	
	//! assignment operator for rectangular meshes
	Meshvar<real_t>& operator=( const Meshvar<real_t>& m )
	{
		m_nx = m.m_nx;
		m_ny = m.m_ny;
		m_nz = m.m_nz;

		m_offx = m.m_offx;
		m_offy = m.m_offy;
		m_offz = m.m_offz;

		m_is_distributed = m.m_is_distributed;
		m_global_nx = m.m_global_nx;
		m_global_ny = m.m_global_ny;
		m_global_nz = m.m_global_nz;
		m_local_x_start = m.m_local_x_start;
		m_local_nx = m.m_local_nx;

		if( m_pdata != NULL )
			delete[] m_pdata;

		m_pdata = new real_t[m_nx*m_ny*m_nz];

		for( size_t i=0; i<m_nx*m_ny*m_nz; ++i )
			m_pdata[i] = m.m_pdata[i];

		return *this;
	}

	real_t* get_ptr( void )
	{	return m_pdata;		}

	const real_t* get_ptr( void ) const
	{	return m_pdata;		}

	// ------------------------------------------------------------------------
	// Slab-distribution interface (opt-in). See class-level fields above.
	// ------------------------------------------------------------------------

	//! true if this Meshvar stores only a slab of a larger global grid
	inline bool is_distributed() const { return m_is_distributed; }

	//! global x-extent of the full (possibly distributed) grid
	inline size_t global_size( unsigned dim ) const
	{
		if( dim == 0 ) return m_global_nx;
		if( dim == 1 ) return m_global_ny;
		return m_global_nz;
	}

	//! first global ix owned by this rank (== 0 when not distributed)
	inline size_t local_x_start() const { return m_local_x_start; }

	//! number of global ix owned by this rank (== m_global_nx when not distributed)
	inline size_t local_nx() const { return m_local_nx; }

	//! one-past-last global ix owned by this rank
	inline size_t local_x_end() const { return m_local_x_start + m_local_nx; }

	//! does global ix lie inside this rank's slab?
	inline bool owns_global_x( size_t gix ) const
	{
		return gix >= m_local_x_start && gix < m_local_x_start + m_local_nx;
	}

	//! reinterpret this Meshvar as a slab of a global grid. Caller is responsible
	//! for ensuring m_nx == local_nx_in (or local_nx_in+2*nbnd for MeshvarBnd).
	void mark_as_distributed( size_t global_nx_in, size_t global_ny_in, size_t global_nz_in,
	                          size_t local_x_start_in, size_t local_nx_in )
	{
		m_is_distributed = true;
		m_global_nx = global_nx_in;
		m_global_ny = global_ny_in;
		m_global_nz = global_nz_in;
		m_local_x_start = local_x_start_in;
		m_local_nx = local_nx_in;
	}
};

//! MeshvarBnd derived class adding boundary ghost cell functionality
template< typename T >
class MeshvarBnd : public Meshvar< T >{
	using Meshvar<T>::m_nx;
	using Meshvar<T>::m_ny;
	using Meshvar<T>::m_nz;
	using Meshvar<T>::m_pdata;
	
	
	
public:
	typedef T real_t;
	
	//! number of boundary (ghost) cells
	int m_nbnd;		
	
	//! most general constructor
	MeshvarBnd( int nbnd, size_t nx, size_t ny, size_t nz, size_t xoff, size_t yoff, size_t zoff )
	: Meshvar<real_t>( nx+2*nbnd, ny+2*nbnd, nz+2*nbnd, xoff, yoff, zoff ), m_nbnd( nbnd )
	{ 	}

	//! zero-offset constructor
	MeshvarBnd( size_t nbnd, size_t nx, size_t ny, size_t nz )
	: Meshvar<real_t>( nx+2*nbnd, ny+2*nbnd, nz+2*nbnd, 0, 0, 0 ), m_nbnd( nbnd )
	{ 	}
	
	//! constructor for cubic meshes
	MeshvarBnd( size_t nbnd, size_t n, size_t xoff, size_t yoff, size_t zoff )
	: Meshvar<real_t>( n+2*nbnd, xoff, yoff, zoff ), m_nbnd( nbnd )
	{ 	}
	
	//! constructor for cubic meshes with zero offset
	MeshvarBnd( size_t nbnd, size_t n )
	: Meshvar<real_t>( n+2*nbnd, 0, 0, 0 ), m_nbnd( nbnd )
	{ 	}
	
	//! modified copy constructor, allows to avoid copying actual data
	MeshvarBnd( const MeshvarBnd<real_t>& v, bool copyover )
	: Meshvar<real_t>( v, copyover ), m_nbnd( v.m_nbnd )
	{   }
	
	//! copy constructor
	explicit MeshvarBnd( const MeshvarBnd<real_t>& v )
	: Meshvar<real_t>( v, true ), m_nbnd( v.m_nbnd )
	{   }
	
	//! get extent of the mesh along a specified dimension
	inline size_t size( unsigned dim=0 ) const
	{
		if( dim == 0 ) return m_nx-2*m_nbnd;
		if( dim == 1 ) return m_ny-2*m_nbnd;
		return m_nz-2*m_nbnd;
	}

	//! 3D random access to the data block via index 3-tuples
	inline real_t& operator()(const int ix, const int iy, const int iz )
	{
		size_t iix(ix+m_nbnd), iiy(iy+m_nbnd), iiz(iz+m_nbnd);
		return m_pdata[ (iix*m_ny+iiy)*m_nz + iiz ];
	}
	
	//! 3D random access to the data block via index 3-tuples (const)
	inline const real_t& operator()(const int ix, const int iy, const int iz ) const
	{
		size_t iix(ix+m_nbnd), iiy(iy+m_nbnd), iiz(iz+m_nbnd);
		return m_pdata[ (iix*m_ny+iiy)*m_nz + iiz ];
	}
	
	//! assignment operator for rectangular meshes with ghost zones
	MeshvarBnd<real_t>& operator=( const MeshvarBnd<real_t>& m )
	{
		if( this->m_nx != m.m_nx || this->m_ny != m.m_ny || this->m_nz != m.m_nz )
		{
			this->m_nx = m.m_nx;
			this->m_ny = m.m_ny;
			this->m_nz = m.m_nz;
			
			if( m_pdata != NULL )
				delete[] m_pdata;
			
			m_pdata = new real_t[m_nx*m_ny*m_nz];			
		}
		
		for( size_t i=0; i<m_nx*m_ny*m_nz; ++i )
			this->m_pdata[i] = m.m_pdata[i];
		
		return *this;
	}

	//! sets the value of all ghost zones to zero
	void zero_bnd( void )
	{
		
		int nx,ny,nz;
		nx = this->size(0);
		ny = this->size(1);
		nz = this->size(2);
		
		
		for( int j=-m_nbnd; j<ny+m_nbnd; ++j )
			for( int k=-m_nbnd; k<nz+m_nbnd; ++k ){
				for( int i=-m_nbnd;i<0;++i )
				{
					(*this)(i,j,k)  = 0.0;
					(*this)(nx-1-i,j,k) = 0.0;	
				}
				
			}
		
		for( int i=-m_nbnd; i<nx+m_nbnd; ++i )
			for( int k=-m_nbnd; k<nz+m_nbnd; ++k ){
				for( int j=-m_nbnd;j<0;++j )
				{
					(*this)(i,j,k) = 0.0;
					(*this)(i,ny-j-1,k) = 0.0;
				}
				
			}
		
		for( int i=-m_nbnd; i<nx+m_nbnd; ++i )
			for( int j=-m_nbnd; j<ny+m_nbnd; ++j ){
				for( int k=-m_nbnd;k<0;++k )
				{
					(*this)(i,j,k) = 0.0;
					(*this)(i,j,nz-k-1) = 0.0;	
				}
				
			}
	}
	
	//! outputs the data, for debugging only, not practical for large datasets
	void print( void ) const
	{
		int nbnd = m_nbnd;
		
		std::cout << "size is [" << this->size(0) << ", " << this->size(1) << ", " << this->size(2) << "]\n";
		std::cout << "ghost region has length of " << nbnd << std::endl;
		
		std::cout.precision(3);
		for(int i=-nbnd; i<(int)this->size(0)+nbnd; ++i )
		{
			std::cout << "ix = " << i << ": \n";
			
			for (int j=-nbnd; j<(int)this->size(1)+nbnd; ++j) {
				for (int k=-nbnd; k<(int)this->size(2)+nbnd; ++k) {
					if( i<0||i>=this->size(0)||j<0||j>=this->size(1)||k<0||k>=this->size(2))
						std::cout << "[" << std::setw(6) << (*this)(i,j,k) << "] ";
					else
						std::cout << std::setw(8) <<  (*this)(i,j,k) << " ";
				}
				std::cout << std::endl;
			}
			
			std::cout << std::endl;
			
		}
	}
};



// Phase E.2.0: forward declaration for slab MeshvarBnd factory used by
// GridHierarchy<T>::allocate_union_slab_at. The body lives in
// mesh_distributed.hh, which any TU that calls allocate_union_slab_at must
// also include (it pulls in <fftw3-mpi.h>).
namespace MUSIC { namespace dist {
    template<typename real_t>
    MeshvarBnd<real_t>* make_slab_meshvarbnd( int nbnd,
                                              size_t gnx, size_t gny, size_t gnz,
                                              int offx, int offy, int offz );
}}

//! one disjoint refinement box at one level (multibox Phase D.2 bookkeeping).
//! Fields mirror the per-level scalar arrays in refinement_hierarchy but per box.
//! Declared here (before GridHierarchy) so both classes can reference it.
struct LevelBox {
    unsigned oax, oay, oaz;   //!< absolute offset in fine cells at this level
    unsigned nx,  ny,  nz;    //!< extent in fine cells at this level
    int      ox,  oy,  oz;    //!< offset relative to parent box (in coarse cells at level-1)
    double   x0,  y0,  z0;    //!< absolute origin in [0,1)
    double   xl,  yl,  zl;    //!< extent in [0,1)
    size_t   parent_idx;      //!< index into level_boxes_[level-1] of the containing parent box
};

//! class that subsumes a nested grid collection
template< typename T >
class GridHierarchy
{
public:

	typedef T real_t;

	//! number of ghost cells on boundary
	size_t m_nbnd;
	
	//! highest level without adaptive refinement
	unsigned m_levelmin;
	
	//! vector of pointers to the underlying rectangular mesh data for each level
	std::vector< MeshvarBnd<T>* > m_pgrids;

	std::vector<int>
		m_xoffabs,		//!< vector of x-offsets of a level mesh relative to the coarser level
		m_yoffabs,		//!< vector of x-offsets of a level mesh relative to the coarser level
		m_zoffabs;		//!< vector of x-offsets of a level mesh relative to the coarser level

    std::vector< refinement_mask* > m_ref_masks;
    bool bhave_refmask;

    //-------- per-box meshes (Phase D.2b) --------------------------------
    //! m_pgrids_per_box_[L][b] is one MeshvarBnd per disjoint sub-mesh at
    //! level L. Allocated by populate_per_box_meshes(level_boxes). For
    //! single-box plugins (1 box per level) this carries one mesh per level
    //! sized identically to m_pgrids[L]. For region=multibox with N
    //! clusters this carries N small meshes per level. Compute paths still
    //! consume m_pgrids[L] (union mesh) — D.3 migrates them.
    std::vector< std::vector<MeshvarBnd<T>*> > m_pgrids_per_box_;
    //! Per-box absolute offsets (fine cells at this level). Indexing
    //! matches m_pgrids_per_box_.
    std::vector< std::vector<int> > m_xoffabs_per_box_, m_yoffabs_per_box_, m_zoffabs_per_box_;
    //! Phase D.3.2: per-box parent-box index at level L-1, copied from
    //! LevelBox.parent_idx so the multi-box V-cycle can route each child's
    //! restrict/prolong into the correct parent mesh when L-1 is itself
    //! multi-box.
    std::vector< std::vector<size_t> > m_parent_idx_per_box_;
    //! Phase E.1: MPI rank that owns the storage for m_pgrids_per_box_[L][b].
    //! On non-owner ranks the corresponding MeshvarBnd<T>* is NULL but the
    //! offset/parent vectors are replicated for geometry queries. Geometry
    //! is replicated on every rank so that gather/scatter helpers and the
    //! sync* methods can route data with only the box id.
    std::vector< std::vector<int> > m_pbox_owner_;
    //! Phase E.1b: per-box extent and parent-relative offset, replicated on
    //! every rank. Needed by gather_per_box_to_root() so rank 0 can allocate
    //! a MeshvarBnd<T> tenant copy of a per-box mesh it does not own.
    std::vector< std::vector<unsigned> > m_nx_per_box_, m_ny_per_box_, m_nz_per_box_;
    std::vector< std::vector<int> >      m_oxrel_per_box_, m_oyrel_per_box_, m_ozrel_per_box_;
    //! Phase E.1b: true at rank-0 slots holding a transient tenant copy of
    //! a non-owned per-box mesh (allocated by gather_per_box_to_root,
    //! released by scatter_per_box_from_root).
    std::vector< std::vector<bool> > m_pbox_tenant_;
    //---------------------------------------------------------------------

    //-------- per-cluster z-slab storage (Phase G.1) --------------------
    //! True after populate_per_box_zslabs() has been called. Default false.
    //! When false, the m_zslab_* vectors are empty and zslab_* accessors
    //! throw / return 0 / return NULL — every consumer path is gated by
    //! has_zslabs() OR the opt-in flag at the call site.
    bool m_zslab_enabled_ = false;
    //! Halo width used to allocate the local-with-halo z-slab buffers.
    //! Stored separately from m_nbnd because G.2+ may use a larger halo for
    //! restriction stencils that cross z-faces.
    int  m_zslab_halo_w_  = 0;
    //! Per-cluster z-slab layout (Phase G.0 primitive). One per (L, b).
    //! Built by populate_per_box_zslabs() over MPI_COMM_WORLD as sub_comm.
    //! When G.1+ adds true per-cluster sub-comms (round-robin or cluster
    //! groups), the sub_comm field changes but the layout footprint stays.
    std::vector< std::vector< MUSIC::zoom_slab::ZoomSlabLayout > > m_zslab_layouts_;
    //! Per-cluster z-slab data buffer (local-with-halo) on this rank.
    //! Size matches MUSIC::zoom_slab::local_with_halo_size(layout); contents follow
    //! the (i,j,k)-row-major layout described in zoom_slab.hh. Allocated on
    //! every rank in sub_comm; consumed by sync_zslab_from_per_box() and
    //! produced by sync_per_box_from_zslab().
    std::vector< std::vector< std::vector<T> > > m_zslab_data_;
    //---------------------------------------------------------------------

    //-------- per-level slab-distributed union mesh (Phase E.2.0) --------
    //! Per-level flag: true when m_pgrids[L] is a slab-distributed
    //! MeshvarBnd<T> (each rank holds only its x-slab of the global level
    //! grid). Defaults to false on every level so legacy hierarchies are
    //! unchanged. Set by allocate_union_slab_at(L). Consumers that need a
    //! full-extent union on rank 0 must wrap their compute in
    //! gather_union_to_root(L) / scatter_union_from_root(L).
    std::vector<bool> m_level_is_slab_;
    //! Phase E.2.0: true when m_pgrids[L] currently points to a rank-0
    //! transient full-union MeshvarBnd<T> tenant (allocated by
    //! gather_union_to_root). The underlying slab pointer is held in
    //! m_level_slab_backup_[L] for the duration; scatter_union_from_root
    //! ships the tenant back to all slabs, frees the tenant, and restores
    //! the slab pointer. Always false on non-root ranks. Always false on
    //! non-slab levels.
    std::vector<bool> m_level_tenant_;
    //! Phase E.2.0: rank-0 only; non-NULL while a tenant is active at
    //! level L; equals the slab pointer that m_pgrids[L] held before
    //! gather_union_to_root(L) swapped it for the tenant. Restored back
    //! into m_pgrids[L] by scatter_union_from_root / release_union_root_tenant.
    std::vector< MeshvarBnd<T>* > m_level_slab_backup_;
    //! Phase E.2.0: cached global dims of slab levels, needed by
    //! gather_union_to_root to allocate the rank-0 tenant. Zero on
    //! non-slab levels.
    std::vector<size_t> m_level_gnx_, m_level_gny_, m_level_gnz_;
    //---------------------------------------------------------------------

    //-------- per-level parent replicas (Phase G.2b B.2b.1) --------------
    //! Transient cross-rank replicas of parent boxes, indexed by [child_level]
    //! and keyed by parent_idx (∈ m_pgrids_per_box_[L-1]).
    //! Populated by broadcast_parents_to_child_owners(L) on every rank that
    //! owns at least one child of a remote parent at level L. Drained by
    //! accumulate_children_to_parents(L) or release_parent_replicas(L).
    //! Replicas are full deep-copies of the parent MeshvarBnd<T> (data + halo,
    //! same fine-cell coordinate system). NULL/empty otherwise.
    std::vector< std::map<size_t, MeshvarBnd<T>*> > m_parent_replicas_;
    //---------------------------------------------------------------------

    //-------- transient single-box-parent union scratch (Task #135 Opt B) --
    //! During a single-box-parent restrict in compute_2LPT_source_distributed,
    //! worker child-owners (which carry a size-0 union m_pgrids for this
    //! hierarchy) need a full-union-geometry target to restrict into and then
    //! consolidate to the parent owner. acquire_parent_union_scratch installs
    //! a zeroed scratch into m_pgrids[L] keyed by parent level L; release
    //! restores m_pgrids to its prior size (recorded in _prevsz_).
    std::map<unsigned, MeshvarBnd<T>*> m_union_scratch_;
    std::map<unsigned, size_t>         m_union_scratch_prevsz_;
    //---------------------------------------------------------------------

public:
    //! Phase E.1: ownership policy. Returns the rank that should hold the
    //! per-box mesh for level L, box b given a total of nboxes at that level
    //! and an MPI communicator of size nproc. E.1a keeps everything on
    //! rank 0 (no behavior change). E.1b flips to round-robin (b % nproc).
    static int default_owner_of_box( size_t /*L*/, size_t b,
                                     size_t /*nboxes*/, int nproc )
    {
        if( nproc <= 1 ) return 0;
        return (int)(b % (size_t)nproc);
    }
protected:
	
protected:
	
	//! check whether a given grid has identical hierarchy, dimensions to this 
	bool is_consistent( const GridHierarchy<T>& gh )
	{
		// SPMD-light: an empty hierarchy (m_pgrids.size() == 0) has
		// levelmax() == UINT_MAX (size_t underflow) and size(i,j) deref
		// is UB. Always take the inconsistent (rebuild) path when either
		// side is empty — even both-empty must rebuild because per-box
		// layout (m_pgrids_per_box_) may differ.
		if( m_pgrids.empty() || gh.m_pgrids.empty() )
			return false;

		if( gh.levelmax()!=levelmax() )
			return false;

		if( gh.levelmin()!=levelmin() )
			return false;

		for( unsigned i=levelmin(); i<=levelmax(); ++i )
			for( int j=0; j<3; ++j )
			{
				if( size(i,j) != gh.size(i,j) )
					return false;
				if( offset(i,j) != gh.offset(i,j) )
					return false;
			}

		return true;
	}
	
public:
	
	
	//! return a pointer to the MeshvarBnd object representing data for one level
	MeshvarBnd<T> *get_grid( unsigned ilevel )
	{	

		if( ilevel >= m_pgrids.size() )
		{
			LOGERR("Attempt to access level %d but maxlevel = %d", ilevel, m_pgrids.size()-1);
			throw std::runtime_error("Fatal: attempt to access non-existent grid");
		}
		return m_pgrids[ilevel];  
	}

	//! return a pointer to the MeshvarBnd object representing data for one level (const)	
	const MeshvarBnd<T> *get_grid( unsigned ilevel ) const
	{	
		if( ilevel >= m_pgrids.size() )
		{
            LOGERR("Attempt to access level %d but maxlevel = %d", ilevel, m_pgrids.size()-1 );
			throw std::runtime_error("Fatal: attempt to access non-existent grid");
		}

		return m_pgrids[ilevel];  
	}
	
	
	//! constructor for a collection of rectangular grids representing a multi-level hierarchy
	/*! creates an empty hierarchy, levelmin is initially zero, no grids are stored
	 * @param nbnd number of ghost zones added at the boundary
	 */
	explicit GridHierarchy( size_t nbnd )
	: m_nbnd( nbnd ), m_levelmin( 0 ), bhave_refmask( false )
	{
		m_pgrids.clear();
	}
	
	//! copy constructor
	explicit GridHierarchy( const GridHierarchy<T> & gh )
	{
		// Guard against copying an empty hierarchy: levelmax() returns
		// m_pgrids.size()-1 which underflows to UINT_MAX when empty (MPI
		// workers carry empty hierarchies under SPMD-light).
		// H.4: workers can hold a mixed shell where m_pgrids[0..L-1] are
		// full meshes (from create_base_hierarchy_slab) but m_pgrids[L]==NULL
		// after convert_level_slab_to_full. Skip NULL slots; downstream code
		// already tolerates NULL via get_grid().
		for( size_t i=0; i<gh.m_pgrids.size(); ++i )
			m_pgrids.push_back( gh.m_pgrids[i]
			    ? new MeshvarBnd<T>( *gh.m_pgrids[i] )
			    : (MeshvarBnd<T>*)NULL );

		m_nbnd = gh.m_nbnd;
		m_levelmin = gh.m_levelmin;

		m_xoffabs = gh.m_xoffabs;
		m_yoffabs = gh.m_yoffabs;
		m_zoffabs = gh.m_zoffabs;

        //ref_mask   = gh.ref_mask;
        bhave_refmask = gh.bhave_refmask;

        if( bhave_refmask )
        {
            for( size_t i=0; i<gh.m_ref_masks.size(); ++i )
                m_ref_masks.push_back( new refinement_mask( *(gh.m_ref_masks[i]) ) );
        }

        // Deep-copy per-box meshes (Phase D.2b / E.1: skip NULL non-owner slots).
        m_pgrids_per_box_.assign( gh.m_pgrids_per_box_.size(), std::vector<MeshvarBnd<T>*>() );
        for( size_t L=0; L<gh.m_pgrids_per_box_.size(); ++L )
            for( size_t b=0; b<gh.m_pgrids_per_box_[L].size(); ++b )
                m_pgrids_per_box_[L].push_back(
                    gh.m_pgrids_per_box_[L][b]
                        ? new MeshvarBnd<T>( *gh.m_pgrids_per_box_[L][b] )
                        : NULL );
        m_xoffabs_per_box_    = gh.m_xoffabs_per_box_;
        m_yoffabs_per_box_    = gh.m_yoffabs_per_box_;
        m_zoffabs_per_box_    = gh.m_zoffabs_per_box_;
        m_parent_idx_per_box_ = gh.m_parent_idx_per_box_;
        m_pbox_owner_         = gh.m_pbox_owner_;
        m_nx_per_box_         = gh.m_nx_per_box_;
        m_ny_per_box_         = gh.m_ny_per_box_;
        m_nz_per_box_         = gh.m_nz_per_box_;
        m_oxrel_per_box_      = gh.m_oxrel_per_box_;
        m_oyrel_per_box_      = gh.m_oyrel_per_box_;
        m_ozrel_per_box_      = gh.m_ozrel_per_box_;
        m_pbox_tenant_        = gh.m_pbox_tenant_;

        // Phase E.2.0: copy slab metadata. Tenant pointer is NOT copied
        // (it is a transient on the source rank only); recipients restart
        // in "no tenant active" state.
        m_level_is_slab_   = gh.m_level_is_slab_;
        m_level_gnx_       = gh.m_level_gnx_;
        m_level_gny_       = gh.m_level_gny_;
        m_level_gnz_       = gh.m_level_gnz_;
        m_level_tenant_.assign(m_level_is_slab_.size(), false);
        m_level_slab_backup_.assign(m_level_is_slab_.size(), (MeshvarBnd<T>*)NULL);
	}

	//! destructor
	~GridHierarchy()
	{
		this->deallocate();
	}
	
	//! free all memory occupied by the grid hierarchy
	void deallocate()
	{
		// Task #135 Option B: free any lingering single-box-parent union
		// scratch first; it NULLs its m_pgrids slot so the loop below cannot
		// double-free it.
		release_all_parent_union_scratch();

		// Phase E.2.0: if a slab level has an active rank-0 tenant the
		// pointer in m_pgrids holds the tenant; the underlying slab is in
		// m_level_slab_backup_. Free both to avoid leaking the slab.
		for( unsigned i=0; i<m_pgrids.size(); ++i ) {
			if( i < m_level_tenant_.size() && m_level_tenant_[i] ) {
				delete m_pgrids[i];
				if( i < m_level_slab_backup_.size() ) {
					delete m_level_slab_backup_[i];
					m_level_slab_backup_[i] = NULL;
				}
			} else {
				delete m_pgrids[i];
			}
		}
		m_pgrids.clear();
		std::vector< MeshvarBnd<T>* >().swap( m_pgrids );

		m_xoffabs.clear();
		m_yoffabs.clear();
		m_zoffabs.clear();
		m_levelmin = 0;

        for( size_t i=0; i<m_ref_masks.size(); ++i )
            delete m_ref_masks[i];
        m_ref_masks.clear();

        deallocate_per_box_meshes();

        // Phase G.2b B.2b.1: drop any lingering parent replicas. Normally
        // accumulate_children_to_parents / release_parent_replicas drain
        // them within a single V-cycle, but a thrown exception mid-cycle
        // could leave them allocated.
        release_all_parent_replicas();

        // Phase E.2.0: clear per-level slab metadata.
        m_level_is_slab_.clear();
        m_level_tenant_.clear();
        m_level_slab_backup_.clear();
        m_level_gnx_.clear(); m_level_gny_.clear(); m_level_gnz_.clear();
	}

	//! release just the per-box mesh allocations (Phase D.2b).
	void deallocate_per_box_meshes()
	{
	    for( size_t L=0; L<m_pgrids_per_box_.size(); ++L )
	        for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b )
	            delete m_pgrids_per_box_[L][b]; // delete NULL is a no-op
	    m_pgrids_per_box_.clear();
	    m_parent_idx_per_box_.clear();
	    m_xoffabs_per_box_.clear();
	    m_yoffabs_per_box_.clear();
	    m_zoffabs_per_box_.clear();
	    m_pbox_owner_.clear();
	    m_nx_per_box_.clear();
	    m_ny_per_box_.clear();
	    m_nz_per_box_.clear();
	    m_oxrel_per_box_.clear();
	    m_oyrel_per_box_.clear();
	    m_ozrel_per_box_.clear();
	    m_pbox_tenant_.clear();
	}

	//! Allocate per-box MeshvarBnd<T> meshes from a per-level box list.
	//! Replaces any prior per-box allocation. Called once per hierarchy
	//! after add_patch loop completes; takes a vector view to avoid a
	//! forward declaration of refinement_hierarchy.
	void populate_per_box_meshes( const std::vector< std::vector<LevelBox> > & level_boxes )
	{
	    deallocate_per_box_meshes();
	    size_t Lmax = level_boxes.size();
	    if( Lmax == 0 ) return;
	    m_pgrids_per_box_.assign(Lmax, std::vector<MeshvarBnd<T>*>());
	    m_xoffabs_per_box_.assign(Lmax, std::vector<int>());
	    m_yoffabs_per_box_.assign(Lmax, std::vector<int>());
	    m_zoffabs_per_box_.assign(Lmax, std::vector<int>());
	    m_parent_idx_per_box_.assign(Lmax, std::vector<size_t>());
	    m_pbox_owner_.assign(Lmax, std::vector<int>());
	    m_nx_per_box_.assign(Lmax, std::vector<unsigned>());
	    m_ny_per_box_.assign(Lmax, std::vector<unsigned>());
	    m_nz_per_box_.assign(Lmax, std::vector<unsigned>());
	    m_oxrel_per_box_.assign(Lmax, std::vector<int>());
	    m_oyrel_per_box_.assign(Lmax, std::vector<int>());
	    m_ozrel_per_box_.assign(Lmax, std::vector<int>());
	    m_pbox_tenant_.assign(Lmax, std::vector<bool>());

	    const int my_rank = MUSIC::mpi::rank();
	    const int nproc   = MUSIC::mpi::size();

	    size_t total_bytes_local = 0;
	    size_t total_bytes_global = 0;
	    size_t multi_levels = 0;

	    for( size_t L = 0; L < Lmax; ++L )
	    {
	        const auto & lvl = level_boxes[L];
	        for( size_t b = 0; b < lvl.size(); ++b )
	        {
	            const LevelBox & bx = lvl[b];
	            const int owner = default_owner_of_box(L, b, lvl.size(), nproc);

	            MeshvarBnd<T> * m = NULL;
	            if( owner == my_rank ){
	                m = new MeshvarBnd<T>(m_nbnd, bx.nx, bx.ny, bx.nz,
	                                      (unsigned)bx.ox, (unsigned)bx.oy, (unsigned)bx.oz);
	                m->zero();
	            }
	            m_pgrids_per_box_[L].push_back(m);
	            m_pbox_owner_[L].push_back(owner);
	            m_xoffabs_per_box_[L].push_back((int)bx.oax);
	            m_yoffabs_per_box_[L].push_back((int)bx.oay);
	            m_zoffabs_per_box_[L].push_back((int)bx.oaz);
	            m_parent_idx_per_box_[L].push_back(bx.parent_idx);
	            m_nx_per_box_[L].push_back(bx.nx);
	            m_ny_per_box_[L].push_back(bx.ny);
	            m_nz_per_box_[L].push_back(bx.nz);
	            m_oxrel_per_box_[L].push_back(bx.ox);
	            m_oyrel_per_box_[L].push_back(bx.oy);
	            m_ozrel_per_box_[L].push_back(bx.oz);
	            m_pbox_tenant_[L].push_back(false);

	            size_t span_x = (size_t)bx.nx + 2*m_nbnd;
	            size_t span_y = (size_t)bx.ny + 2*m_nbnd;
	            size_t span_z = (size_t)bx.nz + 2*m_nbnd;
	            size_t bytes  = sizeof(T) * span_x * span_y * span_z;
	            total_bytes_global += bytes;
	            if( owner == my_rank ) total_bytes_local += bytes;
	        }
	        if( lvl.size() > 1 )
	        {
	            ++multi_levels;
	            LOGINFO("GridHierarchy per-box: level %zu allocated %zu MeshvarBnd<T> (each excl. halo: n=(%u,%u,%u) ...)",
	                    L, lvl.size(), lvl[0].nx, lvl[0].ny, lvl[0].nz);
	        }
	    }

	    if( multi_levels > 0 ){
	        if( nproc > 1 )
	            LOGINFO("GridHierarchy::populate_per_box_meshes: per-box total = %.2f MB global, %.2f MB local (rank %d) across %zu multi-box level(s)",
	                    total_bytes_global / (1024.0*1024.0),
	                    total_bytes_local  / (1024.0*1024.0),
	                    my_rank, multi_levels);
	        else
	            LOGINFO("GridHierarchy::populate_per_box_meshes: per-box total = %.2f MB across %zu multi-box level(s)",
	                    total_bytes_global / (1024.0*1024.0), multi_levels);
	    }
	}

	//-------- multibox per-box accessors (Phase D.2b) --------------------
	//! Number of disjoint sub-meshes allocated at level L; 0 if
	//! populate_per_box_meshes was never called.
	size_t num_boxes( unsigned ilevel ) const
	{
	    if( ilevel >= m_pgrids_per_box_.size() ) return 0;
	    return m_pgrids_per_box_[ilevel].size();
	}

	//! Pointer to one per-box mesh (Phase D.2b). Falls back to the union
	//! mesh when box_id==0 and per-box meshes have not been allocated, so
	//! existing single-arg get_grid callers keep working.
	MeshvarBnd<T> * get_grid( unsigned ilevel, size_t box_id )
	{
	    if( ilevel >= m_pgrids_per_box_.size() ||
	        box_id >= m_pgrids_per_box_[ilevel].size() )
	    {
	        LOGERR("GridHierarchy::get_grid(%u, %zu): per-box mesh missing", ilevel, box_id);
	        throw std::runtime_error("GridHierarchy::get_grid: per-box mesh missing");
	    }
	    return m_pgrids_per_box_[ilevel][box_id];
	}

	const MeshvarBnd<T> * get_grid( unsigned ilevel, size_t box_id ) const
	{
	    if( ilevel >= m_pgrids_per_box_.size() ||
	        box_id >= m_pgrids_per_box_[ilevel].size() )
	    {
	        LOGERR("GridHierarchy::get_grid(%u, %zu): per-box mesh missing", ilevel, box_id);
	        throw std::runtime_error("GridHierarchy::get_grid: per-box mesh missing");
	    }
	    return m_pgrids_per_box_[ilevel][box_id];
	}

	int offset_abs( int ilevel, size_t box_id, int idim ) const
	{
	    if( idim == 0 ) return m_xoffabs_per_box_.at(ilevel).at(box_id);
	    if( idim == 1 ) return m_yoffabs_per_box_.at(ilevel).at(box_id);
	    return m_zoffabs_per_box_.at(ilevel).at(box_id);
	}

	//! Phase D.3.2: parent-box index at level L-1 for child box b at L.
	//! Returns 0 if there's only one parent box (always valid then).
	size_t parent_box_index( unsigned ilevel, size_t box_id ) const
	{
	    if( ilevel >= m_parent_idx_per_box_.size() ||
	        box_id >= m_parent_idx_per_box_[ilevel].size() )
	        return 0;
	    return m_parent_idx_per_box_[ilevel][box_id];
	}

	//! Phase E.1: rank that owns the per-box mesh for level L, box b. Defaults
	//! to 0 when ownership has not been recorded (single-rank or pre-D.2b).
	int owner_of_box( unsigned ilevel, size_t box_id ) const
	{
	    if( ilevel >= m_pbox_owner_.size() ||
	        box_id >= m_pbox_owner_[ilevel].size() )
	        return 0;
	    return m_pbox_owner_[ilevel][box_id];
	}

	//! Phase E.1: convenience -- does this MPI rank hold storage for this box.
	bool owns_box( unsigned ilevel, size_t box_id ) const
	{
	    return owner_of_box(ilevel, box_id) == MUSIC::mpi::rank();
	}

	//! Phase D.3.1: copy per-cluster sub-regions out of the union mesh
	//! m_pgrids[L] into m_pgrids_per_box_[L][b]. Per-box meshes share the
	//! same fine-cell coordinate system as the union mesh, so the copy is a
	//! windowed translation by (dx,dy,dz) = box_oa - union_oa. Cells of the
	//! per-box mesh that fall outside the union range are zero-filled.
	//! The halo (m_nbnd) is included in the copy when the corresponding
	//! union cells exist; otherwise zero. Used by densities.cc post-density
	//! and main.cc post-Poisson so D.4 output can read from per-box meshes
	//! without changing the compute path.
	void sync_per_box_from_union()
	{
	    if( m_pgrids_per_box_.empty() ) return;

	    size_t multi_log = 0;
	    for( size_t L=0; L<m_pgrids_per_box_.size(); ++L )
	    {
	        if( L >= m_pgrids.size() || m_pgrids[L] == NULL ) continue;
	        if( m_pgrids_per_box_[L].empty() ) continue;
	        // H.4 C.3: slab levels go through sync_per_box_from_union_slab
	        // (collective); the windowed-from-full copy below assumes a
	        // rank-0 full union at m_pgrids[L].
	        if( is_level_slab((unsigned)L) ) continue;

	        MeshvarBnd<T> * um = m_pgrids[L];
	        int u_oax = m_xoffabs[L];
	        int u_oay = m_yoffabs[L];
	        int u_oaz = m_zoffabs[L];
	        int unx = (int)um->size(0);
	        int uny = (int)um->size(1);
	        int unz = (int)um->size(2);
	        int unbnd = um->m_nbnd;

	        for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b )
	        {
	            MeshvarBnd<T> * bm = m_pgrids_per_box_[L][b];
	            if( !bm ) continue; // E.1: this rank does not own this per-box mesh
	            int b_oax = m_xoffabs_per_box_[L][b];
	            int b_oay = m_yoffabs_per_box_[L][b];
	            int b_oaz = m_zoffabs_per_box_[L][b];
	            int bnx = (int)bm->size(0);
	            int bny = (int)bm->size(1);
	            int bnz = (int)bm->size(2);
	            int bnbnd = bm->m_nbnd;

	            int dx = b_oax - u_oax;
	            int dy = b_oay - u_oay;
	            int dz = b_oaz - u_oaz;

	            for( int i=-bnbnd; i<bnx+bnbnd; ++i )
	            {
	                int ui = i + dx;
	                for( int j=-bnbnd; j<bny+bnbnd; ++j )
	                {
	                    int uj = j + dy;
	                    for( int k=-bnbnd; k<bnz+bnbnd; ++k )
	                    {
	                        int uk = k + dz;
	                        if( ui<-unbnd || ui>=unx+unbnd ||
	                            uj<-unbnd || uj>=uny+unbnd ||
	                            uk<-unbnd || uk>=unz+unbnd )
	                            (*bm)(i,j,k) = T(0);
	                        else
	                            (*bm)(i,j,k) = (*um)(ui,uj,uk);
	                    }
	                }
	            }
	        }

	        if( m_pgrids_per_box_[L].size() > 1 ) ++multi_log;
	    }

	    if( multi_log > 0 )
	        LOGUSER("GridHierarchy::sync_per_box_from_union: synced %zu multi-box level(s)",
	                multi_log);
	}

	//! H.4 C.3: SPMD slab-aware counterpart of sync_per_box_from_union for a
	//! single level. Each rank packs cells out of its slab strip at m_pgrids[L]
	//! for every per-box mesh that intersects its strip, sends to the box's
	//! owner. Owners receive (and read their own strip locally) into the per-
	//! box mesh, zero-filling halo cells outside the union. After this call
	//! every owner has a fully populated per-box mesh; rank 0 does NOT need
	//! the full union to be present (the slab on every rank IS the data).
	//!
	//! Preconditions:
	//!   - is_level_slab(L) == true on every rank.
	//!   - populate_per_box_meshes(...) has been called so m_pbox_owner_ and
	//!     m_pgrids_per_box_[L] are sized for level L on every rank.
	//!   - This is collective. Must run OUTSIDE phase_scope.
	void sync_per_box_from_union_slab( unsigned ilevel )
	{
#ifdef USE_MPI
	    const int nproc = MUSIC::mpi::size();
	    if( nproc <= 1 ) {
	        // Serial degenerate: slab == full union. Fall through to the
	        // existing rank-0 windowed copy by temporarily flipping the slab
	        // flag, calling sync_per_box_from_union, then restoring.
	        if( ilevel >= m_level_is_slab_.size() ) return;
	        if( ilevel >= m_pgrids_per_box_.size() ) return;
	        if( m_pgrids_per_box_[ilevel].empty() ) return;
	        if( ilevel >= m_pgrids.size() || m_pgrids[ilevel] == NULL ) return;
	        // At nproc==1 a slab equals its full-union mesh; the per-box
	        // windowed copy in sync_per_box_from_union reads m_pgrids[L]
	        // through MeshvarBnd::operator()(i,j,k) which is geometry-agnostic.
	        // No real flip needed — just iterate over the same body inline.
	        MeshvarBnd<T> * um = m_pgrids[ilevel];
	        int u_oax = m_xoffabs[ilevel];
	        int u_oay = m_yoffabs[ilevel];
	        int u_oaz = m_zoffabs[ilevel];
	        int unx = (int)um->size(0);
	        int uny = (int)um->size(1);
	        int unz = (int)um->size(2);
	        int unbnd = um->m_nbnd;
	        for( size_t b=0; b<m_pgrids_per_box_[ilevel].size(); ++b ){
	            MeshvarBnd<T> * bm = m_pgrids_per_box_[ilevel][b];
	            if( !bm ) continue;
	            int b_oax = m_xoffabs_per_box_[ilevel][b];
	            int b_oay = m_yoffabs_per_box_[ilevel][b];
	            int b_oaz = m_zoffabs_per_box_[ilevel][b];
	            int bnx = (int)bm->size(0);
	            int bny = (int)bm->size(1);
	            int bnz = (int)bm->size(2);
	            int bnbnd = bm->m_nbnd;
	            int dx = b_oax - u_oax;
	            int dy = b_oay - u_oay;
	            int dz = b_oaz - u_oaz;
	            for( int i=-bnbnd; i<bnx+bnbnd; ++i ){
	                int ui = i + dx;
	                for( int j=-bnbnd; j<bny+bnbnd; ++j ){
	                    int uj = j + dy;
	                    for( int k=-bnbnd; k<bnz+bnbnd; ++k ){
	                        int uk = k + dz;
	                        if( ui<-unbnd || ui>=unx+unbnd ||
	                            uj<-unbnd || uj>=uny+unbnd ||
	                            uk<-unbnd || uk>=unz+unbnd )
	                            (*bm)(i,j,k) = T(0);
	                        else
	                            (*bm)(i,j,k) = (*um)(ui,uj,uk);
	                    }
	                }
	            }
	        }
	        return;
	    }
	    if( ilevel >= m_pgrids_per_box_.size() ) return;
	    if( m_pgrids_per_box_[ilevel].empty() ) return;
	    if( !is_level_slab(ilevel) ) {
	        LOGERR("sync_per_box_from_union_slab: L=%u is not a slab", ilevel);
	        throw std::runtime_error("sync_per_box_from_union_slab: not a slab");
	    }

	    const int rk = MUSIC::mpi::rank();
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
	    // tag = tag_base + sender_rank; each box exchange is fully drained
	    // before the next box begins, so a single tag-per-sender suffices.
	    const int tag_base = 2700; // distinct from 1100/1300/1500/1700/2100/2300/2500

	    MeshvarBnd<T> * slab = m_pgrids[ilevel];
	    const long long my_lx = slab ? (long long)slab->local_x_start() : 0;
	    const long long my_nx = slab ? (long long)slab->local_nx()      : 0;
	    const long long gnx   = (long long)m_level_gnx_[ilevel];
	    const long long gny   = (long long)m_level_gny_[ilevel];
	    const long long gnz   = (long long)m_level_gnz_[ilevel];

	    // gather everyone's slab strip metadata
	    long long my_loc[2] = { my_lx, my_nx };
	    std::vector<long long> all_loc(2*nproc, 0);
	    MPI_Allgather(my_loc, 2, MPI_LONG_LONG, all_loc.data(), 2, MPI_LONG_LONG,
	                  MUSIC::mpi::world());

	    for( size_t b=0; b<m_pgrids_per_box_[ilevel].size(); ++b ){
	        const int owner = m_pbox_owner_[ilevel][b];
	        const int b_oax = m_xoffabs_per_box_[ilevel][b];
	        const int b_oay = m_yoffabs_per_box_[ilevel][b];
	        const int b_oaz = m_zoffabs_per_box_[ilevel][b];
	        const int bnx   = (int)m_nx_per_box_[ilevel][b];
	        const int bny   = (int)m_ny_per_box_[ilevel][b];
	        const int bnz   = (int)m_nz_per_box_[ilevel][b];
	        const int bnbnd = (int)m_nbnd;

	        // box absolute coord range (inclusive lo, exclusive hi)
	        const long long bx_lo_abs = (long long)b_oax - bnbnd;
	        const long long bx_hi_abs = (long long)b_oax + bnx + bnbnd;
	        const long long by_lo_abs = (long long)b_oay - bnbnd;
	        const long long by_hi_abs = (long long)b_oay + bny + bnbnd;
	        const long long bz_lo_abs = (long long)b_oaz - bnbnd;
	        const long long bz_hi_abs = (long long)b_oaz + bnz + bnbnd;

	        // intersect with union [0, gn{x,y,z})
	        const long long iy_lo = std::max((long long)0, by_lo_abs);
	        const long long iy_hi = std::min(gny, by_hi_abs);
	        const long long iz_lo = std::max((long long)0, bz_lo_abs);
	        const long long iz_hi = std::min(gnz, bz_hi_abs);
	        const bool any_yz = (iy_hi > iy_lo) && (iz_hi > iz_lo);

	        if( rk == owner ){
	            MeshvarBnd<T> * bm = m_pgrids_per_box_[ilevel][b];
	            if( !bm ){
	                LOGERR("sync_per_box_from_union_slab: owner missing per-box mesh L=%u b=%zu",
	                       ilevel, b);
	                throw std::runtime_error("sync_per_box_from_union_slab: owner mesh NULL");
	            }
	            // zero-fill first (covers cells outside union and any slab gaps)
	            bm->zero();
	            if( !any_yz ) continue;

	            for( int r=0; r<nproc; ++r ){
	                const long long s_lx = all_loc[2*r+0];
	                const long long s_nx = all_loc[2*r+1];
	                const long long s_hi = s_lx + s_nx;
	                const long long ix_lo = std::max(s_lx, std::max((long long)0, bx_lo_abs));
	                const long long ix_hi = std::min(s_hi, std::min(gnx, bx_hi_abs));
	                if( ix_hi <= ix_lo ) continue;

	                const long long nx_chunk = ix_hi - ix_lo;
	                const long long ny_chunk = iy_hi - iy_lo;
	                const long long nz_chunk = iz_hi - iz_lo;
	                const size_t cnt = (size_t)nx_chunk * (size_t)ny_chunk * (size_t)nz_chunk;

	                std::vector<T> buf(cnt);
	                if( r == rk ){
	                    // copy from own slab
	                    for( long long i=0; i<nx_chunk; ++i )
	                        for( long long j=0; j<ny_chunk; ++j )
	                            for( long long k=0; k<nz_chunk; ++k )
	                                buf[(size_t)((i*ny_chunk+j)*nz_chunk+k)]
	                                    = (*slab)((int)(ix_lo+i - my_lx),
	                                              (int)(iy_lo+j),
	                                              (int)(iz_lo+k));
	                } else {
	                    MPI_Recv(buf.data(), (int)cnt, dtype, r, tag_base+r,
	                             MUSIC::mpi::world(), MPI_STATUS_IGNORE);
	                }
	                // unpack into per-box mesh
	                for( long long i=0; i<nx_chunk; ++i )
	                    for( long long j=0; j<ny_chunk; ++j )
	                        for( long long k=0; k<nz_chunk; ++k ){
	                            int bi = (int)(ix_lo+i) - b_oax;
	                            int bj = (int)(iy_lo+j) - b_oay;
	                            int bk = (int)(iz_lo+k) - b_oaz;
	                            (*bm)(bi,bj,bk) = buf[(size_t)((i*ny_chunk+j)*nz_chunk+k)];
	                        }
	            }
	        } else {
	            // non-owner: ship overlap (if any) to owner
	            if( !any_yz ) continue;
	            const long long ix_lo = std::max(my_lx, std::max((long long)0, bx_lo_abs));
	            const long long ix_hi = std::min(my_lx + my_nx, std::min(gnx, bx_hi_abs));
	            if( ix_hi <= ix_lo ) continue;

	            const long long nx_chunk = ix_hi - ix_lo;
	            const long long ny_chunk = iy_hi - iy_lo;
	            const long long nz_chunk = iz_hi - iz_lo;
	            const size_t cnt = (size_t)nx_chunk * (size_t)ny_chunk * (size_t)nz_chunk;
	            std::vector<T> buf(cnt);
	            for( long long i=0; i<nx_chunk; ++i )
	                for( long long j=0; j<ny_chunk; ++j )
	                    for( long long k=0; k<nz_chunk; ++k )
	                        buf[(size_t)((i*ny_chunk+j)*nz_chunk+k)]
	                            = (*slab)((int)(ix_lo+i - my_lx),
	                                      (int)(iy_lo+j),
	                                      (int)(iz_lo+k));
	            MPI_Send(buf.data(), (int)cnt, dtype, owner, tag_base+rk,
	                     MUSIC::mpi::world());
	        }
	    }
#else
	    (void)ilevel;
#endif
	}

	//! Phase E.1b: rank-0-only. Allocate tenant MeshvarBnd<T> for every per-box
	//! slot where the owner is not rank 0, zero-initialised. Caller fills the
	//! tenants from a rank-0-resident source (e.g., sync_per_box_from_union)
	//! and then runs scatter_per_box_from_root() to ship to owners. No MPI
	//! exchange happens here, so this helper is safe to call from inside an
	//! is_root() block. Use this instead of gather_per_box_to_root when the
	//! initial data lives on rank 0 (no point asking owners for stale data).
	void alloc_root_tenants()
	{
#ifdef USE_MPI
	    if( MUSIC::mpi::size() <= 1 ) return;
	    if( m_pgrids_per_box_.empty() ) return;
	    if( MUSIC::mpi::rank() != 0 ) return;
	    for( size_t L=0; L<m_pgrids_per_box_.size(); ++L ){
	        // H.4 C.3: slab levels are populated directly into owner-side
	        // per-box meshes by sync_per_box_from_union_slab; no rank-0
	        // tenant needed (and rank-0 has no full union anyway).
	        if( is_level_slab((unsigned)L) ) continue;
	        for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b ){
	            if( m_pbox_owner_[L][b] == 0 ) continue;
	            if( m_pgrids_per_box_[L][b] != NULL ) continue;
	            MeshvarBnd<T>* t = new MeshvarBnd<T>(
	                m_nbnd,
	                m_nx_per_box_[L][b], m_ny_per_box_[L][b], m_nz_per_box_[L][b],
	                (size_t)m_oxrel_per_box_[L][b],
	                (size_t)m_oyrel_per_box_[L][b],
	                (size_t)m_ozrel_per_box_[L][b] );
	            t->zero();
	            m_pgrids_per_box_[L][b] = t;
	            m_pbox_tenant_[L][b] = true;
	        }
	    }
#endif
	}

	//! Phase E.1b: rank-0-only. Free all transient tenant meshes (allocated by
	//! either alloc_root_tenants or gather_per_box_to_root) WITHOUT shipping
	//! the data back to owners. Use this for read-only gather patterns where
	//! the owner's copy is still authoritative.
	void release_root_tenants()
	{
#ifdef USE_MPI
	    if( MUSIC::mpi::size() <= 1 ) return;
	    if( m_pgrids_per_box_.empty() ) return;
	    if( MUSIC::mpi::rank() != 0 ) return;
	    for( size_t L=0; L<m_pgrids_per_box_.size(); ++L ){
	        for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b ){
	            if( !m_pbox_tenant_[L][b] ) continue;
	            delete m_pgrids_per_box_[L][b];
	            m_pgrids_per_box_[L][b] = NULL;
	            m_pbox_tenant_[L][b] = false;
	        }
	    }
#endif
	}

	//! Phase E.1b: collective. Rank 0 acquires a current copy of every per-box
	//! mesh it does not own; owners ship the contents of their MeshvarBnd to
	//! rank 0 which allocates a tenant MeshvarBnd<T> for the duration of the
	//! compute phase. After return, on rank 0 every m_pgrids_per_box_[L][b]
	//! is non-NULL (owned or tenant); on other ranks only owned slots are
	//! non-NULL. Must be called OUTSIDE MUSIC::poisson::phase_scope (workers
	//! cannot service MPI_Send while parked in worker_pump).
	void gather_per_box_to_root()
	{
#ifdef USE_MPI
	    if( MUSIC::mpi::size() <= 1 ) return;
	    if( m_pgrids_per_box_.empty() ) return;
	    const int rk = MUSIC::mpi::rank();
	    const bool trace = (std::getenv("MUSIC_TRACE_PHASE")!=NULL);
	    if(trace){ std::fprintf(stderr,"[trace rk=%d] gather enter (this=%p)\n", rk, (void*)this); std::fflush(stderr); }
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
	    int tag = 1100; // distinct from other point-to-point tags in the codebase
	    for( size_t L=0; L<m_pgrids_per_box_.size(); ++L ){
	        for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b, ++tag ){
	            const int owner = m_pbox_owner_[L][b];
	            if( owner == 0 ) continue;
	            if(trace){ std::fprintf(stderr,"[trace rk=%d] gather L=%zu b=%zu owner=%d tag=%d %s\n",
	                       rk,L,b,owner,tag, (rk==owner?"SEND":(rk==0?"RECV":"skip"))); std::fflush(stderr); }
	            const size_t cnt = (size_t)(m_nx_per_box_[L][b] + 2*m_nbnd)
	                             * (size_t)(m_ny_per_box_[L][b] + 2*m_nbnd)
	                             * (size_t)(m_nz_per_box_[L][b] + 2*m_nbnd);
	            if( rk == owner ){
	                MeshvarBnd<T>* m = m_pgrids_per_box_[L][b];
	                if( !m ){
	                    LOGERR("gather_per_box_to_root: owner rank %d missing per-box mesh L=%zu b=%zu",
	                           rk, L, b);
	                    throw std::runtime_error("gather_per_box_to_root: owner mesh NULL");
	                }
	                MPI_Send( m->get_ptr(), (int)cnt, dtype, 0, tag, MUSIC::mpi::world() );
	            } else if( rk == 0 ){
	                if( m_pgrids_per_box_[L][b] != NULL ){
	                    LOGERR("gather_per_box_to_root: rank 0 already has slot L=%zu b=%zu (double-gather?)",
	                           L, b);
	                    throw std::runtime_error("gather_per_box_to_root: double gather");
	                }
	                MeshvarBnd<T>* t = new MeshvarBnd<T>(
	                    m_nbnd,
	                    m_nx_per_box_[L][b], m_ny_per_box_[L][b], m_nz_per_box_[L][b],
	                    (size_t)m_oxrel_per_box_[L][b],
	                    (size_t)m_oyrel_per_box_[L][b],
	                    (size_t)m_ozrel_per_box_[L][b] );
	                MPI_Recv( t->get_ptr(), (int)cnt, dtype, owner, tag,
	                          MUSIC::mpi::world(), MPI_STATUS_IGNORE );
	                m_pgrids_per_box_[L][b] = t;
	                m_pbox_tenant_[L][b] = true;
	            }
	        }
	    }
	    if(trace){ std::fprintf(stderr,"[trace rk=%d] gather exit\n", rk); std::fflush(stderr); }
#endif
	}

	//! Phase E.1b: collective. Inverse of gather_per_box_to_root. Rank 0 ships
	//! every tenant mesh back to its owner, who copies the contents into its
	//! local MeshvarBnd. Rank 0 then frees the tenant and resets the pointer
	//! to NULL. Must be called OUTSIDE phase_scope (same reason as gather).
	void scatter_per_box_from_root()
	{
#ifdef USE_MPI
	    if( MUSIC::mpi::size() <= 1 ) return;
	    if( m_pgrids_per_box_.empty() ) return;
	    const int rk = MUSIC::mpi::rank();
	    const bool trace = (std::getenv("MUSIC_TRACE_PHASE")!=NULL);
	    if(trace){ std::fprintf(stderr,"[trace rk=%d] scatter enter (this=%p)\n", rk, (void*)this); std::fflush(stderr); }
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
	    int tag = 1300; // distinct tag space from gather
	    for( size_t L=0; L<m_pgrids_per_box_.size(); ++L ){
	        for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b, ++tag ){
	            const int owner = m_pbox_owner_[L][b];
	            if( owner == 0 ) continue;
	            if(trace){ std::fprintf(stderr,"[trace rk=%d] scatter L=%zu b=%zu owner=%d tag=%d %s\n",
	                       rk,L,b,owner,tag, (rk==0?"SEND":(rk==owner?"RECV":"skip"))); std::fflush(stderr); }
	            const size_t cnt = (size_t)(m_nx_per_box_[L][b] + 2*m_nbnd)
	                             * (size_t)(m_ny_per_box_[L][b] + 2*m_nbnd)
	                             * (size_t)(m_nz_per_box_[L][b] + 2*m_nbnd);
	            if( rk == 0 ){
	                if( !m_pbox_tenant_[L][b] || m_pgrids_per_box_[L][b] == NULL ){
	                    LOGERR("scatter_per_box_from_root: no tenant for L=%zu b=%zu", L, b);
	                    throw std::runtime_error("scatter_per_box_from_root: missing tenant");
	                }
	                MeshvarBnd<T>* t = m_pgrids_per_box_[L][b];
	                MPI_Send( t->get_ptr(), (int)cnt, dtype, owner, tag, MUSIC::mpi::world() );
	                delete t;
	                m_pgrids_per_box_[L][b] = NULL;
	                m_pbox_tenant_[L][b] = false;
	            } else if( rk == owner ){
	                MeshvarBnd<T>* m = m_pgrids_per_box_[L][b];
	                if( !m ){
	                    LOGERR("scatter_per_box_from_root: owner rank %d missing per-box mesh L=%zu b=%zu",
	                           rk, L, b);
	                    throw std::runtime_error("scatter_per_box_from_root: owner mesh NULL");
	                }
	                MPI_Recv( m->get_ptr(), (int)cnt, dtype, 0, tag,
	                          MUSIC::mpi::world(), MPI_STATUS_IGNORE );
	            }
	        }
	    }
	    if(trace){ std::fprintf(stderr,"[trace rk=%d] scatter exit\n", rk); std::fflush(stderr); }
#endif
	}

	//! Phase D.3.2: reverse of sync_per_box_from_union. Copies the interior
	//! of each per-box mesh back into the corresponding window of the union
	//! mesh m_pgrids[L]. Used at end of the multi-box V-cycle so downstream
	//! consumers that still read the union mesh (gradient, output, mask)
	//! see the per-cluster solution. Halo cells are NOT pushed back (the
	//! union mesh has its own halo updated by neighbouring per-box meshes
	//! and by the make_periodic / coarse-fine BC machinery).
	//!
	//! Only multi-box levels (>1 sub-mesh) are synced. Single-box levels
	//! are owned by the union compute path; their per-box mesh is allocated
	//! for bookkeeping but never written by the solver, so copying it back
	//! would wipe the union solution to zero.
	void sync_union_from_per_box()
	{
	    if( m_pgrids_per_box_.empty() ) return;
	    for( size_t L=0; L<m_pgrids_per_box_.size(); ++L )
	    {
	        if( L >= m_pgrids.size() || m_pgrids[L] == NULL ) continue;
	        if( m_pgrids_per_box_[L].size() <= 1 ) continue;
	        MeshvarBnd<T> * um = m_pgrids[L];
	        int u_oax = m_xoffabs[L];
	        int u_oay = m_yoffabs[L];
	        int u_oaz = m_zoffabs[L];
	        int unx = (int)um->size(0);
	        int uny = (int)um->size(1);
	        int unz = (int)um->size(2);
	        for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b )
	        {
	            MeshvarBnd<T> * bm = m_pgrids_per_box_[L][b];
	            if( !bm ) continue; // E.1: non-owner has no data to push back
	            int b_oax = m_xoffabs_per_box_[L][b];
	            int b_oay = m_yoffabs_per_box_[L][b];
	            int b_oaz = m_zoffabs_per_box_[L][b];
	            int bnx = (int)bm->size(0);
	            int bny = (int)bm->size(1);
	            int bnz = (int)bm->size(2);
	            int dx = b_oax - u_oax;
	            int dy = b_oay - u_oay;
	            int dz = b_oaz - u_oaz;
	            for( int i=0; i<bnx; ++i ){
	                int ui = i+dx;
	                if( ui<0 || ui>=unx ) continue;
	                for( int j=0; j<bny; ++j ){
	                    int uj = j+dy;
	                    if( uj<0 || uj>=uny ) continue;
	                    for( int k=0; k<bnz; ++k ){
	                        int uk = k+dz;
	                        if( uk<0 || uk>=unz ) continue;
	                        (*um)(ui,uj,uk) = (*bm)(i,j,k);
	                    }
	                }
	            }
	        }
	    }
	}

	//-------- Phase G.2b B.2b.1: cross-rank parent replicas ----------------
	//! Local accessor: returns the parent MeshvarBnd that the per-box op for
	//! (level, box) should read. Returns the locally-owned parent if
	//! owner_of_box(level-1, parent_idx) == my_rank, else the transient
	//! replica from m_parent_replicas_[level][parent_idx], else NULL (caller
	//! is non-owner of every child of this parent and broadcast hasn't been
	//! called, or this rank doesn't participate in the parent's data at all).
	//! Safe to call when level==0 (returns NULL — no parent).
	MeshvarBnd<T> * parent_for_box( unsigned level, size_t box_id )
	{
	    if( level == 0 ) return NULL;
	    if( level >= m_pgrids_per_box_.size() ) return NULL;
	    if( box_id >= m_parent_idx_per_box_[level].size() ) return NULL;
	    const size_t p = m_parent_idx_per_box_[level][box_id];
	    const unsigned Lp = level - 1;
	    if( Lp < m_pgrids_per_box_.size()
	        && p < m_pgrids_per_box_[Lp].size()
	        && m_pgrids_per_box_[Lp][p] != NULL )
	        return m_pgrids_per_box_[Lp][p];
	    if( level < m_parent_replicas_.size() ){
	        auto it = m_parent_replicas_[level].find(p);
	        if( it != m_parent_replicas_[level].end() ) return it->second;
	    }
	    return NULL;
	}

	const MeshvarBnd<T> * parent_for_box( unsigned level, size_t box_id ) const
	{
	    if( level == 0 ) return NULL;
	    if( level >= m_pgrids_per_box_.size() ) return NULL;
	    if( box_id >= m_parent_idx_per_box_[level].size() ) return NULL;
	    const size_t p = m_parent_idx_per_box_[level][box_id];
	    const unsigned Lp = level - 1;
	    if( Lp < m_pgrids_per_box_.size()
	        && p < m_pgrids_per_box_[Lp].size()
	        && m_pgrids_per_box_[Lp][p] != NULL )
	        return m_pgrids_per_box_[Lp][p];
	    if( level < m_parent_replicas_.size() ){
	        auto it = m_parent_replicas_[level].find(p);
	        if( it != m_parent_replicas_[level].end() ) return it->second;
	    }
	    return NULL;
	}

	//! True when the replica map at `level` has been populated for parent
	//! index `parent_idx`. Useful for assertions.
	bool has_parent_replica( unsigned level, size_t parent_idx ) const
	{
	    if( level >= m_parent_replicas_.size() ) return false;
	    return m_parent_replicas_[level].find(parent_idx) != m_parent_replicas_[level].end();
	}

	//! Phase G.2b B.2b.1: at child-level `level` (>= 1), for every parent box
	//! p at level-1 whose owner differs from at least one child's owner, the
	//! parent owner MPI_Isends a full deep copy of m_pgrids_per_box_[level-1][p]
	//! to each distinct child-owner rank, which MPI_Irecvs into a freshly
	//! allocated MeshvarBnd<T> stored in m_parent_replicas_[level][p].
	//!
	//! Post-call invariant on every rank: parent_for_box(level, b) returns a
	//! non-NULL pointer for every locally-owned child box b. The replica is
	//! a deep copy at this instant; the parent owner's MeshvarBnd is not
	//! aliased.
	//!
	//! Must be called OUTSIDE phase_scope (workers cannot service MPI_Send
	//! while parked in worker_pump). Idempotent only if release_parent_replicas
	//! is called first — a second broadcast without release will leak.
	//!
	//! Tag layout: 1900 + level*1000 + parent_idx (parent_idx is unique within
	//! its level, so collisions across (level, parent_idx) tuples are avoided
	//! provided num_boxes(level-1) ≤ 1000).
	void broadcast_parents_to_child_owners( unsigned level )
	{
#ifdef USE_MPI
	    if( MUSIC::mpi::size() <= 1 ) return;
	    if( level == 0 ) return;
	    if( level >= m_pgrids_per_box_.size() ) return;
	    const unsigned Lp = level - 1;
	    if( Lp >= m_pgrids_per_box_.size() ) return;
	    if( m_pgrids_per_box_[level].empty() ) return;
	    if( m_pgrids_per_box_[Lp].empty() ) return;

	    if( m_parent_replicas_.size() < (size_t)level+1 )
	        m_parent_replicas_.resize( (size_t)level+1 );
	    if( !m_parent_replicas_[level].empty() ){
	        LOGERR("broadcast_parents_to_child_owners: replicas already exist at level=%u (missed release?)", level);
	        throw std::runtime_error("broadcast_parents_to_child_owners: replicas not released");
	    }

	    const int my_rank = MUSIC::mpi::rank();
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;

	    // For each parent p, build the set of child-owner ranks (excluding the
	    // parent's own owner, since that rank already has the data locally).
	    const size_t Np = m_pgrids_per_box_[Lp].size();
	    std::vector< std::set<int> > child_owners_per_parent( Np );
	    for( size_t b=0; b<m_pgrids_per_box_[level].size(); ++b ){
	        const size_t p = m_parent_idx_per_box_[level][b];
	        if( p >= Np ) continue;  // single-parent fallback returns 0; safe
	        const int co = m_pbox_owner_[level][b];
	        const int po = m_pbox_owner_[Lp][p];
	        if( co != po ) child_owners_per_parent[p].insert(co);
	    }

	    std::vector<MPI_Request> reqs;
	    reqs.reserve(32);

	    for( size_t p=0; p<Np; ++p ){
	        const auto & co_set = child_owners_per_parent[p];
	        if( co_set.empty() ) continue;
	        const int po = m_pbox_owner_[Lp][p];
	        const int tag = 1900 + (int)level*1000 + (int)p;
	        const size_t cnt = (size_t)(m_nx_per_box_[Lp][p] + 2*m_nbnd)
	                         * (size_t)(m_ny_per_box_[Lp][p] + 2*m_nbnd)
	                         * (size_t)(m_nz_per_box_[Lp][p] + 2*m_nbnd);

	        if( my_rank == po ){
	            MeshvarBnd<T>* parent = m_pgrids_per_box_[Lp][p];
	            if( !parent ){
	                LOGERR("broadcast_parents_to_child_owners: parent owner rk=%d missing mesh L=%u p=%zu",
	                       my_rank, Lp, p);
	                throw std::runtime_error("broadcast_parents_to_child_owners: parent NULL on owner");
	            }
	            for( int dest : co_set ){
	                reqs.emplace_back();
	                MPI_Isend( parent->get_ptr(), (int)cnt, dtype, dest, tag,
	                           MUSIC::mpi::world(), &reqs.back() );
	            }
	        } else if( co_set.count(my_rank) ){
	            MeshvarBnd<T>* rep = new MeshvarBnd<T>(
	                m_nbnd,
	                m_nx_per_box_[Lp][p], m_ny_per_box_[Lp][p], m_nz_per_box_[Lp][p],
	                (size_t)m_oxrel_per_box_[Lp][p],
	                (size_t)m_oyrel_per_box_[Lp][p],
	                (size_t)m_ozrel_per_box_[Lp][p] );
	            m_parent_replicas_[level][p] = rep;
	            reqs.emplace_back();
	            MPI_Irecv( rep->get_ptr(), (int)cnt, dtype, po, tag,
	                       MUSIC::mpi::world(), &reqs.back() );
	        }
	    }

	    if( !reqs.empty() )
	        MPI_Waitall( (int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE );
#else
	    (void)level;
#endif
	}

	//! Phase G.2b B.2b.1: reverse of broadcast_parents_to_child_owners.
	//! Each child-owner ships its (potentially modified) replica back to
	//! the parent owner via MPI_Isend; the parent owner MPI_Irecvs into a
	//! staging buffer (one per sender) and ADDS the received data into the
	//! authoritative parent mesh. Used after per-box ops that contribute to
	//! the parent (restrict, FAS source update). Replicas are then freed.
	//!
	//! Tag layout: 2100 + level*1000 + parent_idx (distinct from broadcast).
	//!
	//! The accumulation order across child-owners is deterministic by sender
	//! rank ascending. For exact bit-reproducibility across runs, all
	//! contributors must produce the same floating-point bits in each cell
	//! they touch — caller guarantees this by gating each per-box op to its
	//! owner.
	void accumulate_children_to_parents( unsigned level )
	{
#ifdef USE_MPI
	    if( MUSIC::mpi::size() <= 1 ) return;
	    if( level == 0 ) return;
	    if( level >= m_pgrids_per_box_.size() ) return;
	    const unsigned Lp = level - 1;
	    if( Lp >= m_pgrids_per_box_.size() ) return;
	    if( level >= m_parent_replicas_.size() ) return;

	    const int my_rank = MUSIC::mpi::rank();
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;

	    const size_t Np = m_pgrids_per_box_[Lp].size();
	    std::vector< std::set<int> > child_owners_per_parent( Np );
	    for( size_t b=0; b<m_pgrids_per_box_[level].size(); ++b ){
	        const size_t p = m_parent_idx_per_box_[level][b];
	        if( p >= Np ) continue;
	        const int co = m_pbox_owner_[level][b];
	        const int po = m_pbox_owner_[Lp][p];
	        if( co != po ) child_owners_per_parent[p].insert(co);
	    }

	    // Parent-owner side: receive into a per-sender staging buffer.
	    // child-owner side: send the replica buffer.
	    std::vector< std::vector<T> > recv_bufs;  // owned by parent_owner
	    struct RecvSlot { size_t parent_idx; int source_rank; size_t cnt; size_t buf_idx; };
	    std::vector<RecvSlot> recv_slots;
	    std::vector<MPI_Request> reqs;
	    reqs.reserve(32);
	    recv_bufs.reserve(32);
	    recv_slots.reserve(32);

	    for( size_t p=0; p<Np; ++p ){
	        const auto & co_set = child_owners_per_parent[p];
	        if( co_set.empty() ) continue;
	        const int po = m_pbox_owner_[Lp][p];
	        const int tag_base = 2100 + (int)level*1000 + (int)p;
	        const size_t cnt = (size_t)(m_nx_per_box_[Lp][p] + 2*m_nbnd)
	                         * (size_t)(m_ny_per_box_[Lp][p] + 2*m_nbnd)
	                         * (size_t)(m_nz_per_box_[Lp][p] + 2*m_nbnd);

	        if( my_rank == po ){
	            // Receive once per distinct child-owner rank.
	            for( int src : co_set ){
	                const int tag = tag_base;  // single tag; source rank disambiguates
	                recv_bufs.emplace_back( cnt, T(0) );
	                recv_slots.push_back( RecvSlot{ p, src, cnt, recv_bufs.size()-1 } );
	                reqs.emplace_back();
	                MPI_Irecv( recv_bufs.back().data(), (int)cnt, dtype, src, tag,
	                           MUSIC::mpi::world(), &reqs.back() );
	            }
	        } else if( co_set.count(my_rank) ){
	            auto it = m_parent_replicas_[level].find(p);
	            if( it == m_parent_replicas_[level].end() || it->second == NULL ){
	                LOGERR("accumulate_children_to_parents: rk=%d missing replica L=%u p=%zu",
	                       my_rank, level, p);
	                throw std::runtime_error("accumulate_children_to_parents: replica NULL");
	            }
	            const int tag = tag_base;
	            reqs.emplace_back();
	            MPI_Isend( it->second->get_ptr(), (int)cnt, dtype, po, tag,
	                       MUSIC::mpi::world(), &reqs.back() );
	        }
	    }

	    if( !reqs.empty() )
	        MPI_Waitall( (int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE );

	    // Parent owner: sort recv_slots by (parent_idx, source_rank) ascending
	    // for deterministic accumulation order, then add into the parent mesh.
	    if( !recv_slots.empty() ){
	        std::sort( recv_slots.begin(), recv_slots.end(),
	            []( const RecvSlot & a, const RecvSlot & b ){
	                if( a.parent_idx != b.parent_idx ) return a.parent_idx < b.parent_idx;
	                return a.source_rank < b.source_rank;
	            } );
	        for( const auto & s : recv_slots ){
	            MeshvarBnd<T>* parent = m_pgrids_per_box_[Lp][s.parent_idx];
	            if( !parent ){
	                LOGERR("accumulate_children_to_parents: parent rk=%d L=%u p=%zu NULL",
	                       my_rank, Lp, s.parent_idx);
	                throw std::runtime_error("accumulate_children_to_parents: parent NULL");
	            }
	            T * dst = parent->get_ptr();
	            const T * src = recv_bufs[s.buf_idx].data();
	            for( size_t i=0; i<s.cnt; ++i ) dst[i] += src[i];
	        }
	    }

	    release_parent_replicas( level );
#else
	    (void)level;
#endif
	}

	//! Free all replica buffers at `level` without contributing them back to
	//! the parent. Used when (a) the per-box ops were read-only on the parent,
	//! or (b) after accumulate_children_to_parents has consumed them.
	void release_parent_replicas( unsigned level )
	{
	    if( level >= m_parent_replicas_.size() ) return;
	    for( auto & kv : m_parent_replicas_[level] )
	        delete kv.second;
	    m_parent_replicas_[level].clear();
	}

	//! Free replicas at every level; convenience for end-of-V-cycle cleanup.
	void release_all_parent_replicas()
	{
	    for( size_t L=0; L<m_parent_replicas_.size(); ++L )
	        for( auto & kv : m_parent_replicas_[L] )
	            delete kv.second;
	    m_parent_replicas_.clear();
	}

	//! Phase G.2b B.2b.2.2.1.a: broadcast m_pgrids[level] body from
	//! owner_of_box(level, 0) to all other ranks. Intended for single-box
	//! parent levels in twoGrid_multibox_spmd: after only the box owner runs
	//! `twoGrid(level)` serially, non-owners' m_pgrids[level] is stale and
	//! must be refreshed before the SPMD function reads it as `ucs[b]` at
	//! the next-finer level.
	//!
	//! Preconditions: m_pgrids[level] exists on every rank with identical
	//! geometry (nx, ny, nz, nbnd). create_base_hierarchy + add_patch
	//! guarantee this on every rank at construction time; later code paths
	//! that resize m_pgrids (Phase E.2.1b slab flip) must restore the full
	//! mesh before this primitive runs. Throws on mismatch.
	//!
	//! np<=1: no-op. Must be called OUTSIDE phase_scope (workers must be in
	//! the function body, not parked in worker_pump).
	void broadcast_union_mesh_from_owner( unsigned level )
	{
#ifdef USE_MPI
	    if( MUSIC::mpi::size() <= 1 ) return;
	    if( level >= m_pgrids.size() ) return;

	    MeshvarBnd<T>* um = m_pgrids[level];
	    if( !um ){
	        LOGERR("broadcast_union_mesh_from_owner: m_pgrids[%u] NULL on rk=%d",
	               level, MUSIC::mpi::rank());
	        throw std::runtime_error("broadcast_union_mesh_from_owner: union mesh NULL");
	    }

	    const int owner = owner_of_box( level, 0 );
	    const int my_rank = MUSIC::mpi::rank();
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;

	    const int nbnd = um->m_nbnd;
	    const size_t nx = um->size(0) + 2*nbnd;
	    const size_t ny = um->size(1) + 2*nbnd;
	    const size_t nz = um->size(2) + 2*nbnd;
	    const size_t cnt = nx * ny * nz;

	    // Sanity: broadcast the owner's geometry as 4 ints and have everyone
	    // confirm it matches their local mesh. Cheap insurance against
	    // silent corruption when callers violate the equal-geometry invariant.
	    int hdr[4];
	    if( my_rank == owner ){
	        hdr[0] = (int)nbnd;
	        hdr[1] = (int)nx;
	        hdr[2] = (int)ny;
	        hdr[3] = (int)nz;
	    }
	    MPI_Bcast( hdr, 4, MPI_INT, owner, MUSIC::mpi::world() );
	    if( hdr[0] != (int)nbnd || hdr[1] != (int)nx ||
	        hdr[2] != (int)ny || hdr[3] != (int)nz ){
	        LOGERR("broadcast_union_mesh_from_owner: geometry mismatch rk=%d L=%u "
	               "owner(nbnd,nx,ny,nz)=(%d,%d,%d,%d) mine=(%d,%zu,%zu,%zu)",
	               my_rank, level, hdr[0], hdr[1], hdr[2], hdr[3],
	               nbnd, nx, ny, nz);
	        throw std::runtime_error("broadcast_union_mesh_from_owner: geometry mismatch");
	    }

	    MPI_Bcast( um->get_ptr(), (int)cnt, dtype, owner, MUSIC::mpi::world() );
#else
	    (void)level;
#endif
	}

	//! Phase G.2b B.2b.2.2.1.d: consolidate per-child write contributions on a
	//! UNION mesh (m_pgrids[child_level-1]) to the parent box owner.
	//!
	//! Context: in twoGrid_multibox_spmd at a single-box-parent level, each
	//! child's restrict and FAS source update writes into the parent's UNION
	//! mesh (since num_boxes(child_level-1) == 1, parent == union). At np>1
	//! the per-box loop runs each child write on its OWN rank, so different
	//! ranks see different child contributions on their LOCAL union mesh. The
	//! parent box owner then runs the serial twoGrid(child_level-1) recurse;
	//! it needs every child's contribution gathered into its own local union
	//! mesh first.
	//!
	//! Semantics: for each child b at child_level, the OWNER of b packs the
	//! parent-footprint interior cells of m_pgrids[child_level-1] (origin =
	//! (m_oxrel_per_box_, m_oyrel_per_box_, m_ozrel_per_box_)[child_level][b],
	//! extent = (nx/2, ny/2, nz/2)) and sends them to the parent box owner.
	//! The parent box owner OVERWRITES those interior cells in its own copy
	//! of m_pgrids[child_level-1] with the received values. Child-footprints
	//! are disjoint by D.1 construction, so OVERWRITE is correct (each cell
	//! belongs to at most one child).
	//!
	//! No-ops when:
	//!   - MUSIC::mpi::size() <= 1
	//!   - child_level == 0 (no parent)
	//!   - num_boxes(child_level-1) != 1 (multi-box parent uses the
	//!     accumulate_children_to_parents replica machinery instead)
	//!   - owner_of_box(child_level, b) == parent_owner (already in place)
	//!
	//! Must be called OUTSIDE phase_scope (workers must be in the function,
	//! not parked in worker_pump).
	void consolidate_child_writes_to_parent_owner( unsigned child_level )
	{
#ifdef USE_MPI
	    if( MUSIC::mpi::size() <= 1 ) return;
	    if( child_level == 0 ) return;
	    // Guard the PARENT union (the only mesh this function touches). Workers
	    // in the 2LPT path carry m_pgrids sized exactly to child_level after
	    // acquire_parent_union_scratch installs the parent scratch at
	    // child_level-1; guarding on child_level would early-return the child
	    // owner and deadlock the parent owner's Recv.
	    if( (size_t)(child_level - 1) >= m_pgrids.size() ) return;

	    const size_t nb = num_boxes(child_level);
	    if( nb == 0 ) return;
	    if( num_boxes(child_level - 1) != 1 ) return;

	    MeshvarBnd<T>* parent_um = m_pgrids[child_level - 1];
	    if( !parent_um ){
	        LOGERR("consolidate_child_writes_to_parent_owner: parent union NULL at L=%u rk=%d",
	               child_level - 1, MUSIC::mpi::rank());
	        throw std::runtime_error("consolidate_child_writes_to_parent_owner: parent union NULL");
	    }

	    const int parent_owner = owner_of_box(child_level - 1, 0);
	    const int my_rank = MUSIC::mpi::rank();
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;

	    for( size_t b = 0; b < nb; ++b ){
	        const int owner_b = owner_of_box(child_level, b);
	        if( owner_b == parent_owner ) continue;

	        const int oxc = m_oxrel_per_box_[child_level][b];
	        const int oyc = m_oyrel_per_box_[child_level][b];
	        const int ozc = m_ozrel_per_box_[child_level][b];
	        const int sxc = (int)m_nx_per_box_[child_level][b] / 2;
	        const int syc = (int)m_ny_per_box_[child_level][b] / 2;
	        const int szc = (int)m_nz_per_box_[child_level][b] / 2;
	        if( sxc <= 0 || syc <= 0 || szc <= 0 ) continue;

	        const size_t cnt = (size_t)sxc * (size_t)syc * (size_t)szc;
	        const int tag = 17000 + (int)(b & 0x7FFF);

	        if( my_rank == owner_b ){
	            std::vector<T> buf(cnt);
	            for( int ix = 0; ix < sxc; ++ix )
	                for( int iy = 0; iy < syc; ++iy )
	                    for( int iz = 0; iz < szc; ++iz )
	                        buf[((size_t)ix * syc + iy) * szc + iz] =
	                            (*parent_um)(oxc + ix, oyc + iy, ozc + iz);
	            MPI_Send(buf.data(), (int)cnt, dtype, parent_owner, tag, MUSIC::mpi::world());
	        } else if( my_rank == parent_owner ){
	            std::vector<T> buf(cnt);
	            MPI_Recv(buf.data(), (int)cnt, dtype, owner_b, tag,
	                     MUSIC::mpi::world(), MPI_STATUS_IGNORE);
	            for( int ix = 0; ix < sxc; ++ix )
	                for( int iy = 0; iy < syc; ++iy )
	                    for( int iz = 0; iz < szc; ++iz )
	                        (*parent_um)(oxc + ix, oyc + iy, ozc + iz) =
	                            buf[((size_t)ix * syc + iy) * szc + iz];
	        }
	    }
#else
	    (void)child_level;
#endif
	}

	//! Task #135 Option B: install a transient zeroed full-union scratch into
	//! m_pgrids[child_level-1] on worker child-owners so the single-box-parent
	//! restrict in compute_2LPT_source_distributed has a valid target. The
	//! restrict writes V(ox+i,...) with ox = vf.offset(0) into the union, then
	//! consolidate_child_writes_to_parent_owner ships those footprints to the
	//! parent owner. Workers carry a size-0 union m_pgrids for the 2LPT
	//! hierarchy, so we grow m_pgrids to child_level (filling NULL) and record
	//! the prior size for exact restore in release_parent_union_scratch.
	//!
	//! Collective: parent_owner broadcasts the union geometry to all ranks.
	//! No-ops when np<=1, child_level==0, or parent is multi-box (which uses
	//! the broadcast_parents_to_child_owners replica machinery instead).
	//! Must be called OUTSIDE phase_scope (all ranks call).
	void acquire_parent_union_scratch( unsigned child_level )
	{
#ifdef USE_MPI
	    if( MUSIC::mpi::size() <= 1 ) return;
	    if( child_level == 0 ) return;
	    const unsigned Lp = child_level - 1;
	    if( num_boxes( Lp ) != 1 ) return;

	    const int parent_owner = owner_of_box( Lp, 0 );
	    const int my_rank = MUSIC::mpi::rank();

	    int hdr[7];
	    if( my_rank == parent_owner ){
	        if( Lp >= m_pgrids.size() || m_pgrids[Lp] == NULL ){
	            LOGERR("acquire_parent_union_scratch: parent_owner rk=%d missing union L=%u",
	                   my_rank, Lp);
	            throw std::runtime_error("acquire_parent_union_scratch: parent union NULL on owner");
	        }
	        MeshvarBnd<T>* um = m_pgrids[Lp];
	        hdr[0] = um->m_nbnd;
	        hdr[1] = (int)um->size(0);
	        hdr[2] = (int)um->size(1);
	        hdr[3] = (int)um->size(2);
	        hdr[4] = um->offset(0);
	        hdr[5] = um->offset(1);
	        hdr[6] = um->offset(2);
	    }
	    MPI_Bcast( hdr, 7, MPI_INT, parent_owner, MUSIC::mpi::world() );

	    if( my_rank != parent_owner &&
	        m_union_scratch_.find( Lp ) == m_union_scratch_.end() ){
	        const size_t prevsz = m_pgrids.size();
	        if( m_pgrids.size() <= (size_t)Lp )
	            m_pgrids.resize( Lp + 1, (MeshvarBnd<T>*)NULL );
	        if( m_pgrids[Lp] == NULL ){
	            MeshvarBnd<T>* scratch = new MeshvarBnd<T>(
	                hdr[0], (size_t)hdr[1], (size_t)hdr[2], (size_t)hdr[3] );
	            scratch->offset(0) = hdr[4];
	            scratch->offset(1) = hdr[5];
	            scratch->offset(2) = hdr[6];
	            scratch->zero();
	            m_pgrids[Lp] = scratch;
	            m_union_scratch_[Lp] = scratch;
	            m_union_scratch_prevsz_[Lp] = prevsz;
	        }
	    }
#else
	    (void)child_level;
#endif
	}

	//! Task #135 Option B: free a scratch installed by acquire_parent_union_scratch
	//! and restore m_pgrids to its prior size on the worker. No-op when no
	//! scratch is installed at child_level-1 (e.g. on the parent owner).
	void release_parent_union_scratch( unsigned child_level )
	{
#ifdef USE_MPI
	    if( child_level == 0 ) return;
	    const unsigned Lp = child_level - 1;
	    auto it = m_union_scratch_.find( Lp );
	    if( it == m_union_scratch_.end() ) return;
	    MeshvarBnd<T>* scratch = it->second;
	    if( (size_t)Lp < m_pgrids.size() && m_pgrids[Lp] == scratch )
	        m_pgrids[Lp] = NULL;
	    delete scratch;
	    m_union_scratch_.erase( it );
	    auto sit = m_union_scratch_prevsz_.find( Lp );
	    if( sit != m_union_scratch_prevsz_.end() ){
	        const size_t prevsz = sit->second;
	        if( m_pgrids.size() > prevsz )
	            m_pgrids.resize( prevsz, (MeshvarBnd<T>*)NULL );
	        m_union_scratch_prevsz_.erase( sit );
	    }
#else
	    (void)child_level;
#endif
	}

	//! Free any lingering union scratch (exception-safety drain for deallocate).
	//! NULLs the m_pgrids slot first so the deallocate m_pgrids loop never
	//! double-frees the scratch pointer.
	void release_all_parent_union_scratch()
	{
	    for( auto & kv : m_union_scratch_ ){
	        const unsigned Lp = kv.first;
	        if( (size_t)Lp < m_pgrids.size() && m_pgrids[Lp] == kv.second )
	            m_pgrids[Lp] = NULL;
	        delete kv.second;
	    }
	    m_union_scratch_.clear();
	    m_union_scratch_prevsz_.clear();
	}

	//! Task #135: OVERWRITE-semantics consolidation of per-child restrict writes
	//! into a MULTI-box parent. The overwrite analog of
	//! accumulate_children_to_parents (which does dst += src on the full replica)
	//! and the multi-box analog of consolidate_child_writes_to_parent_owner
	//! (which targets the single-box union mesh).
	//!
	//! Context: distributed compute_2LPT_source restricts each fine source box
	//! into its parent with mg_straight::restrict semantics — an OVERWRITE (=),
	//! not an ADD. Each child owner has written the restricted 8-cell averages
	//! into its parent REPLICA (m_parent_replicas_[child_level][p]) at the
	//! child's parent-relative offset. This primitive ships only that disjoint
	//! coarse footprint to the parent box owner, which OVERWRITES the matching
	//! cells of m_pgrids_per_box_[child_level-1][p]. Child footprints are
	//! disjoint by D.1 construction, so OVERWRITE is correct and the parent's
	//! own (non-covered) cells keep their stage-1 FD values.
	//!
	//! Must be paired with a prior broadcast_parents_to_child_owners(child_level)
	//! (so replicas exist) and the per-box restrict into parent_for_box. Frees
	//! the replicas on exit (like accumulate_children_to_parents).
	//!
	//! No-ops when:
	//!   - MUSIC::mpi::size() <= 1
	//!   - child_level == 0
	//!   - num_boxes(child_level-1) <= 1 (single-box parent → use
	//!     consolidate_child_writes_to_parent_owner instead)
	//!   - owner_of_box(child_level, b) == parent box owner (local parent was
	//!     written directly by restrict; nothing to ship)
	//!
	//! Must be called OUTSIDE phase_scope (all ranks call). SPMD.
	void overwrite_children_to_parents( unsigned child_level )
	{
#ifdef USE_MPI
	    if( MUSIC::mpi::size() <= 1 ){ release_parent_replicas( child_level ); return; }
	    if( child_level == 0 ) return;
	    if( child_level >= m_pgrids_per_box_.size() ) return;
	    const unsigned Lp = child_level - 1;
	    if( Lp >= m_pgrids_per_box_.size() ) return;
	    if( num_boxes( Lp ) <= 1 ) return;        // single-box parent: use consolidate

	    const size_t nb = num_boxes( child_level );
	    if( nb == 0 ) return;
	    const int my_rank = MUSIC::mpi::rank();
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;

	    std::vector< std::vector<T> > send_bufs;   // keep alive across Waitall
	    send_bufs.reserve( 32 );
	    std::vector<MPI_Request> reqs;
	    reqs.reserve( 32 );
	    struct RecvSlot { size_t parent_idx; int oxc, oyc, ozc, sxc, syc, szc;
	                      size_t cnt, buf_idx; };
	    std::vector<RecvSlot> recv_slots;
	    std::vector< std::vector<T> > recv_bufs;
	    recv_slots.reserve( 32 );
	    recv_bufs.reserve( 32 );

	    for( size_t b = 0; b < nb; ++b ){
	        const size_t p = m_parent_idx_per_box_[child_level][b];
	        if( p >= m_pgrids_per_box_[Lp].size() ) continue;
	        const int owner_b = m_pbox_owner_[child_level][b];
	        const int parent_owner = m_pbox_owner_[Lp][p];
	        if( owner_b == parent_owner ) continue;   // local parent already written

	        const int oxc = m_oxrel_per_box_[child_level][b];
	        const int oyc = m_oyrel_per_box_[child_level][b];
	        const int ozc = m_ozrel_per_box_[child_level][b];
	        const int sxc = (int)m_nx_per_box_[child_level][b] / 2;
	        const int syc = (int)m_ny_per_box_[child_level][b] / 2;
	        const int szc = (int)m_nz_per_box_[child_level][b] / 2;
	        if( sxc <= 0 || syc <= 0 || szc <= 0 ) continue;
	        const size_t cnt = (size_t)sxc * (size_t)syc * (size_t)szc;
	        const int tag = 17500 + (int)(b & 0x7FFF);

	        if( my_rank == owner_b ){
	            auto it = m_parent_replicas_[child_level].find( p );
	            if( it == m_parent_replicas_[child_level].end() || it->second == NULL ){
	                LOGERR("overwrite_children_to_parents: rk=%d missing replica L=%u p=%zu",
	                       my_rank, child_level, p);
	                throw std::runtime_error("overwrite_children_to_parents: replica NULL");
	            }
	            MeshvarBnd<T>* rep = it->second;
	            send_bufs.emplace_back( cnt );
	            std::vector<T> & buf = send_bufs.back();
	            for( int ix = 0; ix < sxc; ++ix )
	                for( int iy = 0; iy < syc; ++iy )
	                    for( int iz = 0; iz < szc; ++iz )
	                        buf[((size_t)ix * syc + iy) * szc + iz] =
	                            (*rep)(oxc + ix, oyc + iy, ozc + iz);
	            reqs.emplace_back();
	            MPI_Isend( buf.data(), (int)cnt, dtype, parent_owner, tag,
	                       MUSIC::mpi::world(), &reqs.back() );
	        } else if( my_rank == parent_owner ){
	            recv_bufs.emplace_back( cnt );
	            recv_slots.push_back( RecvSlot{ p, oxc, oyc, ozc, sxc, syc, szc,
	                                            cnt, recv_bufs.size()-1 } );
	            reqs.emplace_back();
	            MPI_Irecv( recv_bufs.back().data(), (int)cnt, dtype, owner_b, tag,
	                       MUSIC::mpi::world(), &reqs.back() );
	        }
	    }

	    if( !reqs.empty() )
	        MPI_Waitall( (int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE );

	    for( const auto & s : recv_slots ){
	        MeshvarBnd<T>* parent = m_pgrids_per_box_[Lp][s.parent_idx];
	        if( !parent ){
	            LOGERR("overwrite_children_to_parents: parent rk=%d L=%u p=%zu NULL",
	                   my_rank, Lp, s.parent_idx);
	            throw std::runtime_error("overwrite_children_to_parents: parent NULL");
	        }
	        const T * src = recv_bufs[s.buf_idx].data();
	        for( int ix = 0; ix < s.sxc; ++ix )
	            for( int iy = 0; iy < s.syc; ++iy )
	                for( int iz = 0; iz < s.szc; ++iz )
	                    (*parent)(s.oxc + ix, s.oyc + iy, s.ozc + iz) =
	                        src[((size_t)ix * s.syc + iy) * s.szc + iz];
	    }

	    release_parent_replicas( child_level );
#else
	    (void)child_level;
#endif
	}

	//! Phase G.2b B.2b.1: standalone smoke test for the cross-rank parent
	//! replica primitives. Constructs a small 2-level multibox hierarchy
	//! (parent at L=0 with `n_children` children at L=1), populates it,
	//! exercises broadcast/accumulate, and verifies bit-identical replicas
	//! and correct ADD accumulation. Returns true on success.
	//!
	//! At np=1 the test is trivially correct: no cross-owner pairs exist,
	//! broadcast/accumulate are no-ops, parent stays at initial pattern.
	//!
	//! At np>=2 with default owner=b%np: parent owner = 0, child owners
	//! ∈ {0..n_children-1} % np. Distinct cross-owners receive replicas.
	//!
	//! Must be called outside phase_scope. SPMD (all ranks call).
	static bool test_b2b1_replica_roundtrip( unsigned n_children = 2,
	                                          int box_nx = 8,
	                                          int verbose = 0 )
	{
#ifdef USE_MPI
	    const int my_rank = MUSIC::mpi::rank();
	    const int np      = MUSIC::mpi::size();

	    GridHierarchy<T> gh( /*nbnd=*/2 );
	    // Per populate_per_box_meshes contract: m_pgrids is independent
	    // bookkeeping; we don't allocate union meshes here, only per-box.
	    std::vector< std::vector<LevelBox> > lb(2);

	    // L=0: one parent box, size = (n_children*box_nx) along x, box_nx along y/z.
	    {
	        LevelBox p{};
	        p.oax = 0; p.oay = 0; p.oaz = 0;
	        p.nx = (unsigned)(n_children * box_nx);
	        p.ny = (unsigned)box_nx;
	        p.nz = (unsigned)box_nx;
	        p.ox = 0; p.oy = 0; p.oz = 0;
	        p.x0 = 0; p.y0 = 0; p.z0 = 0;
	        p.xl = 1; p.yl = 1; p.zl = 1;
	        p.parent_idx = 0;
	        lb[0].push_back(p);
	    }
	    // L=1: n_children children, tiling along x.
	    for( unsigned c=0; c<n_children; ++c ){
	        LevelBox cbx{};
	        cbx.oax = c * (unsigned)box_nx;
	        cbx.oay = 0; cbx.oaz = 0;
	        cbx.nx = (unsigned)box_nx;
	        cbx.ny = (unsigned)box_nx;
	        cbx.nz = (unsigned)box_nx;
	        cbx.ox = (int)(c * box_nx); cbx.oy = 0; cbx.oz = 0;
	        cbx.x0 = 0; cbx.y0 = 0; cbx.z0 = 0;
	        cbx.xl = 1; cbx.yl = 1; cbx.zl = 1;
	        cbx.parent_idx = 0;
	        lb[1].push_back(cbx);
	    }

	    gh.populate_per_box_meshes(lb);

	    const int parent_owner = gh.owner_of_box(0, 0);

	    // Phase 1: parent owner fills parent mesh with a deterministic pattern
	    // (including halo). All other ranks have NULL parent slot.
	    auto pattern = []( int i, int j, int k ) -> T {
	        return (T)(i * 1000.0 + j * 100.0 + k * 10.0 + 1.0);
	    };
	    if( my_rank == parent_owner ){
	        MeshvarBnd<T>* pm = gh.get_grid(0, 0);
	        int nbnd = pm->m_nbnd;
	        int nx = (int)pm->size(0), ny = (int)pm->size(1), nz = (int)pm->size(2);
	        for( int i=-nbnd; i<nx+nbnd; ++i )
	            for( int j=-nbnd; j<ny+nbnd; ++j )
	                for( int k=-nbnd; k<nz+nbnd; ++k )
	                    (*pm)(i,j,k) = pattern(i, j, k);
	    }

	    // Phase 2: broadcast.
	    gh.broadcast_parents_to_child_owners(1);

	    // Phase 3: verify replica byte-match on every cross-rank child owner.
	    int local_fail = 0;
	    {
	        std::set<int> child_owners;
	        for( unsigned c=0; c<n_children; ++c )
	            child_owners.insert( gh.owner_of_box(1, c) );
	        // Determine if my_rank is a cross-rank child owner needing a replica.
	        bool need_replica = (my_rank != parent_owner) && child_owners.count(my_rank);
	        if( need_replica ){
	            MeshvarBnd<T>* rep = NULL;
	            // Find via parent_for_box on any owned child.
	            for( unsigned c=0; c<n_children; ++c ){
	                if( gh.owner_of_box(1, c) == my_rank ){
	                    rep = gh.parent_for_box(1, c);
	                    break;
	                }
	            }
	            if( !rep ){ ++local_fail; LOGERR("B.2b.1 smoke: rk=%d expected replica missing", my_rank); }
	            else {
	                int nbnd = rep->m_nbnd;
	                int nx = (int)rep->size(0), ny = (int)rep->size(1), nz = (int)rep->size(2);
	                for( int i=-nbnd; i<nx+nbnd && !local_fail; ++i )
	                    for( int j=-nbnd; j<ny+nbnd && !local_fail; ++j )
	                        for( int k=-nbnd; k<nz+nbnd && !local_fail; ++k ){
	                            T got = (*rep)(i,j,k);
	                            T exp = pattern(i, j, k);
	                            if( got != exp ){
	                                ++local_fail;
	                                LOGERR("B.2b.1 smoke: rk=%d replica mismatch at (%d,%d,%d): got=%.6e exp=%.6e",
	                                       my_rank, i, j, k, (double)got, (double)exp);
	                            }
	                        }
	            }
	        }
	    }

	    int g_fail_1 = 0;
	    MPI_Allreduce(&local_fail, &g_fail_1, 1, MPI_INT, MPI_SUM, MUSIC::mpi::world());

	    // Phase 4: release. (Avoids accumulate-on-untouched-replica which would
	    // double the parent pattern. Tests broadcast in isolation.)
	    gh.release_parent_replicas(1);

	    // Phase 5: re-broadcast and modify replicas with a known constant.
	    gh.broadcast_parents_to_child_owners(1);

	    const T fill_value = (T)7.0;
	    std::set<int> cross_owners;
	    {
	        std::set<int> child_owners;
	        for( unsigned c=0; c<n_children; ++c )
	            child_owners.insert( gh.owner_of_box(1, c) );
	        for( int co : child_owners )
	            if( co != parent_owner ) cross_owners.insert(co);
	    }
	    if( cross_owners.count(my_rank) ){
	        for( unsigned c=0; c<n_children; ++c ){
	            if( gh.owner_of_box(1, c) != my_rank ) continue;
	            MeshvarBnd<T>* rep = gh.parent_for_box(1, c);
	            if( !rep ) continue;
	            int nbnd = rep->m_nbnd;
	            int nx = (int)rep->size(0), ny = (int)rep->size(1), nz = (int)rep->size(2);
	            for( int i=-nbnd; i<nx+nbnd; ++i )
	                for( int j=-nbnd; j<ny+nbnd; ++j )
	                    for( int k=-nbnd; k<nz+nbnd; ++k )
	                        (*rep)(i,j,k) = fill_value;
	            break;  // same replica for any owned child of this parent
	        }
	    }

	    // Phase 6: accumulate. parent receives fill_value from each distinct
	    // cross-owner once.
	    gh.accumulate_children_to_parents(1);

	    // Phase 7: verify parent on parent_owner: pattern(i,j,k) + n_cross * fill_value.
	    int local_fail_2 = 0;
	    if( my_rank == parent_owner ){
	        MeshvarBnd<T>* pm = gh.get_grid(0, 0);
	        int nbnd = pm->m_nbnd;
	        int nx = (int)pm->size(0), ny = (int)pm->size(1), nz = (int)pm->size(2);
	        T expected_add = (T)((double)cross_owners.size() * (double)fill_value);
	        for( int i=-nbnd; i<nx+nbnd && local_fail_2 < 8; ++i )
	            for( int j=-nbnd; j<ny+nbnd && local_fail_2 < 8; ++j )
	                for( int k=-nbnd; k<nz+nbnd && local_fail_2 < 8; ++k ){
	                    T got = (*pm)(i,j,k);
	                    T exp = pattern(i,j,k) + expected_add;
	                    if( got != exp ){
	                        ++local_fail_2;
	                        LOGERR("B.2b.1 smoke: rk=%d accumulate mismatch at (%d,%d,%d): got=%.6e exp=%.6e (n_cross=%zu fill=%.3f)",
	                               my_rank, i, j, k, (double)got, (double)exp,
	                               cross_owners.size(), (double)fill_value);
	                    }
	                }
	    }
	    int g_fail_2 = 0;
	    MPI_Allreduce(&local_fail_2, &g_fail_2, 1, MPI_INT, MPI_SUM, MUSIC::mpi::world());

	    const bool ok = (g_fail_1 == 0) && (g_fail_2 == 0);
	    if( verbose > 0 && my_rank == 0 ){
	        LOGINFO("B.2b.1 smoke (n_children=%u, np=%d): broadcast %s, accumulate %s [cross_owners=%zu]",
	                n_children, np,
	                (g_fail_1==0 ? "OK" : "FAIL"),
	                (g_fail_2==0 ? "OK" : "FAIL"),
	                cross_owners.size());
	    }
	    return ok;
#else
	    (void)n_children; (void)box_nx; (void)verbose;
	    return true;
#endif
	}

	//! Phase G.2b B.2b.2.2.1.a: standalone smoke test for
	//! broadcast_union_mesh_from_owner. Builds a flat hierarchy of
	//! (lmax+1) single-box union levels via create_base_hierarchy. On
	//! rank 0 (the default owner_of_box(_,0)), fills m_pgrids[level] with
	//! a deterministic pattern (including halo). All other ranks fill
	//! their m_pgrids[level] with a sentinel that does NOT match the
	//! pattern. After broadcast, every rank's m_pgrids[level] must match
	//! the pattern bit-for-bit. Returns true on success.
	//!
	//! At np=1 the broadcast is a no-op; rank 0 already has the pattern,
	//! so the test trivially passes.
	static bool test_b2b221a_broadcast_union( unsigned lmax = 3,
	                                          unsigned target_level = 2,
	                                          int verbose = 0 )
	{
#ifdef USE_MPI
	    const int my_rank = MUSIC::mpi::rank();
	    const int np      = MUSIC::mpi::size();

	    if( target_level > lmax ){
	        if( my_rank == 0 ) LOGERR("B.2b.2.2.1.a smoke: target_level=%u > lmax=%u",
	                                    target_level, lmax);
	        return false;
	    }

	    GridHierarchy<T> gh( /*nbnd=*/2 );
	    gh.create_base_hierarchy( lmax );  // every rank holds m_pgrids[0..lmax]

	    MeshvarBnd<T>* um = gh.get_grid( target_level );
	    if( !um ){
	        LOGERR("B.2b.2.2.1.a smoke: rk=%d missing m_pgrids[%u]", my_rank, target_level);
	        return false;
	    }
	    const int nbnd = um->m_nbnd;
	    const int nx = (int)um->size(0), ny = (int)um->size(1), nz = (int)um->size(2);

	    auto pattern = []( int i, int j, int k ) -> T {
	        return (T)(i * 1000.0 + j * 100.0 + k * 10.0 + 1.0);
	    };
	    const T sentinel = (T)(-999.0);

	    // Owner: pattern. Others: sentinel (must be overwritten by broadcast).
	    const int owner = gh.owner_of_box( target_level, 0 );
	    for( int i=-nbnd; i<nx+nbnd; ++i )
	        for( int j=-nbnd; j<ny+nbnd; ++j )
	            for( int k=-nbnd; k<nz+nbnd; ++k )
	                (*um)(i,j,k) = (my_rank == owner) ? pattern(i,j,k) : sentinel;

	    gh.broadcast_union_mesh_from_owner( target_level );

	    int local_fail = 0;
	    for( int i=-nbnd; i<nx+nbnd && local_fail < 8; ++i )
	        for( int j=-nbnd; j<ny+nbnd && local_fail < 8; ++j )
	            for( int k=-nbnd; k<nz+nbnd && local_fail < 8; ++k ){
	                T got = (*um)(i,j,k);
	                T exp = pattern(i,j,k);
	                if( got != exp ){
	                    ++local_fail;
	                    LOGERR("B.2b.2.2.1.a smoke: rk=%d mismatch at (%d,%d,%d): got=%.6e exp=%.6e",
	                           my_rank, i, j, k, (double)got, (double)exp);
	                }
	            }
	    int g_fail = 0;
	    MPI_Allreduce(&local_fail, &g_fail, 1, MPI_INT, MPI_SUM, MUSIC::mpi::world());

	    const bool ok = (g_fail == 0);
	    if( verbose > 0 && my_rank == 0 ){
	        LOGINFO("B.2b.2.2.1.a smoke (lmax=%u target=%u np=%d): broadcast %s [owner=%d, mesh=(%d,%d,%d) nbnd=%d]",
	                lmax, target_level, np,
	                (ok ? "OK" : "FAIL"),
	                owner, nx, ny, nz, nbnd);
	    }
	    return ok;
#else
	    (void)lmax; (void)target_level; (void)verbose;
	    return true;
#endif
	}

	//---------------------------------------------------------------------

	//! Phase D.3.1: per-cluster mean/sigma over the interior (no halo) of
	//! each per-box mesh, plus the corresponding union sub-window mean/sigma
	//! for cross-check. Logged only at multi-box levels. Diagnostic only.
	void log_per_box_stats( const char * tag ) const
	{
	    for( size_t L=0; L<m_pgrids_per_box_.size(); ++L )
	    {
	        if( m_pgrids_per_box_[L].size() <= 1 ) continue;
	        for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b )
	        {
	            const MeshvarBnd<T> * bm = m_pgrids_per_box_[L][b];
	            if( !bm ) continue; // E.1: stats only computed by owner rank
	            int bnx=(int)bm->size(0), bny=(int)bm->size(1), bnz=(int)bm->size(2);
	            double sum=0.0, sq=0.0;
	            size_t ncell=0;
	            for( int i=0; i<bnx; ++i )
	                for( int j=0; j<bny; ++j )
	                    for( int k=0; k<bnz; ++k ){
	                        double v = (double)(*bm)(i,j,k);
	                        sum += v; sq += v*v; ++ncell;
	                    }
	            double mean = ncell ? sum/ncell : 0.0;
	            double var  = ncell ? (sq/ncell - mean*mean) : 0.0;
	            double sd   = var>0 ? std::sqrt(var) : 0.0;
	            LOGINFO("%s per-box stats L=%zu b=%zu n=(%d,%d,%d) mean=%.6e sigma=%.6e",
	                    tag, L, b, bnx, bny, bnz, mean, sd);
	        }
	    }
	}
	//---------------------------------------------------------------------

	//-------- Phase G.1 per-cluster z-slab API ---------------------------
	bool has_zslabs() const { return m_zslab_enabled_; }
	int  zslab_halo_w() const { return m_zslab_halo_w_; }
	size_t num_zslabs( unsigned ilevel ) const {
	    if( !m_zslab_enabled_ || ilevel >= m_zslab_layouts_.size() ) return 0;
	    return m_zslab_layouts_[ilevel].size();
	}
	const MUSIC::zoom_slab::ZoomSlabLayout & zslab_layout( unsigned ilevel, size_t b ) const {
	    return m_zslab_layouts_.at(ilevel).at(b);
	}
	T*       zslab_data( unsigned ilevel, size_t b ){
	    return m_zslab_data_.at(ilevel).at(b).data();
	}
	const T* zslab_data( unsigned ilevel, size_t b ) const {
	    return m_zslab_data_.at(ilevel).at(b).data();
	}

	//! Phase G.1: free per-cluster z-slab layouts and data buffers.
	void deallocate_per_box_zslabs(){
	    m_zslab_layouts_.clear();
	    m_zslab_data_.clear();
	    m_zslab_enabled_ = false;
	    m_zslab_halo_w_  = 0;
	}

	//! Phase G.1: allocate per-cluster z-slab layouts + local-with-halo
	//! data buffers for every (L, b) in m_pgrids_per_box_. Uses
	//! MPI_COMM_WORLD as the per-cluster sub-comm (every rank participates
	//! in every cluster). G.2+ will introduce true per-cluster sub-comms;
	//! the layout footprint here will not change.
	//!
	//! Requires populate_per_box_meshes() to have run (so per-box geometry
	//! vectors are filled). halo_w is the z-direction halo cell count; for
	//! G.1 it should match m_nbnd so the z-slab footprint mirrors the
	//! per-box MeshvarBnd footprint along z.
	void populate_per_box_zslabs( int halo_w ){
	    deallocate_per_box_zslabs();
	    const size_t Lmax = m_pgrids_per_box_.size();
	    if( Lmax == 0 ) return;
	    if( halo_w < 0 ) halo_w = 0;
	    m_zslab_halo_w_ = halo_w;
	    m_zslab_layouts_.assign(Lmax, std::vector<MUSIC::zoom_slab::ZoomSlabLayout>());
	    m_zslab_data_.assign   (Lmax, std::vector<std::vector<T> >());

#ifdef USE_MPI
	    MPI_Comm sub = MUSIC::mpi::world();
#else
	    MPI_Comm sub = 0;
#endif
	    const int sub_size = MUSIC::mpi::size();

	    size_t multi_levels = 0;
	    size_t total_local_cells = 0;
	    for( size_t L = 0; L < Lmax; ++L ){
	        const size_t nb = m_pgrids_per_box_[L].size();
	        for( size_t b = 0; b < nb; ++b ){
	            const int cnx = (int)m_nx_per_box_[L][b];
	            const int cny = (int)m_ny_per_box_[L][b];
	            const int cnz = (int)m_nz_per_box_[L][b];
	            // Skip levels where the slab decomposition would leave a rank
	            // empty (low levels with cnz < sub_size, or any cluster too
	            // small to halo-exchange). Caller can still inspect num_zslabs
	            // and zslab_layout per level to detect skipped levels.
	            if( sub_size > cnz || cnz < halo_w * sub_size ){
	                continue;
	            }
	            MUSIC::zoom_slab::ZoomSlabLayout lyt = MUSIC::zoom_slab::make_layout(
	                sub, /*cluster_id=*/b, /*level=*/(unsigned)L,
	                m_xoffabs_per_box_[L][b], m_yoffabs_per_box_[L][b], m_zoffabs_per_box_[L][b],
	                cnx, cny, cnz, halo_w);
	            const size_t cells = MUSIC::zoom_slab::local_with_halo_size(lyt);
	            m_zslab_layouts_[L].push_back(lyt);
	            m_zslab_data_[L].emplace_back(cells, T(0));
	            total_local_cells += cells;
	        }
	        if( nb > 1 ) ++multi_levels;
	    }
	    m_zslab_enabled_ = true;

	    if( multi_levels > 0 ){
	        const double mb_local = (double)(total_local_cells * sizeof(T)) / (1024.0*1024.0);
	        LOGINFO("G.1 populate_per_box_zslabs: enabled (halo_w=%d sub_size=%d) — local z-slab footprint = %.2f MB across %zu multi-box level(s)",
	                halo_w, sub_size, mb_local, multi_levels);
	    }
	}

	//! Phase G.1: owner's per-box MeshvarBnd interior -> all-ranks' z-slab
	//! interior (then halo-exchange in z). Edge halos in z are zero-filled
	//! per MUSIC::zoom_slab::halo_exchange policy (clusters are bounded).
	//!
	//! Pre: populate_per_box_zslabs() + per-box mesh contents are
	//! authoritative on the owner rank (e.g., after sync_per_box_from_union).
	//! Post: every rank's z-slab data buffer holds the correct interior
	//! + halo for its z-strip of cluster b at level L.
	void sync_zslab_from_per_box(){
	    if( !m_zslab_enabled_ ) return;
	    const int my_rank = MUSIC::mpi::rank();
	    for( size_t L = 0; L < m_zslab_layouts_.size(); ++L ){
	        for( size_t b = 0; b < m_zslab_layouts_[L].size(); ++b ){
	            const MUSIC::zoom_slab::ZoomSlabLayout & lyt = m_zslab_layouts_[L][b];
	            const int owner = m_pbox_owner_[L][b];

	            // Owner packs its per-box mesh interior into a contiguous
	            // (i, j, k=0..cnz) row-major buffer; non-owners pass NULL.
	            std::vector<T> full_pack;
	            const std::size_t full_cells = MUSIC::zoom_slab::cluster_full_size(lyt);
	            if( my_rank == owner ){
	                const MeshvarBnd<T> * bm = m_pgrids_per_box_[L][b];
	                if( !bm ) throw std::runtime_error("G.1 sync_zslab_from_per_box: owner mesh NULL");
	                full_pack.assign(full_cells, T(0));
	                const int bnx = (int)bm->size(0);
	                const int bny = (int)bm->size(1);
	                const int bnz = (int)bm->size(2);
	                for( int i=0; i<bnx; ++i )
	                    for( int j=0; j<bny; ++j )
	                        for( int k=0; k<bnz; ++k ){
	                            full_pack[((std::size_t)i*(std::size_t)bny + (std::size_t)j)
	                                      * (std::size_t)bnz + (std::size_t)k]
	                                = (*bm)(i,j,k);
	                        }
	            }

	            // Scatter interior into each rank's local-with-halo buffer
	            // (interior region at k in [halo_w, halo_w+local_nz)).
	            const int hw = lyt.halo_w;
	            const int loc_nz = MUSIC::zoom_slab::local_nz(lyt);
	            std::vector<T> interior((std::size_t)lyt.cluster_nx
	                                  * (std::size_t)lyt.cluster_ny
	                                  * (std::size_t)loc_nz, T(0));
	            MUSIC::zoom_slab::scatter_from(lyt, owner,
	                                    my_rank == owner ? full_pack.data() : (const T*)NULL,
	                                    interior.data());

	            // Copy interior into z-slab buffer's interior region.
	            T* buf = m_zslab_data_[L][b].data();
	            // Zero the halo first; halo exchange fills the interior-of-
	            // neighbours portion. Edges stay zero.
	            std::fill(m_zslab_data_[L][b].begin(), m_zslab_data_[L][b].end(), T(0));
	            const int kstride = 2*hw + loc_nz;
	            for( int i=0; i<lyt.cluster_nx; ++i )
	                for( int j=0; j<lyt.cluster_ny; ++j )
	                    for( int kl=0; kl<loc_nz; ++kl ){
	                        const std::size_t dst = ((std::size_t)i * (std::size_t)lyt.cluster_ny
	                                                 + (std::size_t)j) * (std::size_t)kstride
	                                              + (std::size_t)(hw + kl);
	                        const std::size_t src = ((std::size_t)i * (std::size_t)lyt.cluster_ny
	                                                 + (std::size_t)j) * (std::size_t)loc_nz
	                                              + (std::size_t)kl;
	                        buf[dst] = interior[src];
	                    }
	            MUSIC::zoom_slab::halo_exchange_z(lyt, buf);
	        }
	    }
	}

	//! Phase G.1: inverse of sync_zslab_from_per_box. Z-slabs -> owner's
	//! per-box mesh interior. Halo cells in z-slabs are not pushed back
	//! (the per-box mesh's own halo is fed by sync_per_box_from_union from
	//! the union mesh, which carries inter-cluster boundary data).
	//!
	//! On non-owner ranks the owner's per-box mesh is NULL; this method
	//! sends slab interior to the owner who unpacks into bm.
	void sync_per_box_from_zslab(){
	    if( !m_zslab_enabled_ ) return;
	    const int my_rank = MUSIC::mpi::rank();
	    for( size_t L = 0; L < m_zslab_layouts_.size(); ++L ){
	        for( size_t b = 0; b < m_zslab_layouts_[L].size(); ++b ){
	            const MUSIC::zoom_slab::ZoomSlabLayout & lyt = m_zslab_layouts_[L][b];
	            const int owner = m_pbox_owner_[L][b];
	            const int hw = lyt.halo_w;
	            const int loc_nz = MUSIC::zoom_slab::local_nz(lyt);

	            // Pack interior from z-slab buffer (drop halos).
	            std::vector<T> interior((std::size_t)lyt.cluster_nx
	                                  * (std::size_t)lyt.cluster_ny
	                                  * (std::size_t)loc_nz, T(0));
	            const T* buf = m_zslab_data_[L][b].data();
	            const int kstride = 2*hw + loc_nz;
	            for( int i=0; i<lyt.cluster_nx; ++i )
	                for( int j=0; j<lyt.cluster_ny; ++j )
	                    for( int kl=0; kl<loc_nz; ++kl ){
	                        const std::size_t src = ((std::size_t)i * (std::size_t)lyt.cluster_ny
	                                                 + (std::size_t)j) * (std::size_t)kstride
	                                              + (std::size_t)(hw + kl);
	                        const std::size_t dst = ((std::size_t)i * (std::size_t)lyt.cluster_ny
	                                                 + (std::size_t)j) * (std::size_t)loc_nz
	                                              + (std::size_t)kl;
	                        interior[dst] = buf[src];
	                    }

	            // Owner allocates full_recv; non-owners pass NULL.
	            std::vector<T> full_recv;
	            if( my_rank == owner ){
	                full_recv.assign(MUSIC::zoom_slab::cluster_full_size(lyt), T(0));
	            }
	            MUSIC::zoom_slab::gather_to(lyt, owner, interior.data(),
	                                 my_rank == owner ? full_recv.data() : (T*)NULL);

	            // Owner unpacks into per-box mesh interior (halo untouched).
	            if( my_rank == owner ){
	                MeshvarBnd<T> * bm = m_pgrids_per_box_[L][b];
	                if( !bm ) throw std::runtime_error("G.1 sync_per_box_from_zslab: owner mesh NULL");
	                const int bnx = (int)bm->size(0);
	                const int bny = (int)bm->size(1);
	                const int bnz = (int)bm->size(2);
	                for( int i=0; i<bnx; ++i )
	                    for( int j=0; j<bny; ++j )
	                        for( int k=0; k<bnz; ++k ){
	                            (*bm)(i,j,k) =
	                                full_recv[((std::size_t)i*(std::size_t)bny + (std::size_t)j)
	                                          * (std::size_t)bnz + (std::size_t)k];
	                        }
	            }
	        }
	    }
	}

	//! Phase G.1 verification: roundtrip per-box -> z-slab -> per-box and
	//! verify owner's per-box mesh is bit-identical before/after. Returns
	//! true on success; false (after LOGERR) on mismatch.
	bool verify_zslab_roundtrip(){
	    if( !m_zslab_enabled_ ) return true;
	    const int my_rank = MUSIC::mpi::rank();
	    // Snapshot owner's per-box mesh contents.
	    std::vector< std::vector< std::vector<T> > > snap(m_pgrids_per_box_.size());
	    for( size_t L = 0; L < m_pgrids_per_box_.size(); ++L ){
	        snap[L].resize(m_pgrids_per_box_[L].size());
	        for( size_t b = 0; b < m_pgrids_per_box_[L].size(); ++b ){
	            if( m_pbox_owner_[L][b] != my_rank ) continue;
	            const MeshvarBnd<T> * bm = m_pgrids_per_box_[L][b];
	            if( !bm ) continue;
	            const int bnx = (int)bm->size(0);
	            const int bny = (int)bm->size(1);
	            const int bnz = (int)bm->size(2);
	            snap[L][b].assign((std::size_t)bnx*(std::size_t)bny*(std::size_t)bnz, T(0));
	            for( int i=0; i<bnx; ++i )
	                for( int j=0; j<bny; ++j )
	                    for( int k=0; k<bnz; ++k ){
	                        snap[L][b][((std::size_t)i*(std::size_t)bny + (std::size_t)j)
	                                   * (std::size_t)bnz + (std::size_t)k] = (*bm)(i,j,k);
	                    }
	        }
	    }
	    sync_zslab_from_per_box();
	    sync_per_box_from_zslab();
	    // Compare on owner ranks. mismatches counted locally; max via allreduce.
	    std::size_t local_mismatches = 0;
	    for( size_t L = 0; L < m_pgrids_per_box_.size(); ++L ){
	        for( size_t b = 0; b < m_pgrids_per_box_[L].size(); ++b ){
	            if( m_pbox_owner_[L][b] != my_rank ) continue;
	            const MeshvarBnd<T> * bm = m_pgrids_per_box_[L][b];
	            if( !bm ) continue;
	            const int bnx = (int)bm->size(0);
	            const int bny = (int)bm->size(1);
	            const int bnz = (int)bm->size(2);
	            const auto & ref = snap[L][b];
	            for( int i=0; i<bnx; ++i )
	                for( int j=0; j<bny; ++j )
	                    for( int k=0; k<bnz; ++k ){
	                        const T want = ref[((std::size_t)i*(std::size_t)bny + (std::size_t)j)
	                                           * (std::size_t)bnz + (std::size_t)k];
	                        const T got  = (*bm)(i,j,k);
	                        if( got != want ){
	                            if( local_mismatches < 4 )
	                                LOGERR("G.1 verify_zslab_roundtrip: mismatch L=%zu b=%zu (%d,%d,%d) got=%g want=%g",
	                                       L, b, i, j, k, (double)got, (double)want);
	                            ++local_mismatches;
	                        }
	                    }
	        }
	    }
#ifdef USE_MPI
	    std::size_t global_mismatches = 0;
	    MPI_Allreduce(&local_mismatches, &global_mismatches, 1,
	                  MPI_UNSIGNED_LONG, MPI_SUM, MUSIC::mpi::world());
	    local_mismatches = global_mismatches;
#endif
	    if( local_mismatches > 0 ){
	        LOGERR("G.1 verify_zslab_roundtrip: %zu cells differ after roundtrip", local_mismatches);
	        return false;
	    }
	    LOGINFO("G.1 verify_zslab_roundtrip: PASSED — per-box -> z-slab -> per-box bit-identical");
	    return true;
	}
	//---------------------------------------------------------------------

	//-------- Phase E.2.0: slab-union helpers ----------------------------
private:
	void ensure_level_slab_meta_(size_t L)
	{
	    if( m_level_is_slab_.size() <= L ) {
	        m_level_is_slab_     .resize(L+1, false);
	        m_level_tenant_      .resize(L+1, false);
	        m_level_slab_backup_ .resize(L+1, (MeshvarBnd<T>*)NULL);
	        m_level_gnx_         .resize(L+1, 0);
	        m_level_gny_         .resize(L+1, 0);
	        m_level_gnz_         .resize(L+1, 0);
	    }
	}
public:
	//! True when m_pgrids[L] is a slab-distributed mesh (set by
	//! allocate_union_slab_at). False otherwise (legacy full-mesh path).
	bool is_level_slab( unsigned ilevel ) const
	{
	    if( ilevel >= m_level_is_slab_.size() ) return false;
	    return m_level_is_slab_[ilevel];
	}

	//! True only on rank 0 while a transient full-union tenant is active
	//! at level L (between gather_union_to_root and scatter_union_from_root).
	bool has_union_tenant( unsigned ilevel ) const
	{
	    if( ilevel >= m_level_tenant_.size() ) return false;
	    return m_level_tenant_[ilevel];
	}

	//! Global x-extent of the (possibly slab-distributed) union at level L.
	//! Returns the slab metadata when is_level_slab(L), else m_pgrids[L]->size(dim).
	size_t union_global_size( unsigned ilevel, int dim ) const
	{
	    if( is_level_slab(ilevel) ) {
	        if( dim == 0 ) return m_level_gnx_[ilevel];
	        if( dim == 1 ) return m_level_gny_[ilevel];
	        return m_level_gnz_[ilevel];
	    }
	    return m_pgrids[ilevel]->size((unsigned)dim);
	}

	//! Replace m_pgrids[L] with a slab-distributed MeshvarBnd<T>. Each rank
	//! holds only its x-slab of the global (gnx, gny, gnz) grid plus the
	//! standard m_nbnd ghost halo on all faces. Requires consumers to include
	//! "mesh_distributed.hh" (where MUSIC::dist::make_slab_meshvarbnd lives)
	//! before any TU instantiates this method.
	//!
	//! On a serial build (USE_MPI undefined) the slab degenerates to the full
	//! grid, so this method is a no-cost equivalent of allocating a normal
	//! MeshvarBnd<T>. Existing content at m_pgrids[L] is dropped.
	void allocate_union_slab_at( unsigned ilevel,
	                             size_t gnx, size_t gny, size_t gnz,
	                             int oax=0, int oay=0, int oaz=0 )
	{
	    ensure_level_slab_meta_(ilevel);
	    if( ilevel >= m_pgrids.size() ) {
	        LOGERR("allocate_union_slab_at: level %u beyond hierarchy (size=%zu)",
	               ilevel, m_pgrids.size());
	        throw std::runtime_error("allocate_union_slab_at: level out of range");
	    }
	    if( m_level_tenant_[ilevel] ) {
	        LOGERR("allocate_union_slab_at: level %u has active tenant; "
	               "release_union_root_tenant first", ilevel);
	        throw std::runtime_error("allocate_union_slab_at: tenant active");
	    }
	    // Drop existing storage (was either a full union or a stale slab).
	    delete m_pgrids[ilevel];
	    m_pgrids[ilevel] = MUSIC::dist::make_slab_meshvarbnd<T>(
	        (int)m_nbnd, gnx, gny, gnz, oax, oay, oaz );
	    m_level_is_slab_[ilevel] = true;
	    m_level_gnx_[ilevel] = gnx;
	    m_level_gny_[ilevel] = gny;
	    m_level_gnz_[ilevel] = gnz;
	    // Keep the per-level offset cache consistent with the underlying mesh.
	    if( ilevel < m_xoffabs.size() ) m_xoffabs[ilevel] = oax;
	    if( ilevel < m_yoffabs.size() ) m_yoffabs[ilevel] = oay;
	    if( ilevel < m_zoffabs.size() ) m_zoffabs[ilevel] = oaz;
	}

	//! Collective. Rank 0 acquires a full-union MeshvarBnd<T> tenant at
	//! level L by gathering each rank's interior slab. Workers' slab
	//! pointers are unchanged. m_pgrids[L] on rank 0 is swapped to the
	//! tenant; the slab pointer is saved in m_level_slab_backup_[L].
	//!
	//! No-op when L is not slab. Must run OUTSIDE phase_scope (workers
	//! cannot service MPI_Send while parked in worker_pump).
	//!
	//! Halo cells of the tenant are zero-initialised; caller is responsible
	//! for setting periodic / boundary halos if downstream code depends on them.
	void gather_union_to_root( unsigned ilevel )
	{
	    if( !is_level_slab(ilevel) ) return;
	    if( ilevel >= m_pgrids.size() || m_pgrids[ilevel] == NULL ) return;
	    if( m_level_tenant_[ilevel] ) return; // already gathered

#ifdef USE_MPI
	    const int nproc = MUSIC::mpi::size();
	    if( nproc <= 1 ) return; // serial: slab == full union; no work
	    const int rk = MUSIC::mpi::rank();
	    const size_t gnx  = m_level_gnx_[ilevel];
	    const size_t gny  = m_level_gny_[ilevel];
	    const size_t gnz  = m_level_gnz_[ilevel];
	    const int nbnd    = (int)m_nbnd;
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
	    const int tag_base = 1500; // distinct tag block from per-box gather (1100)

	    MeshvarBnd<T> * slab = m_pgrids[ilevel];

	    // ---- collect per-rank slab geometry on rank 0 ----------------------
	    long long my_loc[2] = { (long long)slab->local_x_start(),
	                            (long long)slab->local_nx() };
	    std::vector<long long> all_loc(2*nproc, 0);
	    MPI_Gather(my_loc, 2, MPI_LONG_LONG, all_loc.data(), 2, MPI_LONG_LONG,
	               0, MUSIC::mpi::world());

	    if( rk == 0 ) {
	        // ---- allocate full-union tenant on rank 0 ----------------------
	        MeshvarBnd<T> * tenant = new MeshvarBnd<T>(
	            m_nbnd, gnx, gny, gnz,
	            (size_t)slab->offset(0), (size_t)slab->offset(1), (size_t)slab->offset(2) );
	        tenant->zero();

	        // copy rank-0's own slab interior into tenant interior
	        {
	            const long long lx_start = all_loc[0];
	            const long long lx_nx    = all_loc[1];
	            for( long long i=0; i<lx_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        (*tenant)((int)(lx_start+i),(int)j,(int)k)
	                            = (*slab)((int)i,(int)j,(int)k);
	        }

	        // receive each worker's slab interior
	        for( int r=1; r<nproc; ++r ) {
	            const long long lx_start = all_loc[2*r+0];
	            const long long lx_nx    = all_loc[2*r+1];
	            if( lx_nx <= 0 ) continue;
	            const size_t cnt = (size_t)lx_nx * (size_t)gny * (size_t)gnz;
	            std::vector<T> buf(cnt);
	            MPI_Recv(buf.data(), (int)cnt, dtype, r, tag_base+r,
	                     MUSIC::mpi::world(), MPI_STATUS_IGNORE);
	            // write into tenant at (lx_start+i, j, k)
	            for( long long i=0; i<lx_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        (*tenant)((int)(lx_start+i),(int)j,(int)k)
	                            = buf[(size_t)((i*(long long)gny + j)*(long long)gnz + k)];
	        }

	        m_level_slab_backup_[ilevel] = slab;   // save slab pointer
	        m_pgrids[ilevel]             = tenant; // swap to tenant
	        m_level_tenant_[ilevel]      = true;
	    } else {
	        // workers: pack and send slab interior to rank 0
	        const long long lx_nx = my_loc[1];
	        if( lx_nx > 0 ) {
	            const size_t cnt = (size_t)lx_nx * (size_t)gny * (size_t)gnz;
	            std::vector<T> buf(cnt);
	            for( long long i=0; i<lx_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        buf[(size_t)((i*(long long)gny + j)*(long long)gnz + k)]
	                            = (*slab)((int)i,(int)j,(int)k);
	            MPI_Send(buf.data(), (int)cnt, dtype, 0, tag_base+rk,
	                     MUSIC::mpi::world());
	        }
	    }
#endif
	}

	//! Collective. Inverse of gather_union_to_root. Rank 0 scatters its
	//! tenant back into each rank's slab interior, frees the tenant, and
	//! restores m_pgrids[L] to the original slab pointer. Workers' slabs
	//! receive the up-to-date data. No-op when no tenant is active.
	//! Must run OUTSIDE phase_scope.
	//!
	//! Halo cells of the resulting slabs are NOT exchanged here; caller
	//! must invoke MUSIC::dist::halo_exchange_x() if needed.
	void scatter_union_from_root( unsigned ilevel )
	{
	    if( !is_level_slab(ilevel) ) return;
#ifdef USE_MPI
	    const int nproc = MUSIC::mpi::size();
	    if( nproc <= 1 ) return;
	    const int rk = MUSIC::mpi::rank();
	    const size_t gnx = m_level_gnx_[ilevel];
	    const size_t gny = m_level_gny_[ilevel];
	    const size_t gnz = m_level_gnz_[ilevel];
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
	    const int tag_base = 1700; // distinct from gather (1500)

	    // collect each rank's slab geometry on rank 0
	    MeshvarBnd<T> * local_slab =
	        (rk == 0) ? m_level_slab_backup_[ilevel] : m_pgrids[ilevel];
	    long long my_loc[2] = { local_slab ? (long long)local_slab->local_x_start() : 0,
	                            local_slab ? (long long)local_slab->local_nx()      : 0 };
	    std::vector<long long> all_loc(2*nproc, 0);
	    MPI_Gather(my_loc, 2, MPI_LONG_LONG, all_loc.data(), 2, MPI_LONG_LONG,
	               0, MUSIC::mpi::world());
	    (void)gnx;

	    if( rk == 0 ) {
	        if( !m_level_tenant_[ilevel] ) return;
	        MeshvarBnd<T> * tenant = m_pgrids[ilevel];
	        MeshvarBnd<T> * slab   = m_level_slab_backup_[ilevel];
	        if( !tenant || !slab ) {
	            LOGERR("scatter_union_from_root: missing tenant/slab at L=%u", ilevel);
	            throw std::runtime_error("scatter_union_from_root: missing buffers");
	        }
	        // copy rank-0 own slab interior from tenant
	        {
	            const long long lx_start = all_loc[0];
	            const long long lx_nx    = all_loc[1];
	            for( long long i=0; i<lx_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        (*slab)((int)i,(int)j,(int)k)
	                            = (*tenant)((int)(lx_start+i),(int)j,(int)k);
	        }
	        // send each worker its slab interior
	        for( int r=1; r<nproc; ++r ) {
	            const long long lx_start = all_loc[2*r+0];
	            const long long lx_nx    = all_loc[2*r+1];
	            if( lx_nx <= 0 ) continue;
	            const size_t cnt = (size_t)lx_nx * (size_t)gny * (size_t)gnz;
	            std::vector<T> buf(cnt);
	            for( long long i=0; i<lx_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        buf[(size_t)((i*(long long)gny + j)*(long long)gnz + k)]
	                            = (*tenant)((int)(lx_start+i),(int)j,(int)k);
	            MPI_Send(buf.data(), (int)cnt, dtype, r, tag_base+r,
	                     MUSIC::mpi::world());
	        }
	        // free tenant, restore slab pointer
	        delete tenant;
	        m_pgrids[ilevel] = slab;
	        m_level_slab_backup_[ilevel] = NULL;
	        m_level_tenant_[ilevel] = false;
	    } else {
	        const long long lx_nx = my_loc[1];
	        if( lx_nx > 0 ) {
	            const size_t cnt = (size_t)lx_nx * (size_t)gny * (size_t)gnz;
	            std::vector<T> buf(cnt);
	            MPI_Recv(buf.data(), (int)cnt, dtype, 0, tag_base+rk,
	                     MUSIC::mpi::world(), MPI_STATUS_IGNORE);
	            for( long long i=0; i<lx_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        (*local_slab)((int)i,(int)j,(int)k)
	                            = buf[(size_t)((i*(long long)gny + j)*(long long)gnz + k)];
	        }
	    }
#endif
	}

	//! Rank-0 only. Drop the tenant without shipping data back to workers
	//! (use after a read-only gather pattern where the workers' slabs are
	//! still authoritative or about to be replaced). No-op on other ranks
	//! or when no tenant is active.
	void release_union_root_tenant( unsigned ilevel )
	{
	    if( !is_level_slab(ilevel) ) return;
	    if( MUSIC::mpi::rank() != 0 ) return;
	    if( !m_level_tenant_[ilevel] ) return;
	    delete m_pgrids[ilevel];
	    m_pgrids[ilevel] = m_level_slab_backup_[ilevel];
	    m_level_slab_backup_[ilevel] = NULL;
	    m_level_tenant_[ilevel] = false;
	}

	//! Phase E.2.1a smoke test. Standalone validation of E.2.0 gather/scatter
	//! plumbing on real density data. Preconditions: rank 0 holds m_pgrids[L]
	//! as a populated union MeshvarBnd<T>; workers may have an empty hierarchy.
	//! Does NOT modify m_pgrids[L]. Procedure:
	//!   1. Rank 0 scatters its union content interior-strip-by-interior-strip
	//!      to per-rank slab buffers (workers receive into local slab).
	//!   2. Workers gather slab content back to rank 0; rank 0 receives into a
	//!      verification union buffer.
	//!   3. Rank 0 compares the verification union against m_pgrids[L] cell by
	//!      cell. Throws if any cell mismatches (loud failure: indicates a bug
	//!      in scatter or gather indexing, not floating-point divergence).
	//! After return, all temporary slab/verification buffers are freed.
	//!
	//! Must run OUTSIDE phase_scope (workers cannot service MPI_Send while
	//! parked in worker_pump). On serial builds this is a no-op.
	void test_slab_roundtrip_at( unsigned ilevel,
	                              size_t gnx, size_t gny, size_t gnz )
	{
#ifdef USE_MPI
	    const int nproc = MUSIC::mpi::size();
	    if( nproc <= 1 ) return;
	    const int rk = MUSIC::mpi::rank();
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
	    const int tag_scatter = 1800; // distinct from gather=1500, scatter=1700
	    const int tag_gather  = 1900;

	    // ---- compute slab layout via factory (FFTW MPI decomposition) ----
	    // We don't need the storage, just the geometry. Allocating a transient
	    // slab MeshvarBnd<T> is cheap (workers see only their local slab) and
	    // keeps the FFTW dependency out of mesh.hh.
	    MeshvarBnd<T> * tmp_slab = MUSIC::dist::make_slab_meshvarbnd<T>(
	        (int)m_nbnd, gnx, gny, gnz, 0, 0, 0 );
	    const size_t lx_start = (size_t)tmp_slab->local_x_start();
	    const size_t lx_nx    = (size_t)tmp_slab->local_nx();
	    delete tmp_slab;

	    // ---- broadcast geometry so rank 0 knows each rank's slab range ----
	    long long my_loc[2] = { (long long)lx_start, (long long)lx_nx };
	    std::vector<long long> all_loc(2*nproc, 0);
	    MPI_Gather(my_loc, 2, MPI_LONG_LONG, all_loc.data(), 2, MPI_LONG_LONG,
	               0, MUSIC::mpi::world());

	    // ---- step 1: rank 0 scatters union → per-rank slab buffers -------
	    std::vector<T> my_slab( (lx_nx>0 ? lx_nx*gny*gnz : 0) );

	    if( rk == 0 ) {
	        if( ilevel >= m_pgrids.size() || m_pgrids[ilevel] == NULL ) {
	            LOGERR("test_slab_roundtrip_at: rank 0 has no m_pgrids[%u]", ilevel);
	            throw std::runtime_error("test_slab_roundtrip_at: missing union");
	        }
	        MeshvarBnd<T> * U = m_pgrids[ilevel];
	        // rank-0 own slab: copy directly
	        {
	            const long long s_lx = all_loc[0];
	            const long long s_nx = all_loc[1];
	            for( long long i=0; i<s_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        my_slab[(size_t)((i*(long long)gny+j)*(long long)gnz+k)]
	                            = (*U)((int)(s_lx+i),(int)j,(int)k);
	        }
	        // workers: pack and send their slab portion
	        for( int r=1; r<nproc; ++r ) {
	            const long long s_lx = all_loc[2*r+0];
	            const long long s_nx = all_loc[2*r+1];
	            if( s_nx <= 0 ) continue;
	            const size_t cnt = (size_t)s_nx * gny * gnz;
	            std::vector<T> buf(cnt);
	            for( long long i=0; i<s_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        buf[(size_t)((i*(long long)gny+j)*(long long)gnz+k)]
	                            = (*U)((int)(s_lx+i),(int)j,(int)k);
	            MPI_Send(buf.data(), (int)cnt, dtype, r, tag_scatter+r,
	                     MUSIC::mpi::world());
	        }
	    } else {
	        if( lx_nx > 0 ) {
	            MPI_Recv(my_slab.data(), (int)(lx_nx*gny*gnz), dtype, 0, tag_scatter+rk,
	                     MUSIC::mpi::world(), MPI_STATUS_IGNORE);
	        }
	    }

	    // ---- step 2: workers gather slab → rank-0 verification union -----
	    if( rk == 0 ) {
	        std::vector<T> verify( gnx*gny*gnz, T(0) );
	        // rank-0 own slab → verify
	        {
	            const long long s_lx = all_loc[0];
	            const long long s_nx = all_loc[1];
	            for( long long i=0; i<s_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        verify[(size_t)(((s_lx+i)*(long long)gny+j)*(long long)gnz+k)]
	                            = my_slab[(size_t)((i*(long long)gny+j)*(long long)gnz+k)];
	        }
	        // workers → verify
	        for( int r=1; r<nproc; ++r ) {
	            const long long s_lx = all_loc[2*r+0];
	            const long long s_nx = all_loc[2*r+1];
	            if( s_nx <= 0 ) continue;
	            const size_t cnt = (size_t)s_nx * gny * gnz;
	            std::vector<T> buf(cnt);
	            MPI_Recv(buf.data(), (int)cnt, dtype, r, tag_gather+r,
	                     MUSIC::mpi::world(), MPI_STATUS_IGNORE);
	            for( long long i=0; i<s_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        verify[(size_t)(((s_lx+i)*(long long)gny+j)*(long long)gnz+k)]
	                            = buf[(size_t)((i*(long long)gny+j)*(long long)gnz+k)];
	        }
	        // ---- step 3: compare verify against m_pgrids[L] interior ------
	        MeshvarBnd<T> * U = m_pgrids[ilevel];
	        size_t mismatches = 0;
	        T max_diff = T(0);
	        for( size_t i=0; i<gnx; ++i )
	            for( size_t j=0; j<gny; ++j )
	                for( size_t k=0; k<gnz; ++k ) {
	                    const T a = verify[(i*gny+j)*gnz+k];
	                    const T b = (*U)((int)i,(int)j,(int)k);
	                    const T d = (a > b) ? (a-b) : (b-a);
	                    if( d != T(0) ) { ++mismatches; if( d > max_diff ) max_diff = d; }
	                }
	        if( mismatches > 0 ) {
	            LOGERR("test_slab_roundtrip_at L=%u FAILED: %zu mismatches, max_diff=%.6e",
	                   ilevel, mismatches, (double)max_diff);
	            throw std::runtime_error("test_slab_roundtrip_at: roundtrip not bit-identical");
	        }
	        LOGINFO("E.2.1a roundtrip test L=%u PASSED: %zu cells, all bit-identical across %d ranks",
	                ilevel, gnx*gny*gnz, nproc);
	    } else {
	        if( lx_nx > 0 ) {
	            MPI_Send(my_slab.data(), (int)(lx_nx*gny*gnz), dtype, 0, tag_gather+rk,
	                     MUSIC::mpi::world());
	        }
	    }
#else
	    (void)ilevel; (void)gnx; (void)gny; (void)gnz;
#endif
	}
	//---------------------------------------------------------------------

	//-------- Phase E.2.2a: SPMD slab-solve validation test --------------
	//! Standalone smoke test. Computes the Poisson solve two ways and
	//! checks bit-identical:
	//!   A) "expected": rank 0 snapshots delta union → padded buffer →
	//!      rank0_dist_solve (existing Scatterv/FFT/Gatherv path).
	//!   B) "computed": flip level L to slab storage → allocate dst slab →
	//!      register pointers → rank0_dist_solve_slab (E.2.2a SPMD path
	//!      with no Scatterv/Gatherv) → gather dst back to rank 0.
	//! Then convert_level_slab_to_full restores delta to its original full
	//! storage on rank 0 so callers can continue unchanged.
	//!
	//! Must run OUTSIDE phase_scope (the convert helpers and gather are
	//! collective and cannot run while workers are parked in worker_pump).
	void test_slab_solve_at( unsigned ilevel,
	                          size_t gnx, size_t gny, size_t gnz )
	{
#ifdef USE_MPI
	    const int nproc = MUSIC::mpi::size();
	    if( nproc <= 1 ) return;
	    const int rk = MUSIC::mpi::rank();
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
	    const int tag_gather = 2500; // distinct from 1500/1700/1800/1900/2100/2300

	    // ---- compute slab layout via factory ----
	    MeshvarBnd<T> * tmp_slab = MUSIC::dist::make_slab_meshvarbnd<T>(
	        (int)m_nbnd, gnx, gny, gnz, 0, 0, 0 );
	    const size_t lx_start = (size_t)tmp_slab->local_x_start();
	    const size_t lx_nx    = (size_t)tmp_slab->local_nx();
	    delete tmp_slab;

	    long long my_loc[2] = { (long long)lx_start, (long long)lx_nx };
	    std::vector<long long> all_loc(2*nproc, 0);
	    MPI_Gather(my_loc, 2, MPI_LONG_LONG, all_loc.data(), 2, MPI_LONG_LONG,
	               0, MUSIC::mpi::world());

	    const size_t nz_complex = gnz/2 + 1;
	    const size_t nz_padded  = 2*nz_complex;

	    // ---- Phase A: rank-0 builds "expected" via full-path solve --------
	    std::vector<T> expected; // rank 0 only: gnx*gny*nz_padded layout
	    if( rk == 0 ) {
	        if( ilevel >= m_pgrids.size() || m_pgrids[ilevel] == NULL ) {
	            LOGERR("test_slab_solve_at: rank 0 has no m_pgrids[%u]", ilevel);
	            throw std::runtime_error("test_slab_solve_at: missing union");
	        }
	        MeshvarBnd<T> * U = m_pgrids[ilevel];
	        expected.assign( gnx*gny*nz_padded, T(0) );
	        for( size_t i=0; i<gnx; ++i )
	            for( size_t j=0; j<gny; ++j )
	                for( size_t k=0; k<gnz; ++k )
	                    expected[(i*gny+j)*nz_padded+k] = (*U)((int)i,(int)j,(int)k);
	    }
	    {
	        MUSIC::poisson::phase_scope _ps;
	        if( rk == 0 ) {
	            MUSIC::poisson::rank0_dist_solve<T>( expected.data(), gnx, gny, gnz );
	        }
	    } // phase_scope dtor on rank 0 broadcasts OP_DONE → workers exit pump

	    // ---- Phase B: flip storage to slab; allocate dst slab; register ---
	    convert_level_full_to_slab( ilevel, gnx, gny, gnz );

	    MeshvarBnd<T> * dst_slab = MUSIC::dist::make_slab_meshvarbnd<T>(
	        (int)m_nbnd, gnx, gny, gnz, 0, 0, 0 );

	    // src slab is now m_pgrids[ilevel] (on every rank, allocated by convert)
	    MUSIC::poisson::set_slab_solve_inout<T>( m_pgrids[ilevel], dst_slab );

	    {
	        MUSIC::poisson::phase_scope _ps;
	        if( rk == 0 ) {
	            MUSIC::poisson::rank0_dist_solve_slab<T>( gnx, gny, gnz );
	        }
	    }

	    // ---- Phase C: gather dst_slab back to rank 0 "computed" buffer ----
	    std::vector<T> my_slab( (lx_nx>0 ? lx_nx*gny*gnz : 0) );
	    if( lx_nx > 0 ) {
	        for( size_t i=0; i<lx_nx; ++i )
	            for( size_t j=0; j<gny; ++j )
	                for( size_t k=0; k<gnz; ++k )
	                    my_slab[(i*gny+j)*gnz+k] = (*dst_slab)((int)i,(int)j,(int)k);
	    }
	    if( rk == 0 ) {
	        std::vector<T> computed( gnx*gny*gnz, T(0) );
	        {
	            const long long s_lx = all_loc[0];
	            const long long s_nx = all_loc[1];
	            for( long long i=0; i<s_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        computed[(size_t)(((s_lx+i)*(long long)gny+j)*(long long)gnz+k)]
	                            = my_slab[(size_t)((i*(long long)gny+j)*(long long)gnz+k)];
	        }
	        for( int r=1; r<nproc; ++r ) {
	            const long long s_lx = all_loc[2*r+0];
	            const long long s_nx = all_loc[2*r+1];
	            if( s_nx <= 0 ) continue;
	            const size_t cnt = (size_t)s_nx * gny * gnz;
	            std::vector<T> buf(cnt);
	            MPI_Recv(buf.data(), (int)cnt, dtype, r, tag_gather+r,
	                     MUSIC::mpi::world(), MPI_STATUS_IGNORE);
	            for( long long i=0; i<s_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        computed[(size_t)(((s_lx+i)*(long long)gny+j)*(long long)gnz+k)]
	                            = buf[(size_t)((i*(long long)gny+j)*(long long)gnz+k)];
	        }
	        // compare (only k ∈ [0..gnz); skip nz_padded leftover in expected)
	        size_t mismatches = 0;
	        T max_diff = T(0);
	        for( size_t i=0; i<gnx; ++i )
	            for( size_t j=0; j<gny; ++j )
	                for( size_t k=0; k<gnz; ++k ) {
	                    const T a = expected[(i*gny+j)*nz_padded+k];
	                    const T b = computed[(i*gny+j)*gnz+k];
	                    const T d = (a > b) ? (a-b) : (b-a);
	                    if( d != T(0) ) { ++mismatches; if( d > max_diff ) max_diff = d; }
	                }
	        if( mismatches > 0 ) {
	            LOGERR("test_slab_solve_at L=%u FAILED: %zu mismatches, max_diff=%.6e",
	                   ilevel, mismatches, (double)max_diff);
	            throw std::runtime_error("test_slab_solve_at: slab-solve != full-solve");
	        }
	        LOGINFO("E.2.2a slab-solve test L=%u PASSED: %zu cells, bit-identical vs full-path across %d ranks",
	                ilevel, gnx*gny*gnz, nproc);
	    } else {
	        if( lx_nx > 0 ) {
	            MPI_Send(my_slab.data(), (int)(lx_nx*gny*gnz), dtype, 0, tag_gather+rk,
	                     MUSIC::mpi::world());
	        }
	    }

	    // ---- Phase D: cleanup --------------------------------------------
	    MUSIC::poisson::set_slab_solve_inout<T>( (MeshvarBnd<T>*)NULL, (MeshvarBnd<T>*)NULL );
	    delete dst_slab;
	    convert_level_slab_to_full( ilevel );
#else
	    (void)ilevel; (void)gnx; (void)gny; (void)gnz;
#endif
	}
	//---------------------------------------------------------------------

	//-------- Phase E.2.1b: storage-flip helpers (full <-> slab) --------
	//! Collective. Convert level L from rank-0-full storage to all-rank
	//! slab storage, preserving the data content.
	//!
	//! Preconditions:
	//!   - rank 0 holds m_pgrids[L] as a populated full MeshvarBnd<T>
	//!     with extent (gnx, gny, gnz) (typically just allocated by
	//!     create_base_hierarchy + top->copy)
	//!   - workers may have empty m_pgrids; we resize and populate slot L.
	//!
	//! Postconditions on all ranks: m_pgrids[L] is a slab MeshvarBnd<T>
	//! holding that rank's x-strip; is_level_slab(L) == true; data preserved.
	//!
	//! Must run OUTSIDE phase_scope (workers cannot service MPI_Send while
	//! parked in worker_pump). On serial builds the existing full mesh stays
	//! in place (slab == full); only the metadata flips.
	void convert_level_full_to_slab( unsigned ilevel,
	                                 size_t gnx, size_t gny, size_t gnz,
	                                 int oax=0, int oay=0, int oaz=0 )
	{
#ifdef USE_MPI
	    const int nproc = MUSIC::mpi::size();
	    if( nproc <= 1 ) {
	        ensure_level_slab_meta_(ilevel);
	        if( ilevel < m_pgrids.size() && m_pgrids[ilevel] != NULL ) {
	            m_level_is_slab_[ilevel] = true;
	            m_level_gnx_[ilevel] = gnx;
	            m_level_gny_[ilevel] = gny;
	            m_level_gnz_[ilevel] = gnz;
	        }
	        return;
	    }
	    const int rk = MUSIC::mpi::rank();
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
	    const int tag_base = 2100; // E.2.1b distinct from gather(1500)/scatter(1700)/roundtrip(1800,1900)

	    // ensure m_pgrids and offset metadata sized to L+1 on every rank
	    if( m_pgrids.size() <= ilevel ) m_pgrids.resize(ilevel+1, (MeshvarBnd<T>*)NULL);
	    if( m_xoffabs.size() <= ilevel ) m_xoffabs.resize(ilevel+1, 0);
	    if( m_yoffabs.size() <= ilevel ) m_yoffabs.resize(ilevel+1, 0);
	    if( m_zoffabs.size() <= ilevel ) m_zoffabs.resize(ilevel+1, 0);
	    ensure_level_slab_meta_(ilevel);

	    // H.4 idempotency: if the level is already slab with matching dims,
	    // no-op. Lets create_base_hierarchy_slab + slab_solve_unigrid wrappers
	    // compose without double-converting.
	    if( m_level_is_slab_[ilevel]
	        && m_level_gnx_[ilevel] == gnx
	        && m_level_gny_[ilevel] == gny
	        && m_level_gnz_[ilevel] == gnz )
	    {
	        return;
	    }

	    // detach the rank-0 source full so allocate_union_slab_at doesn't free it
	    MeshvarBnd<T> * src_full = NULL;
	    if( rk == 0 ) {
	        src_full = m_pgrids[ilevel];
	        if( !src_full ) {
	            LOGERR("convert_level_full_to_slab: rank 0 has no m_pgrids[%u]", ilevel);
	            throw std::runtime_error("convert_level_full_to_slab: missing source full");
	        }
	        m_pgrids[ilevel] = NULL;
	    }

	    // every rank now allocates its slab
	    allocate_union_slab_at(ilevel, gnx, gny, gnz, oax, oay, oaz);
	    MeshvarBnd<T> * new_slab = m_pgrids[ilevel];

	    // gather per-rank slab geometry on rank 0
	    long long my_loc[2] = { (long long)new_slab->local_x_start(),
	                            (long long)new_slab->local_nx() };
	    std::vector<long long> all_loc(2*nproc, 0);
	    MPI_Gather(my_loc, 2, MPI_LONG_LONG, all_loc.data(), 2, MPI_LONG_LONG,
	               0, MUSIC::mpi::world());

	    if( rk == 0 ) {
	        // rank-0 own strip
	        {
	            const long long s_lx = all_loc[0];
	            const long long s_nx = all_loc[1];
	            for( long long i=0; i<s_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        (*new_slab)((int)i,(int)j,(int)k)
	                            = (*src_full)((int)(s_lx+i),(int)j,(int)k);
	        }
	        // pack + send each worker's strip
	        for( int r=1; r<nproc; ++r ) {
	            const long long s_lx = all_loc[2*r+0];
	            const long long s_nx = all_loc[2*r+1];
	            if( s_nx <= 0 ) continue;
	            const size_t cnt = (size_t)s_nx * gny * gnz;
	            std::vector<T> buf(cnt);
	            for( long long i=0; i<s_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        buf[(size_t)((i*(long long)gny+j)*(long long)gnz+k)]
	                            = (*src_full)((int)(s_lx+i),(int)j,(int)k);
	            MPI_Send(buf.data(), (int)cnt, dtype, r, tag_base+r,
	                     MUSIC::mpi::world());
	        }
	        delete src_full;
	    } else {
	        const long long s_nx = my_loc[1];
	        if( s_nx > 0 ) {
	            const size_t cnt = (size_t)s_nx * gny * gnz;
	            std::vector<T> buf(cnt);
	            MPI_Recv(buf.data(), (int)cnt, dtype, 0, tag_base+rk,
	                     MUSIC::mpi::world(), MPI_STATUS_IGNORE);
	            for( long long i=0; i<s_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        (*new_slab)((int)i,(int)j,(int)k)
	                            = buf[(size_t)((i*(long long)gny+j)*(long long)gnz+k)];
	        }
	    }
#else
	    (void)ilevel; (void)gnx; (void)gny; (void)gnz; (void)oax; (void)oay; (void)oaz;
#endif
	}

	//! Collective. Inverse of convert_level_full_to_slab. Gathers slabs
	//! back to a rank-0 full MeshvarBnd<T>; workers drop their slab.
	//!
	//! Preconditions on all ranks: m_pgrids[L] is a slab (is_level_slab(L) == true).
	//!
	//! Postconditions:
	//!   - rank 0: m_pgrids[L] is a full MeshvarBnd<T> with extent (gnx,gny,gnz)
	//!     (taken from m_level_gnx_/gny_/gnz_); is_level_slab(L) == false.
	//!   - workers (keep_workers_full=false): m_pgrids[L] == NULL.
	//!   - workers (keep_workers_full=true): m_pgrids[L] is a zero-filled full
	//!     MeshvarBnd<T> with extent (gnx,gny,gnz) matching rank 0. Used by
	//!     the H.4 path when spmd_mg_skel is on so broadcast_union_mesh_from_owner
	//!     and friends do not NULL-deref at single-box-parent recurse sites.
	//!     is_level_slab(L) == false in both cases.
	//!
	//! Must run OUTSIDE phase_scope. On serial builds: flips metadata only.
	void convert_level_slab_to_full( unsigned ilevel, bool keep_workers_full = false )
	{
#ifdef USE_MPI
	    const int nproc = MUSIC::mpi::size();
	    if( nproc <= 1 ) {
	        if( ilevel < m_level_is_slab_.size() ) m_level_is_slab_[ilevel] = false;
	        return;
	    }
	    if( !is_level_slab(ilevel) ) return;
	    if( m_level_tenant_[ilevel] ) {
	        LOGERR("convert_level_slab_to_full: tenant active at L=%u; release first", ilevel);
	        throw std::runtime_error("convert_level_slab_to_full: tenant active");
	    }
	    const int rk = MUSIC::mpi::rank();
	    MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
	    const int tag_base = 2300; // E.2.1b distinct from full->slab (2100)

	    const size_t gnx = m_level_gnx_[ilevel];
	    const size_t gny = m_level_gny_[ilevel];
	    const size_t gnz = m_level_gnz_[ilevel];

	    MeshvarBnd<T> * slab = m_pgrids[ilevel];
	    long long my_loc[2] = { slab ? (long long)slab->local_x_start() : 0,
	                            slab ? (long long)slab->local_nx()      : 0 };
	    std::vector<long long> all_loc(2*nproc, 0);
	    MPI_Gather(my_loc, 2, MPI_LONG_LONG, all_loc.data(), 2, MPI_LONG_LONG,
	               0, MUSIC::mpi::world());

	    if( rk == 0 ) {
	        MeshvarBnd<T> * full = new MeshvarBnd<T>(
	            m_nbnd, gnx, gny, gnz, 0, 0, 0 );
	        full->zero();
	        // rank-0 own slab -> full
	        {
	            const long long s_lx = all_loc[0];
	            const long long s_nx = all_loc[1];
	            for( long long i=0; i<s_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        (*full)((int)(s_lx+i),(int)j,(int)k)
	                            = (*slab)((int)i,(int)j,(int)k);
	        }
	        // workers' slabs -> full
	        for( int r=1; r<nproc; ++r ) {
	            const long long s_lx = all_loc[2*r+0];
	            const long long s_nx = all_loc[2*r+1];
	            if( s_nx <= 0 ) continue;
	            const size_t cnt = (size_t)s_nx * gny * gnz;
	            std::vector<T> buf(cnt);
	            MPI_Recv(buf.data(), (int)cnt, dtype, r, tag_base+r,
	                     MUSIC::mpi::world(), MPI_STATUS_IGNORE);
	            for( long long i=0; i<s_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        (*full)((int)(s_lx+i),(int)j,(int)k)
	                            = buf[(size_t)((i*(long long)gny+j)*(long long)gnz+k)];
	        }
	        // replace slab with full
	        delete slab;
	        m_pgrids[ilevel] = full;
	        m_level_is_slab_[ilevel] = false;
	        m_level_gnx_[ilevel] = 0;
	        m_level_gny_[ilevel] = 0;
	        m_level_gnz_[ilevel] = 0;
	    } else {
	        const long long s_nx = my_loc[1];
	        if( s_nx > 0 ) {
	            const size_t cnt = (size_t)s_nx * gny * gnz;
	            std::vector<T> buf(cnt);
	            for( long long i=0; i<s_nx; ++i )
	                for( long long j=0; j<(long long)gny; ++j )
	                    for( long long k=0; k<(long long)gnz; ++k )
	                        buf[(size_t)((i*(long long)gny+j)*(long long)gnz+k)]
	                            = (*slab)((int)i,(int)j,(int)k);
	            MPI_Send(buf.data(), (int)cnt, dtype, 0, tag_base+rk,
	                     MUSIC::mpi::world());
	        }
	        delete slab;
	        if( keep_workers_full ) {
	            // H.4 + spmd_mg_skel: workers keep a zero-filled full union mesh
	            // at this level so SPMD-MG primitives (broadcast_union_mesh_from_owner,
	            // single-box-parent recurse) can read m_pgrids[ilevel] without
	            // NULL deref. The data is irrelevant — those primitives overwrite
	            // it via Bcast/Send. Matches the non-H.4 create_base_hierarchy path
	            // where workers also hold zero-filled union meshes under spmd_mg_skel.
	            MeshvarBnd<T> * full = new MeshvarBnd<T>(
	                m_nbnd, gnx, gny, gnz, 0, 0, 0 );
	            full->zero();
	            m_pgrids[ilevel] = full;
	        } else {
	            m_pgrids[ilevel] = NULL;
	        }
	        m_level_is_slab_[ilevel] = false;
	        m_level_gnx_[ilevel] = 0;
	        m_level_gny_[ilevel] = 0;
	        m_level_gnz_[ilevel] = 0;
	        // Phase E.2.1b: workers had size==0 pre-flip; restore so levelmax()
	        // underflows to UINT_MAX again and downstream code (which only sees
	        // workers' GridHierarchy as an empty shell) is unchanged.
	        // Skip the shrink when keep_workers_full populated the slot — the
	        // SPMD-MG caller wants the slot live.
	        if( !keep_workers_full && m_pgrids.size() == (size_t)ilevel + 1 ) {
	            bool all_null_below = true;
	            for( unsigned k = 0; k < ilevel; ++k )
	                if( m_pgrids[k] != NULL ) { all_null_below = false; break; }
	            if( all_null_below ) {
	                m_pgrids.resize(0);
	                m_xoffabs.resize(0);
	                m_yoffabs.resize(0);
	                m_zoffabs.resize(0);
	            }
	        }
	    }
#else
	    (void)ilevel;
#endif
	}
	//---------------------------------------------------------------------

	//! H.4 C.4: Collective. Gathers a sub-region of the slab parent at level
	//! `ilevel` into out_buf on rank-0, with periodic wrap on each axis.
	//! Replaces gather_union_to_root + coarse pack in fft_interpolate_dist —
	//! avoids materializing the full nbase^3 union just to read a nxc^3
	//! window. Workers pass NULL for out_buf.
	//!
	//! out_buf layout (rank-0): FFTW-padded last dim — out_buf[(i*nyc+j)*nzcp+k]
	//! where nzcp = nzc + 2, for i ∈ [0,nxc), j ∈ [0,nyc), k ∈ [0,nzc).
	//!
	//! Cell semantics: out_buf[(i*nyc+j)*nzcp+k] = V_full_union[
	//!     wrap(oxf+i, gnx), wrap(oyf+j, gny), wrap(ozf+k, gnz) ]
	//! where V_full_union is what gather_union_to_root would have produced.
	//!
	//! Must run OUTSIDE phase_scope.
	void gather_subregion_with_wrap( unsigned ilevel,
	                                  int oxf, int oyf, int ozf,
	                                  size_t nxc_, size_t nyc_, size_t nzc_,
	                                  T* out_buf )
	{
#ifdef USE_MPI
		const int nproc = MUSIC::mpi::size();
		const int rk    = MUSIC::mpi::rank();
		const int nxc = (int)nxc_, nyc = (int)nyc_, nzc = (int)nzc_;
		const size_t nzcp = (size_t)nzc + 2;

		auto wrap = [](int x, size_t N) {
			int n = (int)N;
			x %= n;
			if( x < 0 ) x += n;
			return (size_t)x;
		};

		if( nproc <= 1 ) {
			// serial: just read from the level's MeshvarBnd directly
			MeshvarBnd<T>* V = m_pgrids[ilevel];
			const size_t gnx = (size_t)V->size(0);
			const size_t gny = (size_t)V->size(1);
			const size_t gnz = (size_t)V->size(2);
#pragma omp parallel for
			for( int i = 0; i < nxc; ++i )
				for( int j = 0; j < nyc; ++j )
					for( int k = 0; k < nzc; ++k ) {
						size_t q = ((size_t)i * (size_t)nyc + (size_t)j) * nzcp + (size_t)k;
						out_buf[q] = (*V)( (int)wrap(oxf+i, gnx),
						                   (int)wrap(oyf+j, gny),
						                   (int)wrap(ozf+k, gnz) );
					}
			return;
		}

		if( !is_level_slab(ilevel) )
			throw std::runtime_error("gather_subregion_with_wrap: level is not slab");

		MeshvarBnd<T>* slab = m_pgrids[ilevel];
		const size_t gnx = m_level_gnx_[ilevel];
		const size_t gny = m_level_gny_[ilevel];
		const size_t gnz = m_level_gnz_[ilevel];
		MPI_Datatype dtype = (sizeof(T) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
		const int tag_n   = 2800;
		const int tag_idx = 2801;
		const int tag_buf = 2802;

		// All ranks share slab x-ownership
		long long my_loc[2] = { slab ? (long long)slab->local_x_start() : 0,
		                        slab ? (long long)slab->local_nx()      : 0 };
		std::vector<long long> all_loc(2*nproc, 0);
		MPI_Allgather(my_loc, 2, MPI_LONG_LONG,
		              all_loc.data(), 2, MPI_LONG_LONG, MUSIC::mpi::world());

		auto owner_of_x = [&](long long ix_wrap) -> int {
			for( int r = 0; r < nproc; ++r ) {
				long long s = all_loc[2*r+0];
				long long n = all_loc[2*r+1];
				if( ix_wrap >= s && ix_wrap < s + n ) return r;
			}
			return -1;
		};

		// Collect the i's I own from the [oxf, oxf+nxc) walk
		std::vector<int> my_is;
		my_is.reserve(nxc);
		for( int i = 0; i < nxc; ++i ) {
			long long ix_wrap = ((long long)(oxf + i) % (long long)gnx
			                     + (long long)gnx) % (long long)gnx;
			if( owner_of_x(ix_wrap) == rk ) my_is.push_back(i);
		}
		const size_t plane_size = (size_t)nyc * nzcp;

		if( rk == 0 ) {
			// Pack own i's directly into out_buf
#pragma omp parallel for
			for( int m = 0; m < (int)my_is.size(); ++m ) {
				int i = my_is[m];
				long long ix_wrap = ((long long)(oxf + i) % (long long)gnx
				                     + (long long)gnx) % (long long)gnx;
				long long ix_local = ix_wrap - all_loc[0];
				for( int j = 0; j < nyc; ++j )
					for( int k = 0; k < nzc; ++k ) {
						size_t q = ((size_t)i * (size_t)nyc + (size_t)j) * nzcp + (size_t)k;
						out_buf[q] = (*slab)( (int)ix_local,
						                      (int)wrap(oyf+j, gny),
						                      (int)wrap(ozf+k, gnz) );
					}
			}
			// Receive from each worker
			for( int r = 1; r < nproc; ++r ) {
				int n_my = 0;
				MPI_Recv(&n_my, 1, MPI_INT, r, tag_n, MUSIC::mpi::world(), MPI_STATUS_IGNORE);
				if( n_my == 0 ) continue;
				std::vector<int> i_list(n_my);
				MPI_Recv(i_list.data(), n_my, MPI_INT, r, tag_idx,
				         MUSIC::mpi::world(), MPI_STATUS_IGNORE);
				std::vector<T> rbuf((size_t)n_my * plane_size);
				MPI_Recv(rbuf.data(), (int)((size_t)n_my * plane_size),
				         dtype, r, tag_buf, MUSIC::mpi::world(), MPI_STATUS_IGNORE);
#pragma omp parallel for
				for( int m = 0; m < n_my; ++m ) {
					int i = i_list[m];
					const T* src = rbuf.data() + (size_t)m * plane_size;
					for( int j = 0; j < nyc; ++j )
						for( int k = 0; k < nzc; ++k ) {
							size_t q = ((size_t)i * (size_t)nyc + (size_t)j) * nzcp + (size_t)k;
							out_buf[q] = src[(size_t)j * nzcp + (size_t)k];
						}
				}
			}
		} else {
			int n_my = (int)my_is.size();
			MPI_Send(&n_my, 1, MPI_INT, 0, tag_n, MUSIC::mpi::world());
			if( n_my > 0 ) {
				MPI_Send(my_is.data(), n_my, MPI_INT, 0, tag_idx, MUSIC::mpi::world());
				std::vector<T> sbuf((size_t)n_my * plane_size, T(0));
#pragma omp parallel for
				for( int m = 0; m < n_my; ++m ) {
					int i = my_is[m];
					long long ix_wrap = ((long long)(oxf + i) % (long long)gnx
					                     + (long long)gnx) % (long long)gnx;
					long long ix_local = ix_wrap - my_loc[0];
					T* dst = sbuf.data() + (size_t)m * plane_size;
					for( int j = 0; j < nyc; ++j )
						for( int k = 0; k < nzc; ++k ) {
							dst[(size_t)j * nzcp + (size_t)k] =
							    (*slab)( (int)ix_local,
							             (int)wrap(oyf+j, gny),
							             (int)wrap(ozf+k, gnz) );
						}
				}
				MPI_Send(sbuf.data(), (int)((size_t)n_my * plane_size),
				         dtype, 0, tag_buf, MUSIC::mpi::world());
			}
		}
#else
		(void)ilevel; (void)oxf; (void)oyf; (void)ozf;
		(void)nxc_; (void)nyc_; (void)nzc_; (void)out_buf;
#endif
	}
	//---------------------------------------------------------------------

	//! H.4 C.3 worker cleanup: free the phantom 0..ilevel-1 meshes (allocated
	//! by create_base_hierarchy_slab) plus the now-NULL slot at ilevel, and
	//! resize m_pgrids/m_xoffabs/m_yoffabs/m_zoffabs to 0 so the worker's
	//! GridHierarchy matches the empty-shell state produced by C.2-alt's
	//! delta.deallocate(). Per-box meshes (m_pgrids_per_box_) are untouched.
	//!
	//! Required after C.3's convert_level_slab_to_full because the standard
	//! shed logic inside that function only fires when all_null_below is true,
	//! and phantoms left in 0..levelmin-1 prevent that.
	void shed_levelmin_union_skeleton( unsigned ilevel )
	{
		for( size_t i = 0; i < m_pgrids.size() && i <= (size_t)ilevel; ++i ) {
			if( m_pgrids[i] != NULL ) {
				delete m_pgrids[i];
				m_pgrids[i] = NULL;
			}
		}
		m_pgrids.resize(0);
		m_xoffabs.resize(0);
		m_yoffabs.resize(0);
		m_zoffabs.resize(0);
		if( ilevel < m_level_is_slab_.size() ) m_level_is_slab_[ilevel] = false;
	}
	//---------------------------------------------------------------------

    // meaning of the mask:
    //  -1  =  outside of mask
    //  0.5 =  in mask and refined (i.e. cell exists also on finer level)
    //  1   =  in mask and not refined (i.e. cell exists only on this level)
    
    void add_refinement_mask( const double *shift )
    {
        bhave_refmask = false;
        
        //! generate a mask
        if( m_levelmin != levelmax() )
        {
            for( int ilevel = (int)levelmax(); ilevel >= (int)levelmin(); --ilevel )
            {
                double xq[3], dx = 1.0/(1ul<<ilevel);
                
                m_ref_masks[ilevel]->init( size(ilevel,0), size(ilevel,1), size(ilevel,2), 0 );
                
                for( size_t i=0; i<size(ilevel,0); i+=2 )
                {
                    xq[0] = (offset_abs(ilevel,0) + i)*dx + 0.5*dx + shift[0];
                    for( size_t j=0; j<size(ilevel,1); j+=2 )
                    {
                        xq[1] = (offset_abs(ilevel,1) + j)*dx + 0.5*dx + shift[1];
                        for( size_t k=0; k<size(ilevel,2); k+=2 )
                        {
                            xq[2] = (offset_abs(ilevel,2) + k)*dx + 0.5*dx + shift[2];
                            
                            
                            short mask_val = -1; // outside mask
                            if( the_region_generator->query_point( xq, ilevel ) || ilevel == (int)levelmin() )
                                mask_val = 1; // inside mask
                            
                            (*m_ref_masks[ilevel])(i+0,j+0,k+0) = mask_val;
                            (*m_ref_masks[ilevel])(i+0,j+0,k+1) = mask_val;
                            (*m_ref_masks[ilevel])(i+0,j+1,k+0) = mask_val;
                            (*m_ref_masks[ilevel])(i+0,j+1,k+1) = mask_val;
                            (*m_ref_masks[ilevel])(i+1,j+0,k+0) = mask_val;
                            (*m_ref_masks[ilevel])(i+1,j+0,k+1) = mask_val;
                            (*m_ref_masks[ilevel])(i+1,j+1,k+0) = mask_val;
                            (*m_ref_masks[ilevel])(i+1,j+1,k+1) = mask_val;
                            
                        }
                    }
                }
            }
            
            bhave_refmask = true;
            
            for( int ilevel = (int)levelmin(); ilevel < (int)levelmax(); ++ilevel )
            {
                for( size_t i=0; i<size(ilevel,0); i++ )
                    for( size_t j=0; j<size(ilevel,1); j++ )
                        for( size_t k=0; k<size(ilevel,2); k++ )
                        {
                            bool fine_is_flagged = false;
                            
                            int ifine[] = {
                                2*(int)i-2*(int)offset(ilevel+1,0),
                                2*(int)j-2*(int)offset(ilevel+1,1),
                                2*(int)k-2*(int)offset(ilevel+1,2),
                            };
                            
                            if(ifine[0]>=0 && ifine[0] < (int)size(ilevel+1,0) &&
                               ifine[1]>=0 && ifine[1] < (int)size(ilevel+1,1) &&
                               ifine[2]>=0 && ifine[2] < (int)size(ilevel+1,2) )
                            {
                                fine_is_flagged |= (*m_ref_masks[ilevel+1])(ifine[0]+0,ifine[1]+0,ifine[2]+0)>0;
                                fine_is_flagged |= (*m_ref_masks[ilevel+1])(ifine[0]+0,ifine[1]+0,ifine[2]+1)>0;
                                fine_is_flagged |= (*m_ref_masks[ilevel+1])(ifine[0]+0,ifine[1]+1,ifine[2]+0)>0;
                                fine_is_flagged |= (*m_ref_masks[ilevel+1])(ifine[0]+0,ifine[1]+1,ifine[2]+1)>0;
                                fine_is_flagged |= (*m_ref_masks[ilevel+1])(ifine[0]+1,ifine[1]+0,ifine[2]+0)>0;
                                fine_is_flagged |= (*m_ref_masks[ilevel+1])(ifine[0]+1,ifine[1]+0,ifine[2]+1)>0;
                                fine_is_flagged |= (*m_ref_masks[ilevel+1])(ifine[0]+1,ifine[1]+1,ifine[2]+0)>0;
                                fine_is_flagged |= (*m_ref_masks[ilevel+1])(ifine[0]+1,ifine[1]+1,ifine[2]+1)>0;
                                
                                if( fine_is_flagged )
                                {
                                    (*m_ref_masks[ilevel])(i,j,k) = 2; // cell is refined
                                    
                                    (*m_ref_masks[ilevel+1])(ifine[0]+0,ifine[1]+0,ifine[2]+0) = 1;
                                    (*m_ref_masks[ilevel+1])(ifine[0]+0,ifine[1]+0,ifine[2]+1) = 1;
                                    (*m_ref_masks[ilevel+1])(ifine[0]+0,ifine[1]+1,ifine[2]+0) = 1;
                                    (*m_ref_masks[ilevel+1])(ifine[0]+0,ifine[1]+1,ifine[2]+1) = 1;
                                    (*m_ref_masks[ilevel+1])(ifine[0]+1,ifine[1]+0,ifine[2]+0) = 1;
                                    (*m_ref_masks[ilevel+1])(ifine[0]+1,ifine[1]+0,ifine[2]+1) = 1;
                                    (*m_ref_masks[ilevel+1])(ifine[0]+1,ifine[1]+1,ifine[2]+0) = 1;
                                    (*m_ref_masks[ilevel+1])(ifine[0]+1,ifine[1]+1,ifine[2]+1) = 1;
                                    
                                }
                            }
                        }
            }
        }
    }
	
	
	//! get offset of a grid at specified refinement level
	/*! the offset describes the shift of a refinement grid with respect to its coarser parent grid
	 *  @param ilevel the level for which the offset is to be determined
	 *  @param idim the dimension along which the offset is to be determined
	 *  @return integer value denoting the offset in units of coarse grid cells
	 *  @sa offset_abs
	 */
	int offset( int ilevel, int idim ) const
	{
		return m_pgrids[ilevel]->offset(idim);
	}
	
	//! get size of a grid at specified refinement level
	/*! the size describes the number of cells along one dimension of a grid
	 *  @param ilevel the level for which the size is to be determined
	 *  @param idim the dimension along which the size is to be determined
	 *  @return integer value denoting the size of refinement grid at level ilevel along dimension idim
	 */
	size_t size( int ilevel, int idim ) const
	{
		return m_pgrids[ilevel]->size(idim);
	}
	
	
	//! get the absolute offset of a grid at specified refinement level
	/*! the absolute offset describes the shift of a refinement grid with respect to the simulation volume
	 *  @param ilevel the level for which the offset is to be determined
	 *  @param idim the dimension along which the offset is to be determined
	 *  @return integer value denoting the absolute offset in units of fine grid cells
	 *  @sa offset
	 */
	int offset_abs( int ilevel, int idim ) const
	{
		if( idim == 0 ) return m_xoffabs[ilevel];
		if( idim == 1 ) return m_yoffabs[ilevel];
		return m_zoffabs[ilevel];
	}
	
	
	//! get the coordinate posisition of a grid cell 
	/*! returns the position of a grid cell at specified level relative to the simulation volume
	 *  @param ilevel the refinement level of the grid cell
	 *  @param i the x-index of the cell in the level grid
	 *  @param j the y-index of the cell in the level grid
	 *  @param k the z-index of the cell in the level grid
	 *  @param ppos pointer to a double[3] array to which the coordinates are written
	 *  @return none
	 */
	void cell_pos( unsigned ilevel, int i, int j, int k, double* ppos ) const
	{
		double h = 1.0/(1<<ilevel);//, htop = h*2.0;
		ppos[0] = h*((double)offset_abs(ilevel,0)+(double)i+0.5);
		ppos[1] = h*((double)offset_abs(ilevel,1)+(double)j+0.5);
		ppos[2] = h*((double)offset_abs(ilevel,2)+(double)k+0.5);
		
		if( ppos[0] >= 1.0 || ppos[1] >= 1.0 || ppos[2] >= 1.0 )
			std::cerr << " - Cell seems outside domain! : (" << ppos[0] << ", " << ppos[1] << ", " << ppos[2] << "\n";
	}
    
    
    //! get the bounding box of a grid in code units
    /*! returns the bounding box of a grid at specified level in code units
     *  @param ilevel the refinement level of the grid
     *  @param left pointer to a double[3] array to which the left corner is written
     *  @param right pointer to a double[3] array to which the right corner is written
     *  @return none
     */
    void grid_bbox( unsigned ilevel, double *left, double *right ) const
    {
        double h = 1.0/(1<<ilevel);//, htop = h*2.0;
		left[0] = h*((double)offset_abs(ilevel,0));
		left[1] = h*((double)offset_abs(ilevel,1));
		left[2] = h*((double)offset_abs(ilevel,2));
        
        right[0] = left[0] + h*((double)size(ilevel,0));
        right[1] = left[1] + h*((double)size(ilevel,1));
        right[2] = left[2] + h*((double)size(ilevel,2));
    }
	
	//! checks whether a given grid cell is refined
	/*! a grid cell counts as refined if it is divided into 8 cells at the next higher level
	 *  @param ilevel the refinement level of the grid cell
	 *  @param i the x-index of the cell in the level grid
	 *  @param j the y-index of the cell in the level grid
	 *  @param k the z-index of the cell in the level grid
	 *  @return true if cell is refined, false otherwise
	 */
	bool is_refined( unsigned ilevel, int i, int j, int k ) const
	{
        // meaning of the mask:
        //  -1  =  outside of mask
        //  2 =  in mask and refined (i.e. cell exists also on finer level)
        //  1   =  in mask and not refined (i.e. cell exists only on this level)

        
        

        if( bhave_refmask ){
            return (*m_ref_masks[ilevel])(i,j,k)==2;
        }
        
        if( !bhave_refmask && ilevel == levelmax() )
            return false;
		
		if( i < offset(ilevel+1,0) || i >= offset(ilevel+1, 0)+(int)size(ilevel+1,0)/2 ||
		    j < offset(ilevel+1,1) || j >= offset(ilevel+1, 1)+(int)size(ilevel+1,1)/2 ||
		    k < offset(ilevel+1,2) || k >= offset(ilevel+1, 2)+(int)size(ilevel+1,2)/2 )
			return false;
		
		return true;
	}
    
    bool is_in_mask( unsigned ilevel, int i, int j, int k ) const
	{
        // meaning of the mask:
        //  -1  =  outside of mask
        //  2 =  in mask and refined (i.e. cell exists also on finer level)
        //  1   =  in mask and not refined (i.e. cell exists only on this level)
        

        if( bhave_refmask ){
            return ((*m_ref_masks[ilevel])(i,j,k)>=0);
        }
        
        return true;
    }
	
	//! sets the values of all grids on all levels to zero
	void zero( void )
	{
		for( unsigned i=0; i<m_pgrids.size(); ++i )
			m_pgrids[i]->zero();
		// Phase D.3.2: also clear per-box meshes so the per-cluster
		// V-cycle starts from u=0 (matching the union path).
		for( size_t L=0; L<m_pgrids_per_box_.size(); ++L )
			for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b )
				if( m_pgrids_per_box_[L][b] )
					m_pgrids_per_box_[L][b]->zero();
	}
	
	
	//! count the number of cells that are not further refined (=leafs)
	/*! for allocation purposes it is useful to query the number of cells to be expected
	 *  @param lmin the minimum refinement level to consider
	 *  @param lmax the maximum refinement level to consider
	 *  @return the integer number of cells between lmin and lmax that are not further refined
	 */
	size_t count_leaf_cells( unsigned lmin, unsigned lmax ) const
	{
		size_t npcount = 0;
		
		for( int ilevel=lmax; ilevel>=(int)lmin; --ilevel )
			for( unsigned i=0; i<get_grid(ilevel)->size(0); ++i )
				for( unsigned j=0; j<get_grid(ilevel)->size(1); ++j )
					for( unsigned k=0; k<get_grid(ilevel)->size(2); ++k )
						if( is_in_mask(ilevel,i,j,k) && !is_refined(ilevel,i,j,k) )
                            ++npcount;
		
		return npcount;
	}
	
	//! count the number of cells that are not further refined (=leafs)
	/*! for allocation purposes it is useful to query the number of cells to be expected
	 *  @return the integer number of cells in the hierarchy that are not further refined
	 */
	size_t count_leaf_cells( void ) const
	{
		return count_leaf_cells( levelmin(), levelmax() );
	}
	
	//! creates a hierarchy of coextensive grids, refined by factors of 2
	/*! creates a hierarchy of lmax grids, each extending over the whole simulation volume with
	 *  grid length 2^n for level 0<=n<=lmax
	 *  @param lmax the maximum refinement level to be added (sets the resolution to 2^lmax for each dim)
	 *  @return none
	 */
	void create_base_hierarchy( unsigned lmax )
	{
		size_t n=1;

		this->deallocate();

		m_pgrids.clear();

		m_xoffabs.clear();
		m_yoffabs.clear();
		m_zoffabs.clear();

		for( unsigned i=0; i<= lmax; ++i )
		{
			//std::cout << "....adding level " << i << " (" << n << ", " << n << ", " << n << ")" << std::endl;
			m_pgrids.push_back( new MeshvarBnd<T>( m_nbnd, n, n, n, 0, 0, 0 ) );
			m_pgrids[i]->zero();
     		n *= 2;

			m_xoffabs.push_back( 0 );
			m_yoffabs.push_back( 0 );
			m_zoffabs.push_back( 0 );
		}

		m_levelmin = lmax;

        for( unsigned i=0; i<= lmax; ++i )
            m_ref_masks.push_back( new refinement_mask(size(i,0),size(i,1),size(i,2),(short)(i!=lmax)) );
	}

	//! H.4: like create_base_hierarchy(lmax), but the topmost level (lmax) is
	//! allocated as a per-rank slab MeshvarBnd instead of a full mesh on every
	//! rank. Smaller levels 0..lmax-1 stay full (sizes 1,2,...,2^(lmax-1) are
	//! negligible). Caller is responsible for zeroing the slab (slab ctor
	//! already zero-inits; this mirrors create_base_hierarchy which calls
	//! ->zero() explicitly).
	//!
	//! Collective. On serial builds (np==1) the slab degenerates to the full
	//! mesh, so this is observationally equivalent to create_base_hierarchy.
	void create_base_hierarchy_slab( unsigned lmax )
	{
		size_t n=1;

		this->deallocate();

		m_pgrids.clear();

		m_xoffabs.clear();
		m_yoffabs.clear();
		m_zoffabs.clear();

		for( unsigned i=0; i< lmax; ++i )
		{
			m_pgrids.push_back( new MeshvarBnd<T>( m_nbnd, n, n, n, 0, 0, 0 ) );
			m_pgrids[i]->zero();
			n *= 2;
			m_xoffabs.push_back( 0 );
			m_yoffabs.push_back( 0 );
			m_zoffabs.push_back( 0 );
		}

		// Top level as per-rank slab. allocate_union_slab_at requires the
		// m_pgrids slot to exist and the offset vectors to be sized to lmax+1.
		m_pgrids.push_back( (MeshvarBnd<T>*)NULL );
		m_xoffabs.push_back( 0 );
		m_yoffabs.push_back( 0 );
		m_zoffabs.push_back( 0 );
		ensure_level_slab_meta_(lmax);
		allocate_union_slab_at( lmax, n, n, n, 0, 0, 0 );
		m_pgrids[lmax]->zero();

		m_levelmin = lmax;

		for( unsigned i=0; i<= lmax; ++i )
			m_ref_masks.push_back( new refinement_mask(size(i,0),size(i,1),size(i,2),(short)(i!=lmax)) );
	}
	
	//! multiply entire grid hierarchy by a constant
	GridHierarchy<T>& operator*=( T x )
	{
		for( unsigned i=0; i<m_pgrids.size(); ++i )
			(*m_pgrids[i]) *= x;
		// D.3.3 proper: per-box meshes are peer storage at multi-box levels.
		// E.1: skip NULL non-owner slots.
		for( size_t L=0; L<m_pgrids_per_box_.size(); ++L )
			for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b )
				if( m_pgrids_per_box_[L][b] )
					(*m_pgrids_per_box_[L][b]) *= x;
		return *this;
	}

	//! divide entire grid hierarchy by a constant
	GridHierarchy<T>& operator/=( T x )
	{
		for( unsigned i=0; i<m_pgrids.size(); ++i )
			(*m_pgrids[i]) /= x;
		for( size_t L=0; L<m_pgrids_per_box_.size(); ++L )
			for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b )
				if( m_pgrids_per_box_[L][b] )
					(*m_pgrids_per_box_[L][b]) /= x;
		return *this;
	}

	//! add a constant to the entire grid hierarchy
	GridHierarchy<T>& operator+=( T x )
	{
		for( unsigned i=0; i<m_pgrids.size(); ++i )
			(*m_pgrids[i]) += x;
		for( size_t L=0; L<m_pgrids_per_box_.size(); ++L )
			for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b )
				if( m_pgrids_per_box_[L][b] )
					(*m_pgrids_per_box_[L][b]) += x;
		return *this;
	}

	//! subtract a constant from the entire grid hierarchy
	GridHierarchy<T>& operator-=( T x )
	{
		for( unsigned i=0; i<m_pgrids.size(); ++i )
			(*m_pgrids[i]) -= x;
		for( size_t L=0; L<m_pgrids_per_box_.size(); ++L )
			for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b )
				if( m_pgrids_per_box_[L][b] )
					(*m_pgrids_per_box_[L][b]) -= x;
		return *this;
	}

	//! multiply (element-wise) two grid hierarchies
	GridHierarchy<T>& operator*=( const GridHierarchy<T>& gh )
	{
		if( !is_consistent(gh) )
		{
            LOGERR("GridHierarchy::operator*= : attempt to operate on incompatible data");
            throw std::runtime_error("GridHierarchy::operator*= : attempt to operate on incompatible data");
		}
		for( unsigned i=0; i<m_pgrids.size(); ++i )
			(*m_pgrids[i]) *= *gh.get_grid(i);
		// E.1: paired iteration must skip slots where either side is NULL
		// (owner mismatch between two hierarchies is treated as no-op for
		// that box on this rank).
		for( size_t L=0; L<m_pgrids_per_box_.size(); ++L )
			if( L < gh.m_pgrids_per_box_.size()
				&& gh.m_pgrids_per_box_[L].size() == m_pgrids_per_box_[L].size() )
				for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b )
					if( m_pgrids_per_box_[L][b] && gh.m_pgrids_per_box_[L][b] )
						(*m_pgrids_per_box_[L][b]) *= *gh.m_pgrids_per_box_[L][b];
		return *this;
	}

	//! divide (element-wise) two grid hierarchies
	GridHierarchy<T>& operator/=( const GridHierarchy<T>& gh )
	{
		if( !is_consistent(gh) )
		{
            LOGERR("GridHierarchy::operator/= : attempt to operate on incompatible data");
            throw std::runtime_error("GridHierarchy::operator/= : attempt to operate on incompatible data");
		}
		for( unsigned i=0; i<m_pgrids.size(); ++i )
			(*m_pgrids[i]) /= *gh.get_grid(i);
		for( size_t L=0; L<m_pgrids_per_box_.size(); ++L )
			if( L < gh.m_pgrids_per_box_.size()
				&& gh.m_pgrids_per_box_[L].size() == m_pgrids_per_box_[L].size() )
				for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b )
					if( m_pgrids_per_box_[L][b] && gh.m_pgrids_per_box_[L][b] )
						(*m_pgrids_per_box_[L][b]) /= *gh.m_pgrids_per_box_[L][b];
		return *this;
	}

	//! add (element-wise) two grid hierarchies
	GridHierarchy<T>& operator+=( const GridHierarchy<T>& gh )
	{
		if( !is_consistent(gh) )
			throw std::runtime_error("GridHierarchy::operator+= : attempt to operate on incompatible data");

		for( unsigned i=0; i<m_pgrids.size(); ++i )
			(*m_pgrids[i]) += *gh.get_grid(i);
		for( size_t L=0; L<m_pgrids_per_box_.size(); ++L )
			if( L < gh.m_pgrids_per_box_.size()
				&& gh.m_pgrids_per_box_[L].size() == m_pgrids_per_box_[L].size() )
				for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b )
					if( m_pgrids_per_box_[L][b] && gh.m_pgrids_per_box_[L][b] )
						(*m_pgrids_per_box_[L][b]) += *gh.m_pgrids_per_box_[L][b];
		return *this;
	}

	//! subtract (element-wise) two grid hierarchies
	GridHierarchy<T>& operator-=( const GridHierarchy<T>& gh )
	{
		if( !is_consistent(gh) )
		{
            LOGERR("GridHierarchy::operator-= : attempt to operate on incompatible data");
            throw std::runtime_error("GridHierarchy::operator-= : attempt to operate on incompatible data");
		}
		for( unsigned i=0; i<m_pgrids.size(); ++i )
			(*m_pgrids[i]) -= *gh.get_grid(i);
		for( size_t L=0; L<m_pgrids_per_box_.size(); ++L )
			if( L < gh.m_pgrids_per_box_.size()
				&& gh.m_pgrids_per_box_[L].size() == m_pgrids_per_box_[L].size() )
				for( size_t b=0; b<m_pgrids_per_box_[L].size(); ++b )
					if( m_pgrids_per_box_[L][b] && gh.m_pgrids_per_box_[L][b] )
						(*m_pgrids_per_box_[L][b]) -= *gh.m_pgrids_per_box_[L][b];
		return *this;
	}
	
	//! assign (element-wise) two grid hierarchies
	GridHierarchy<T>& operator=( const GridHierarchy<T>& gh )
	{
        bhave_refmask = gh.bhave_refmask;

        if( bhave_refmask )
        {
            for( size_t i=0; i<gh.m_ref_masks.size(); ++i )
                m_ref_masks.push_back( new refinement_mask( *(gh.m_ref_masks[i]) ) );
        }

		if( !is_consistent(gh) )
		{
			for( unsigned i=0; i<m_pgrids.size(); ++i )
				delete m_pgrids[i];
			m_pgrids.clear();

			// see copy-ctor note: iterate by size() to handle empty source.
			// H.4: skip NULL slots (workers in mixed shell w/ m_pgrids[L]==NULL).
			for( size_t i=0; i<gh.m_pgrids.size(); ++i )
				m_pgrids.push_back( gh.m_pgrids[i]
				    ? new MeshvarBnd<T>( *gh.m_pgrids[i] )
				    : (MeshvarBnd<T>*)NULL );
			m_levelmin = gh.levelmin();
			m_nbnd = gh.m_nbnd;

			m_xoffabs = gh.m_xoffabs;
			m_yoffabs = gh.m_yoffabs;
			m_zoffabs = gh.m_zoffabs;

			// Per-box meshes (Phase D.2b / E.1) — deep copy with replace.
			deallocate_per_box_meshes();
			m_pgrids_per_box_.assign( gh.m_pgrids_per_box_.size(), std::vector<MeshvarBnd<T>*>() );
			for( size_t L=0; L<gh.m_pgrids_per_box_.size(); ++L )
				for( size_t b=0; b<gh.m_pgrids_per_box_[L].size(); ++b )
					m_pgrids_per_box_[L].push_back(
						gh.m_pgrids_per_box_[L][b]
							? new MeshvarBnd<T>( *gh.m_pgrids_per_box_[L][b] )
							: NULL );
			m_xoffabs_per_box_    = gh.m_xoffabs_per_box_;
			m_yoffabs_per_box_    = gh.m_yoffabs_per_box_;
			m_zoffabs_per_box_    = gh.m_zoffabs_per_box_;
			m_parent_idx_per_box_ = gh.m_parent_idx_per_box_;
			m_pbox_owner_         = gh.m_pbox_owner_;
			m_nx_per_box_         = gh.m_nx_per_box_;
			m_ny_per_box_         = gh.m_ny_per_box_;
			m_nz_per_box_         = gh.m_nz_per_box_;
			m_oxrel_per_box_      = gh.m_oxrel_per_box_;
			m_oyrel_per_box_      = gh.m_oyrel_per_box_;
			m_ozrel_per_box_      = gh.m_ozrel_per_box_;
			m_pbox_tenant_        = gh.m_pbox_tenant_;

			// Phase E.2.0: replicate slab metadata; tenants are transient.
			m_level_is_slab_   = gh.m_level_is_slab_;
			m_level_gnx_       = gh.m_level_gnx_;
			m_level_gny_       = gh.m_level_gny_;
			m_level_gnz_       = gh.m_level_gnz_;
			m_level_tenant_.assign(m_level_is_slab_.size(), false);
			m_level_slab_backup_.assign(m_level_is_slab_.size(), (MeshvarBnd<T>*)NULL);

			return *this;
		}//throw std::runtime_error("GridHierarchy::operator= : attempt to operate on incompatible data");

		// H.4: workers may carry NULL slots (m_pgrids[L]==NULL after
		// convert_level_slab_to_full); skip those rather than deref.
		for( unsigned i=0; i<m_pgrids.size(); ++i )
			if( m_pgrids[i] && gh.get_grid(i) )
				(*m_pgrids[i]) = *gh.get_grid(i);

		// Per-box meshes — element-wise assign if sizes match (consistent path).
		// E.1: skip NULL slots (owner mismatch).
		for( size_t L=0; L<m_pgrids_per_box_.size() && L<gh.m_pgrids_per_box_.size(); ++L )
			for( size_t b=0; b<m_pgrids_per_box_[L].size() && b<gh.m_pgrids_per_box_[L].size(); ++b )
				if( m_pgrids_per_box_[L][b] && gh.m_pgrids_per_box_[L][b] )
					(*m_pgrids_per_box_[L][b]) = *gh.m_pgrids_per_box_[L][b];

		return *this;
	}
	
	/*
	//! assignment operator
	GridHierarchy& operator=( const GridHierarchy<T>& gh )
	{
		for( unsigned i=0; i<m_pgrids.size(); ++i )
			delete m_pgrids[i];
		m_pgrids.clear();
		
		for( unsigned i=0; i<=gh.levelmax(); ++i )
			m_pgrids.push_back( new MeshvarBnd<T>( *gh.get_grid(i) ) );
		m_levelmin = gh.levelmin();
		m_nbnd = gh.m_nbnd;
		
		m_xoffabs = gh.m_xoffabs;
		m_yoffabs = gh.m_yoffabs;
		m_zoffabs = gh.m_zoffabs;
		
		
		return *this;
	}
	*/
	
	/*! add a new refinement patch to the so-far finest level
	 * @param xoff x-offset in units of the coarser grid (finest grid before adding new patch)
	 * @param yoff y-offset in units of the coarser grid (finest grid before adding new patch)
	 * @param zoff z-offset in units of the coarser grid (finest grid before adding new patch)
	 * @param nx x-extent in fine grid cells
	 * @param ny y-extent in fine grid cells
	 * @param nz z-extent in fine grid cells
	 */
	void add_patch( unsigned xoff, unsigned yoff, unsigned zoff, unsigned nx, unsigned ny, unsigned nz )
	{
		m_pgrids.push_back( new MeshvarBnd<T>( m_nbnd, nx, ny, nz, xoff, yoff, zoff ) );
		m_pgrids.back()->zero();
		
		//.. add absolute offsets (in units of current level grid cells)
		m_xoffabs.push_back( 2*(m_xoffabs.back() + xoff) );
		m_yoffabs.push_back( 2*(m_yoffabs.back() + yoff) );
		m_zoffabs.push_back( 2*(m_zoffabs.back() + zoff) );
        
        m_ref_masks.push_back( new refinement_mask(nx,ny,nz,0) );
	}
	
	/*! cut a refinement patch to the specified size
	 * @param ilevel grid level on which to perform the size adjustment
	 * @param xoff new x-offset in units of the coarser grid (finest grid before adding new patch)
	 * @param yoff new y-offset in units of the coarser grid (finest grid before adding new patch)
	 * @param zoff new z-offset in units of the coarser grid (finest grid before adding new patch)
	 * @param nx new x-extent in fine grid cells
	 * @param ny new y-extent in fine grid cells
	 * @param nz new z-extent in fine grid cells
	 */
	void cut_patch( unsigned ilevel, unsigned xoff, unsigned yoff, unsigned zoff, unsigned nx, unsigned ny, unsigned nz)
	{
		unsigned dx,dy,dz,dxtop,dytop,dztop;
		
		dx = xoff-m_xoffabs[ilevel];
		dy = yoff-m_yoffabs[ilevel];
		dz = zoff-m_zoffabs[ilevel];

		assert( dx%2==0 && dy%2==0 && dz%2==0 );
		
		dxtop = m_pgrids[ilevel]->offset(0)+dx/2;
		dytop = m_pgrids[ilevel]->offset(1)+dy/2;
		dztop = m_pgrids[ilevel]->offset(2)+dz/2;
		
		MeshvarBnd<T> *mnew = new MeshvarBnd<T>( m_nbnd, nx, ny, nz, dxtop, dytop, dztop );
		
		//... copy data
		for( unsigned i=0; i<nx; ++i )
			for( unsigned j=0; j<ny; ++j )
				for( unsigned k=0; k<nz; ++k )
					(*mnew)(i,j,k) = (*m_pgrids[ilevel])(i+dx,j+dy,k+dz);

		//... replace in hierarchy
		delete m_pgrids[ilevel];
		m_pgrids[ilevel] = mnew;
		
		//... update offsets
		m_xoffabs[ilevel] += dx;
		m_yoffabs[ilevel] += dy;
		m_zoffabs[ilevel] += dz;
		
		if( ilevel < levelmax() )
		{
			m_pgrids[ilevel+1]->offset(0) -= dx;
			m_pgrids[ilevel+1]->offset(1) -= dy;
			m_pgrids[ilevel+1]->offset(2) -= dz;
		}
		
		find_new_levelmin();
	}

  void cut_patch_enforce_top_density( unsigned ilevel, unsigned xoff, unsigned yoff, unsigned zoff, unsigned nx, unsigned ny, unsigned nz)
  {
    unsigned dx,dy,dz,dxtop,dytop,dztop;
		
    dx = xoff-m_xoffabs[ilevel];
    dy = yoff-m_yoffabs[ilevel];
    dz = zoff-m_zoffabs[ilevel];
    
    assert( dx%2==0 && dy%2==0 && dz%2==0 );
    
    dxtop = m_pgrids[ilevel]->offset(0)+dx/2;
    dytop = m_pgrids[ilevel]->offset(1)+dy/2;
    dztop = m_pgrids[ilevel]->offset(2)+dz/2;
    
    MeshvarBnd<T> *mnew = new MeshvarBnd<T>( m_nbnd, nx, ny, nz, dxtop, dytop, dztop );
    
    double coarsesum = 0.0, finesum = 0.0;
    size_t coarsecount = 0, finecount = 0;

    //... copy data
    for( unsigned i=0; i<nx; ++i )
      for( unsigned j=0; j<ny; ++j )
	for( unsigned k=0; k<nz; ++k ){
	  (*mnew)(i,j,k) = (*m_pgrids[ilevel])(i+dx,j+dy,k+dz);
	  finesum += (*mnew)(i,j,k);
	  finecount++;
	}

    //... replace in hierarchy
    delete m_pgrids[ilevel];
    m_pgrids[ilevel] = mnew;
    
    //... update offsets
    m_xoffabs[ilevel] += dx;
    m_yoffabs[ilevel] += dy;
    m_zoffabs[ilevel] += dz;
    
    if( ilevel < levelmax() )
      {
	m_pgrids[ilevel+1]->offset(0) -= dx;
	m_pgrids[ilevel+1]->offset(1) -= dy;
	m_pgrids[ilevel+1]->offset(2) -= dz;
      }

    //... enforce top mean density over same patch
    if( ilevel > levelmin() )
      {
	int ox = m_pgrids[ilevel]->offset(0);
	int oy = m_pgrids[ilevel]->offset(1);
	int oz = m_pgrids[ilevel]->offset(2);

	for( unsigned i=0; i<nx/2; ++i )
	  for( unsigned j=0; j<ny/2; ++j )
	    for( unsigned k=0; k<nz/2; ++k ){
	      coarsesum += (*m_pgrids[ilevel-1])(i+ox,j+oy,k+oz);
	      coarsecount++;
	    }

	coarsesum /= (double)coarsecount;
	finesum /= (double)finecount;

	for( unsigned i=0; i<nx; ++i )
	  for( unsigned j=0; j<ny; ++j )
	    for( unsigned k=0; k<nz; ++k )
	      (*mnew)(i,j,k) += (coarsesum-finesum);

	LOGINFO("level %d : corrected patch overlap mean density by %f",ilevel,coarsesum-finesum);
      }

    
    find_new_levelmin();
  }
  
	/*! determine level for which grid extends over entire domain */
	void find_new_levelmin( void )
	{
		for( unsigned i=0; i<=levelmax(); ++i )
		{
			unsigned n = 1<<i;
			if(	m_pgrids[i]->size(0) == n &&
			   	m_pgrids[i]->size(1) == n &&
			   	m_pgrids[i]->size(2) == n )
			{
				m_levelmin=i;
			}
		}
	}
	
	//! return maximum level in refinement hierarchy
	unsigned levelmax( void ) const
	{
		return m_pgrids.size()-1;
	}
	
	//! return minimum level in refinement hierarchy (the level which extends over the entire domain)
	unsigned levelmin( void ) const
	{
		return m_levelmin;
	}
	
};



//! class that computes the refinement structure given parameters
class refinement_hierarchy
{
	std::vector<double>
		x0_,	//!< x-coordinates of grid origins (in [0..1[)
		y0_,	//!< y-coordinates of grid origins (in [0..1[)
		z0_,	//!< z-coordinates of grid origins (in [0..1[)
		xl_,	//!< x-extent of grids (in [0..1[)
		yl_,	//!< y-extent of grids (in [0..1[)
		zl_;	//!< z-extent of grids (in [0..1[)

	std::vector<unsigned>
		ox_,	//!< relative x-coordinates of grid origins (in coarser grid cells)
		oy_,	//!< relative y-coordinates of grid origins (in coarser grid cells)
		oz_,	//!< relative z-coordinates of grid origins (in coarser grid cells)
		oax_,	//!< absolute x-coordinates of grid origins (in fine grid cells)
		oay_,	//!< absolute y-coordinates of grid origins (in fine grid cells)
		oaz_,	//!< absolute z-coordinates of grid origins (in fine grid cells)
		nx_,	//!< x-extent of grids (in fine grid cells)
		ny_,	//!< y-extent of grids (in fine grid cells)
		nz_;	//!< z-extent of grids (in fine grid cells)

	//! Per-level list of disjoint refinement boxes (multibox Phase D.2 bookkeeping).
	//! For single-box plugins (or single connected component), level_boxes_[L] has one
	//! entry equal to the union arrays at level L — so legacy compute paths stay
	//! bit-identical. For region=multibox with N>1 clusters, level_boxes_[L] holds the
	//! N disjoint cluster sub-AABBs (post align/pad/blocking). Compute paths still
	//! consume the union arrays in D.2a; per-box GridHierarchy meshes / per-box
	//! Poisson V-cycle are wired up in D.2b / D.3.
	std::vector< std::vector<LevelBox> > level_boxes_;
	
	unsigned 
		levelmin_,		//!< minimum grid level for Poisson solver
		levelmax_,		//!< maximum grid level for all operations
		levelmin_tf_,	//!< minimum grid level for density calculation
		padding_,		//!< padding in number of coarse cells between refinement levels
		blocking_factor_;
	
	
	config_file& cf_;	//!< reference to config_file
	
	bool align_top_,	//!< bool whether to align all grids with coarsest grid cells
	     equal_extent_; //!< bool whether the simulation code requires squared refinement regions (e.g. RAMSES)
	
	double 
		x0ref_[3],	//!< coordinates of refinement region origin (in [0..1[)
		lxref_[3];	//!< extent of refinement region (int [0..1[)

        size_t lnref_[3];
  bool   bhave_nref;
	
	int xshift_[3];	//!< shift of refinement region in coarse cells (in order to center it in the domain)
    double rshift_[3];
	
public:
	
	//! copy constructor
	refinement_hierarchy( const refinement_hierarchy& rh )
	: cf_( rh.cf_ )
	{
		*this = rh;
	}
	
	//! constructor from a config_file holding information about the desired refinement geometry
	explicit refinement_hierarchy( config_file& cf )
	: cf_( cf )
	{
	  //... query the parameter data we need
	  levelmin_	= cf_.getValue<unsigned>("setup","levelmin");
	  levelmax_	= cf_.getValue<unsigned>("setup","levelmax");
	  levelmin_tf_= cf_.getValueSafe<unsigned>("setup","levelmin_TF",levelmin_);
	  align_top_	= cf_.getValueSafe<bool>("setup","align_top",false);
	  equal_extent_ = cf_.getValueSafe<bool>("setup","force_equal_extent",false);
	  blocking_factor_= cf.getValueSafe<unsigned>( "setup", "blocking_factor",0);
		
	  bool bnoshift = cf_.getValueSafe<bool>("setup","no_shift",false);
	  bool force_shift = cf_.getValueSafe<bool>("setup","force_shift",false);
	  
        
	  //... call the region generator
	  if( levelmin_ != levelmax_ )
	    {
	      
	      double x1ref[3];
	      the_region_generator->get_AABB(x0ref_,x1ref,levelmax_);
	      for( int i=0; i<3; ++i )
		lxref_[i] = x1ref[i]-x0ref_[i];
	      bhave_nref = false;
	      
	      std::string region_type = cf.getValueSafe<std::string>("setup","region","box");
	      
	      LOGINFO("refinement region is \'%s\', w/ bounding box\n        left = [%f,%f,%f]\n       right = [%f,%f,%f]",
		      region_type.c_str(),x0ref_[0],x0ref_[1],x0ref_[2],x1ref[0],x1ref[1],x1ref[2]);

	      // Diagnostic: report per-level disjoint refinement boxes as seen by the region
	      // generator. Most plugins return 1 (single union bbox); region=multibox exposes
	      // the true connected-component count after Phase D.1 of the multibox refactor.
	      // Mesh allocation still consumes the union bbox here — disjoint sub-meshes will
	      // be plumbed through in Phase D.2.
	      for( unsigned L = levelmin_+1; L <= levelmax_; ++L )
	      {
	          size_t nb = the_region_generator->get_num_boxes(L);
	          if( nb > 1 )
	          {
	              LOGINFO("Level %u: %zu disjoint refinement box(es)", L, nb);
	              for( size_t b = 0; b < nb; ++b )
	              {
	                  double l[3], r[3];
	                  the_region_generator->get_AABB_box(l, r, L, b);
	                  LOGINFO("  box %zu: [%.4f,%.4f] x [%.4f,%.4f] x [%.4f,%.4f]",
	                          b, l[0], r[0], l[1], r[1], l[2], r[2]);
	              }
	          }
	      }

	      bhave_nref = the_region_generator->is_grid_dim_forced( lnref_ );
	    }
	  
	  
	  // if not doing any refinement levels, set extent to full box
	  if( levelmin_ == levelmax_ )
	    {
	      x0ref_[0] = 0.0;
	      x0ref_[1] = 0.0;
	      x0ref_[2] = 0.0;
	      
	      lxref_[0] = 1.0;
	      lxref_[1] = 1.0;
	      lxref_[2] = 1.0;
	    }
		
	  unsigned ncoarse = 1<<levelmin_;
	  
	  //... determine shift
	  
	  double xc[3];
	  xc[0] = fmod(x0ref_[0]+0.5*lxref_[0],1.0);
	  xc[1] = fmod(x0ref_[1]+0.5*lxref_[1],1.0);
	  xc[2] = fmod(x0ref_[2]+0.5*lxref_[2],1.0);
	  
	  if( (levelmin_ != levelmax_) && (!bnoshift || force_shift) )
	    {
	      xshift_[0] = (int)((0.5-xc[0])*ncoarse);
	      xshift_[1] = (int)((0.5-xc[1])*ncoarse);
	      xshift_[2] = (int)((0.5-xc[2])*ncoarse);
	    }else{
	    xshift_[0] = 0;
	    xshift_[1] = 0;
	    xshift_[2] = 0;
	  }
	  
	  char strtmp[32];
	  sprintf( strtmp, "%d", xshift_[0] );	cf_.insertValue( "setup", "shift_x", strtmp );
	  sprintf( strtmp, "%d", xshift_[1] );	cf_.insertValue( "setup", "shift_y", strtmp );
	  sprintf( strtmp, "%d", xshift_[2] );	cf_.insertValue( "setup", "shift_z", strtmp );
	  
	  rshift_[0] = -(double)xshift_[0]/ncoarse;
	  rshift_[1] = -(double)xshift_[1]/ncoarse;
	  rshift_[2] = -(double)xshift_[2]/ncoarse;
	  
	  x0ref_[0] += (double)xshift_[0]/ncoarse;
	  x0ref_[1] += (double)xshift_[1]/ncoarse;
	  x0ref_[2] += (double)xshift_[2]/ncoarse;
	  
	  
	  //... initialize arrays 
	  x0_.assign(levelmax_+1,0.0);	xl_.assign(levelmax_+1,1.0);
	  y0_.assign(levelmax_+1,0.0);	yl_.assign(levelmax_+1,1.0);
	  z0_.assign(levelmax_+1,0.0);	zl_.assign(levelmax_+1,1.0);
	  ox_.assign(levelmax_+1,0);	nx_.assign(levelmax_+1,0);
	  oy_.assign(levelmax_+1,0);	ny_.assign(levelmax_+1,0);
	  oz_.assign(levelmax_+1,0);	nz_.assign(levelmax_+1,0);
	  
	  oax_.assign(levelmax_+1,0);
	  oay_.assign(levelmax_+1,0);
	  oaz_.assign(levelmax_+1,0);
	  
		
	  nx_[levelmin_] = ncoarse;
	  ny_[levelmin_] = ncoarse;
	  nz_[levelmin_] = ncoarse;
	  
	  // set up base hierarchy sizes
	  for( unsigned ilevel=0; ilevel <=levelmin_; ++ilevel )
	    {
	      unsigned n = 1<<ilevel;
	      
	      xl_[ilevel] = yl_[ilevel] = zl_[ilevel] = 1.0;
	      nx_[ilevel] = ny_[ilevel] = nz_[ilevel] = n;
	    }
	  
	  // if no refinement, we can exit here
	  if( levelmax_ == levelmin_ )
            return;
	  
		
	  //... determine the position of the refinement region on the finest grid
	  int il,jl,kl,ir,jr,kr;
	  int nresmax = 1<<levelmax_;
	  
	  il = (int)(x0ref_[0] * nresmax);
	  jl = (int)(x0ref_[1] * nresmax);
	  kl = (int)(x0ref_[2] * nresmax);
	  ir = (int)((x0ref_[0]+lxref_[0]) * nresmax );//+ 1.0);
	  jr = (int)((x0ref_[1]+lxref_[1]) * nresmax );//+ 1.0);
	  kr = (int)((x0ref_[2]+lxref_[2]) * nresmax );//+ 1.0);
	  
	  //... align with coarser grids ...
	  if( align_top_ )
	    {
	      //... require alignment with top grid
	      unsigned nref = 1<<(levelmax_-levelmin_+1);
	      
	      if( bhave_nref )
		{
		  if( lnref_[0] % (1ul<<(levelmax_-levelmin_)) != 0 ||
		      lnref_[1] % (1ul<<(levelmax_-levelmin_)) != 0 ||
		      lnref_[2] % (1ul<<(levelmax_-levelmin_)) != 0 )
		    {
		      LOGERR("specified ref_dims and align_top=yes but cannot be aligned with coarse grid!");
		      throw std::runtime_error("specified ref_dims and align_top=yes but cannot be aligned with coarse grid!");
		    }
		}
	      
	      
	      il = (int)((double)il/nref)*nref;
	      jl = (int)((double)jl/nref)*nref;
	      kl = (int)((double)kl/nref)*nref;
	      
	      int irr = (int)((double)ir/nref)*nref;
	      int jrr = (int)((double)jr/nref)*nref;
	      int krr = (int)((double)kr/nref)*nref;
	      
	      if( irr < ir )
		ir = (int)((double)ir/nref + 1.0)*nref;
	      else
		ir = irr;
	      
	      if( jrr < jr )
		jr = (int)((double)jr/nref + 1.0)*nref;
	      else
		jr = jrr;
	      
	      if( krr < kr )
		kr = (int)((double)kr/nref + 1.0)*nref;
	      else
		kr = krr;
	      
	      
	    }else{
	    //... require alignment with coarser grid
	    il -= il%2; jl -= jl%2; kl -= kl%2;
	    ir += ir%2; jr += jr%2; kr += kr%2; 
	  }

		// require alighment with coarser block
		if (blocking_factor_)
		{
			unsigned coarse_block = 2 * blocking_factor_;
			il -= il % coarse_block;
			jl -= jl % coarse_block;
			kl -= kl % coarse_block;
			ir += (nresmax - ir) % coarse_block;
			jr += (nresmax - jr) % coarse_block;
			kr += (nresmax - kr) % coarse_block;
		}
	  
	  // if doing unigrid, set region to whole box
	  if( levelmin_ == levelmax_ )
	    {
	      il = jl = kl = 0;
	      ir = jr = kr = nresmax-1;
	    }
	  if( bhave_nref )
	    {
	      ir = il+lnref_[0];
	      jr = jl+lnref_[1];
	      kr = kl+lnref_[2];
	      
	    }
	  
	  //... make sure bounding box lies in domain
	  // Track original spans before modulo: a full-extent box (span == nresmax)
	  // would have its right edge wrap to 0, breaking the il<ir check below.
	  // This case arises from multibox plugins whose union AABB spans a full
	  // periodic axis (e.g. z-column pillars).
	  const int orig_span_x = ir - il;
	  const int orig_span_y = jr - jl;
	  const int orig_span_z = kr - kl;
	  il = (il+nresmax)%nresmax; ir = (ir+nresmax)%nresmax;
	  jl = (jl+nresmax)%nresmax; jr = (jr+nresmax)%nresmax;
	  kl = (kl+nresmax)%nresmax; kr = (kr+nresmax)%nresmax;
	  if( orig_span_x >= nresmax ) { il = 0; ir = nresmax; }
	  if( orig_span_y >= nresmax ) { jl = 0; jr = nresmax; }
	  if( orig_span_z >= nresmax ) { kl = 0; kr = nresmax; }

	  if( il>=ir || jl>=jr || kl>=kr )
	    {
	      LOGERR("Internal refinement bounding box error: [%d,%d]x[%d,%d]x[%d,%d]",il,ir,jl,jr,kl,kr);
	      throw std::runtime_error("refinement_hierarchy: Internal refinement bounding box error 1");
	    }
	  //... determine offsets
	  if( levelmin_ != levelmax_ )
	    {
	      
	      oax_[levelmax_] = (il+nresmax)%nresmax;
	      oay_[levelmax_] = (jl+nresmax)%nresmax;
	      oaz_[levelmax_] = (kl+nresmax)%nresmax;
	      nx_[levelmax_]  = ir-il;	
	      ny_[levelmax_]  = jr-jl;	
	      nz_[levelmax_]  = kr-kl;
	      
	      if( equal_extent_ )
		{
		  
		  if( bhave_nref && (lnref_[0]!=lnref_[1]||lnref_[0]!=lnref_[2]) )
		    {
		      LOGERR("Specified equal_extent=yes conflicting with ref_dims which are not equal.");
		      throw std::runtime_error("Specified equal_extent=yes conflicting with ref_dims which are not equal.");
		    }
		  size_t ilevel = levelmax_;
		  size_t nmax = std::max( nx_[ilevel], std::max( ny_[ilevel], nz_[ilevel] ) );				
		  int dx = (int)((double)(nmax-nx_[ilevel])*0.5);
		  int dy = (int)((double)(nmax-ny_[ilevel])*0.5);
		  int dz = (int)((double)(nmax-nz_[ilevel])*0.5);
		  
		  oax_[ilevel] -= dx;
		  oay_[ilevel] -= dy;
		  oaz_[ilevel] -= dz;
		  nx_[ilevel] = nmax;
		  ny_[ilevel] = nmax;
		  nz_[ilevel] = nmax;
		  
		  il = oax_[ilevel];
		  jl = oay_[ilevel];
		  kl = oaz_[ilevel];
		  ir = il + nmax;
		  jr = jl + nmax;
		  kr = kl + nmax;
		}
	    }
	  
	  padding_	= cf_.getValueSafe<unsigned>("setup","padding", 8);
	  
	  //... determine position of coarser grids
	  for( unsigned ilevel=levelmax_-1; ilevel> levelmin_; --ilevel )
	    {
	      il = (int)((double)il * 0.5 - padding_);
	      jl = (int)((double)jl * 0.5 - padding_);
	      kl = (int)((double)kl * 0.5 - padding_);
	      
	      ir = (int)((double)ir * 0.5 + padding_);
	      jr = (int)((double)jr * 0.5 + padding_);
	      kr = (int)((double)kr * 0.5 + padding_);
	      
	      //... align with coarser grids ...
	      if( align_top_ )
		{
		  //... require alignment with top grid
		  unsigned nref = 1<<(ilevel-levelmin_);
		  
		  il = (int)((double)il/nref)*nref;
		  jl = (int)((double)jl/nref)*nref;
		  kl = (int)((double)kl/nref)*nref;
		  
		  ir = (int)((double)ir/nref+1.0)*nref;
		  jr = (int)((double)jr/nref+1.0)*nref;
		  kr = (int)((double)kr/nref+1.0)*nref;
		  
		}
	      else
		{
		  //... require alignment with coarser grid
		  il -= il%2; jl -= jl%2; kl -= kl%2;
		  ir += ir%2; jr += jr%2; kr += kr%2; 
		}

		// require alighment with coarser block
		if (blocking_factor_)
		{
			unsigned coarse_block = 2 * blocking_factor_;
			int nres = 1 << ilevel;
			il -= il % coarse_block;
			jl -= jl % coarse_block;
			kl -= kl % coarse_block;
			ir += (nres - ir) % coarse_block;
			jr += (nres - jr) % coarse_block;
			kr += (nres - kr) % coarse_block;
		}


	      // Full-extent at this coarser level: after halving + padding + alignment,
	      // a full-axis region may span beyond [0, 1<<ilevel) on both sides. Clamp
	      // to the full level extent so downstream offsets/sizes stay sensible.
	      {
	        const int nres_lev = 1<<ilevel;
	        if( ir - il >= nres_lev ) { il = 0; ir = nres_lev; }
	        if( jr - jl >= nres_lev ) { jl = 0; jr = nres_lev; }
	        if( kr - kl >= nres_lev ) { kl = 0; kr = nres_lev; }
	      }

	      if( il>=ir || jl>=jr || kl>=kr || il < 0 || jl < 0 || kl < 0)
		{
		  LOGERR("Internal refinement bounding box error: [%d,%d]x[%d,%d]x[%d,%d], level=%d",il,ir,jl,jr,kl,kr,ilevel);
		  throw std::runtime_error("refinement_hierarchy: Internal refinement bounding box error 2");
		}
	      oax_[ilevel] = il;
	      oay_[ilevel] = jl;
	      oaz_[ilevel] = kl;
	      nx_[ilevel]  = ir-il;
	      ny_[ilevel]  = jr-jl;
	      nz_[ilevel]  = kr-kl;
	      
		// should be achieved by the alignment
	  //     if (blocking_factor_)
		// {
		//   nx_[ilevel] += nx_[ilevel] % blocking_factor_;
		//   ny_[ilevel] += ny_[ilevel] % blocking_factor_;
		//   nz_[ilevel] += nz_[ilevel] % blocking_factor_;
		// }
	      
	      if( equal_extent_ )
		{
		  size_t nmax = std::max( nx_[ilevel], std::max( ny_[ilevel], nz_[ilevel] ) );				
		  int dx = (int)((double)(nmax-nx_[ilevel])*0.5);
		  int dy = (int)((double)(nmax-ny_[ilevel])*0.5);
		  int dz = (int)((double)(nmax-nz_[ilevel])*0.5);
				
		  oax_[ilevel] -= dx;
		  oay_[ilevel] -= dy;
		  oaz_[ilevel] -= dz;
		  nx_[ilevel] = nmax;
		  ny_[ilevel] = nmax;
		  nz_[ilevel] = nmax;
		  
		  il = oax_[ilevel];
		  jl = oay_[ilevel];
		  kl = oaz_[ilevel];
		  ir = il + nmax;
		  jr = jl + nmax;
		  kr = kl + nmax;
		}
	      
	    }
	  
	  //... determine relative offsets between grids
	  for( unsigned ilevel=levelmax_; ilevel>levelmin_; --ilevel )
	    {
	      ox_[ilevel] = (oax_[ilevel]/2 - oax_[ilevel-1]);
	      oy_[ilevel] = (oay_[ilevel]/2 - oay_[ilevel-1]);
	      oz_[ilevel] = (oaz_[ilevel]/2 - oaz_[ilevel-1]);
	    }
	  
	  //... do a forward sweep to ensure that absolute offsets are also correct now
	  for( unsigned ilevel=levelmin_+1; ilevel<=levelmax_; ++ilevel )
	    {
	      oax_[ilevel] = 2*oax_[ilevel-1]+2*ox_[ilevel];
	      oay_[ilevel] = 2*oay_[ilevel-1]+2*oy_[ilevel];
	      oaz_[ilevel] = 2*oaz_[ilevel-1]+2*oz_[ilevel];
	    }
	  
	  for( unsigned ilevel=levelmin_+1; ilevel<=levelmax_; ++ilevel )
	    {
	      double h = 1.0/(1ul<<ilevel);
	      
	      x0_[ilevel] = h*(double)oax_[ilevel];
	      y0_[ilevel] = h*(double)oay_[ilevel];
	      z0_[ilevel] = h*(double)oaz_[ilevel];
	      
	      xl_[ilevel] = h*(double)nx_[ilevel];
	      yl_[ilevel] = h*(double)ny_[ilevel];
	      zl_[ilevel] = h*(double)nz_[ilevel];
	    }
	  
	  
	  
	  // do a consistency check that largest subgrid in zoom is not larger than half the box size.
	  // For multibox runs with >1 disjoint cluster, this check applies to each per-box AABB
	  // independently rather than to the union (the legacy union mesh is allowed to span more
	  // than half the box because real compute now happens on the per-box sub-meshes).
	  for( unsigned ilevel=levelmin_+1; ilevel<=levelmax_; ++ilevel )
	    {
	      const size_t nb = the_region_generator->get_num_boxes(ilevel);
	      if( nb <= 1 )
		{
		  if( nx_[ilevel] > (1ul<<(ilevel-1)) ||
		      ny_[ilevel] > (1ul<<(ilevel-1)) ||
		      nz_[ilevel] > (1ul<<(ilevel-1)) )
		    {
		      LOGERR("On level %d, subgrid is larger than half the box. This is not allowed!",ilevel);
		      throw std::runtime_error("Fatal: Subgrid larger than half boxin zoom.");
		    }
		}
	      else
		{
		  // Multibox: clusters come from connected-component labeling on
		  // a user-supplied Lagrangian point file, so any extent up to the
		  // full periodic axis is legitimate by construction. We only emit
		  // an INFO note for unusually large clusters (>half) so anomalous
		  // geometries still leave a trail.
		  for( size_t b=0; b<nb; ++b )
		    {
		      double l[3], r[3];
		      the_region_generator->get_AABB_box(l, r, ilevel, b);
		      for( int d=0; d<3; ++d )
			{
			  double ext = r[d]-l[d];
			  if( ext > 0.5 )
			    LOGINFO("Level %d multibox cluster %zu axis=%d extent=%.4f (> half box; allowed for multibox)",
				    ilevel, b, d, ext);
			}
		    }
		}
	    }
	  
	  // update the region generator with what has been actually created
	  double left[3] =  { x0_[levelmax_]+rshift_[0], y0_[levelmax_]+rshift_[1], z0_[levelmax_]+rshift_[2] };
	  double right[3] = { left[0]+xl_[levelmax_], left[1]+yl_[levelmax_], left[2]+zl_[levelmax_] };
      the_region_generator->update_AABB( left, right );

	  //==================================================================
	  // Phase D.2a: populate per-box LevelBox storage.
	  //
	  // For single-box plugins (default get_num_boxes==1) level_boxes_[L][0]
	  // is just a copy of the existing union arrays — zoom paths remain
	  // bit-identical because no compute path consumes level_boxes_ yet.
	  // For region=multibox with N>1 connected components level_boxes_[L]
	  // holds the N disjoint sub-AABBs after align/pad/blocking. Future
	  // GridHierarchy per-box allocation (D.2b) and per-box Poisson V-cycle
	  // (D.3) will read these.
	  //==================================================================
	  populate_level_boxes();
	}

private:
	//! Mirror a single-box level (union arrays) into level_boxes_[L] as one entry.
	void push_union_as_single_box( unsigned L )
	{
	    LevelBox bx;
	    bx.oax = oax_[L]; bx.oay = oay_[L]; bx.oaz = oaz_[L];
	    bx.nx  = nx_[L];  bx.ny  = ny_[L];  bx.nz  = nz_[L];
	    bx.ox  = (int)ox_[L]; bx.oy  = (int)oy_[L]; bx.oz  = (int)oz_[L];
	    bx.x0  = x0_[L];  bx.y0  = y0_[L];  bx.z0  = z0_[L];
	    bx.xl  = xl_[L];  bx.yl  = yl_[L];  bx.zl  = zl_[L];
	    bx.parent_idx = 0;
	    level_boxes_[L].push_back(bx);
	}

	//! Apply the same coarser-grid / blocking_factor alignment that the union
	//! path uses at lines 1547-1605 (levelmax) and 1685-1718 (coarser), to
	//! one box at level L given by integer fine-cell bounds [il,ir)x[jl,jr)x[kl,kr).
	void align_box_inplace( unsigned L, int &il, int &jl, int &kl,
	                                     int &ir, int &jr, int &kr ) const
	{
	    int nresL = 1<<L;
	    if( align_top_ )
	    {
	        unsigned nref = (L == levelmax_) ? (1u<<(levelmax_-levelmin_+1))
	                                          : (1u<<(L-levelmin_));
	        il = (int)((double)il/nref)*nref;
	        jl = (int)((double)jl/nref)*nref;
	        kl = (int)((double)kl/nref)*nref;
	        int irr = (int)((double)ir/nref)*nref;
	        int jrr = (int)((double)jr/nref)*nref;
	        int krr = (int)((double)kr/nref)*nref;
	        ir = (irr < ir) ? (int)((double)ir/nref + 1.0)*nref : irr;
	        jr = (jrr < jr) ? (int)((double)jr/nref + 1.0)*nref : jrr;
	        kr = (krr < kr) ? (int)((double)kr/nref + 1.0)*nref : krr;
	    } else {
	        il -= il%2; jl -= jl%2; kl -= kl%2;
	        ir += ir%2; jr += jr%2; kr += kr%2;
	    }
	    if( blocking_factor_ ) {
	        unsigned coarse_block = 2u * blocking_factor_;
	        il -= il % coarse_block;
	        jl -= jl % coarse_block;
	        kl -= kl % coarse_block;
	        ir += (nresL - ir) % coarse_block;
	        jr += (nresL - jr) % coarse_block;
	        kr += (nresL - kr) % coarse_block;
	    }
	}

	void populate_level_boxes( void )
	{
	    level_boxes_.assign(levelmax_+1, std::vector<LevelBox>());

	    // Levels [0, levelmin_]: full periodic domain, single box.
	    for( unsigned L = 0; L <= levelmin_; ++L )
	    {
	        unsigned n = 1u<<L;
	        LevelBox bx;
	        bx.oax = bx.oay = bx.oaz = 0;
	        bx.nx  = bx.ny  = bx.nz  = n;
	        bx.ox  = bx.oy  = bx.oz  = 0;
	        bx.x0  = bx.y0  = bx.z0 = 0.0;
	        bx.xl  = bx.yl  = bx.zl = 1.0;
	        bx.parent_idx = 0;
	        level_boxes_[L].push_back(bx);
	    }

	    if( levelmin_ == levelmax_ ) return;

	    // Levels [levelmin_+1, levelmax_]: ask the region generator for the
	    // disjoint boxes, run the same alignment/blocking logic the union
	    // path used at the finest level, and attach each to its parent.
	    for( unsigned L = levelmin_+1; L <= levelmax_; ++L )
	    {
	        size_t nb = the_region_generator->get_num_boxes(L);

	        // Single-box plugin (default) → mirror the union arrays so that
	        // legacy callers asking the per-box accessors get identical
	        // numbers as the legacy scalar accessors.
	        if( nb <= 1 ) { push_union_as_single_box(L); continue; }

	        int nresL = 1<<L;
	        double shift_norm[3] = {
	            (double)xshift_[0]/(double)(1u<<levelmin_),
	            (double)xshift_[1]/(double)(1u<<levelmin_),
	            (double)xshift_[2]/(double)(1u<<levelmin_),
	        };

	        for( size_t b = 0; b < nb; ++b )
	        {
	            double l[3], r[3];
	            the_region_generator->get_AABB_box(l, r, L, b);
	            for( int d=0; d<3; ++d ) { l[d] += shift_norm[d]; r[d] += shift_norm[d]; }

	            int il = (int)(l[0]*nresL), jl = (int)(l[1]*nresL), kl = (int)(l[2]*nresL);
	            int ir = (int)(r[0]*nresL), jr = (int)(r[1]*nresL), kr = (int)(r[2]*nresL);

	            // Mirror the union-path "ir = il+lnref" forced size only when
	            // user pinned grid dims — keep it simple here and skip.
	            align_box_inplace(L, il, jl, kl, ir, jr, kr);

	            // Wrap into [0,nresL). Same minimal handling the union path does.
	            il = (il + nresL) % nresL;
	            jl = (jl + nresL) % nresL;
	            kl = (kl + nresL) % nresL;
	            if( ir <= il ) ir = il + 1;
	            if( jr <= jl ) jr = jl + 1;
	            if( kr <= kl ) kr = kl + 1;

	            LevelBox bx;
	            bx.oax = (unsigned)il;
	            bx.oay = (unsigned)jl;
	            bx.oaz = (unsigned)kl;
	            bx.nx  = (unsigned)(ir - il);
	            bx.ny  = (unsigned)(jr - jl);
	            bx.nz  = (unsigned)(kr - kl);

	            double h = 1.0/(double)nresL;
	            bx.x0 = h*(double)bx.oax;
	            bx.y0 = h*(double)bx.oay;
	            bx.z0 = h*(double)bx.oaz;
	            bx.xl = h*(double)bx.nx;
	            bx.yl = h*(double)bx.ny;
	            bx.zl = h*(double)bx.nz;

	            // Parent box: the level-(L-1) box that contains this one.
	            // Containment in fine cells at level L (parent's cells doubled).
	            bx.parent_idx = (size_t)-1;
	            const auto & parents = level_boxes_[L-1];
	            for( size_t p = 0; p < parents.size(); ++p )
	            {
	                unsigned pox = parents[p].oax * 2, poy = parents[p].oay * 2, poz = parents[p].oaz * 2;
	                unsigned pnx = parents[p].nx  * 2, pny = parents[p].ny  * 2, pnz = parents[p].nz  * 2;
	                if( bx.oax >= pox && bx.oax + bx.nx <= pox + pnx &&
	                    bx.oay >= poy && bx.oay + bx.ny <= poy + pny &&
	                    bx.oaz >= poz && bx.oaz + bx.nz <= poz + pnz )
	                { bx.parent_idx = p; break; }
	            }
	            if( bx.parent_idx == (size_t)-1 )
	            {
	                char emsg[512];
	                snprintf(emsg, sizeof(emsg),
	                    "Multibox allocator: level %u box %zu at oax=(%u,%u,%u) n=(%u,%u,%u) has no parent at level %u. "
	                    "Increase padding or check region_generator output.",
	                    L, level_boxes_[L].size(), bx.oax, bx.oay, bx.oaz, bx.nx, bx.ny, bx.nz, L-1);
	                throw std::runtime_error(emsg);
	            }

	            const LevelBox & parent = level_boxes_[L-1][bx.parent_idx];
	            bx.ox = (int)(bx.oax/2) - (int)parent.oax;
	            bx.oy = (int)(bx.oay/2) - (int)parent.oay;
	            bx.oz = (int)(bx.oaz/2) - (int)parent.oaz;

	            level_boxes_[L].push_back(bx);
	        }

	        LOGINFO("Multibox allocator: level %u → %zu disjoint sub-mesh(es) (post align/blocking)", L, level_boxes_[L].size());
	        for( size_t b = 0; b < level_boxes_[L].size(); ++b )
	        {
	            const LevelBox & bx = level_boxes_[L][b];
	            LOGINFO("    box %zu: oax=(%u,%u,%u) n=(%u,%u,%u) parent=%zu  rel_off=(%d,%d,%d)",
	                    b, bx.oax, bx.oay, bx.oaz, bx.nx, bx.ny, bx.nz, bx.parent_idx,
	                    bx.ox, bx.oy, bx.oz);
	        }
	    }
	}

public:
  
  //! asignment operator
  refinement_hierarchy& operator=( const refinement_hierarchy& o )
  {
    levelmin_ = o.levelmin_;
    levelmax_ = o.levelmax_;
    levelmin_tf_ = o.levelmin_tf_;
    padding_  = o.padding_;
    cf_ = o.cf_;
    align_top_ = o.align_top_;
    for( int i=0; i<3; ++i )
      {
	x0ref_[i] = o.x0ref_[i];
	lxref_[i] = o.lxref_[i];
	xshift_[i] = o.xshift_[i];
	rshift_[i] = o.rshift_[i];
      }
    
    x0_ = o.x0_; y0_ = o.y0_; z0_ = o.z0_;
    xl_ = o.xl_; yl_ = o.yl_; zl_ = o.zl_;
    ox_ = o.ox_; oy_ = o.oy_; oz_ = o.oz_;
    oax_= o.oax_; oay_ = o.oay_; oaz_ = o.oaz_;
    nx_ = o.nx_; ny_=o.ny_; nz_=o.nz_;
    level_boxes_ = o.level_boxes_;

    return *this;
  }
	
	/*! cut a grid level to the specified size
	 * @param ilevel grid level on which to perform the size adjustment
	 * @param nx new x-extent in fine grid cells
	 * @param ny new y-extent in fine grid cells
	 * @param nz new z-extent in fine grid cells
	 * @param oax new x-offset in units fine grid units
	 * @param oay new y-offset in units fine grid units
	 * @param oaz new z-offset in units fine grid units
	
	 */	
	void adjust_level( unsigned ilevel, int nx, int ny, int nz, int oax, int oay, int oaz )
	{
		double h = 1.0/(1<<ilevel);
		
		int dx,dy,dz;
		
		dx = oax_[ilevel] - oax;
		dy = oay_[ilevel] - oay;
		dz = oaz_[ilevel] - oaz;
		
		ox_[ilevel] -= dx/2;
		oy_[ilevel] -= dy/2;
		oz_[ilevel] -= dz/2;
		
		oax_[ilevel] = oax; 
		oay_[ilevel] = oay; 
		oaz_[ilevel] = oaz;
		
		nx_[ilevel] = nx;
		ny_[ilevel] = ny;
		nz_[ilevel] = nz;
		
		x0_[ilevel] = h*oax;
		y0_[ilevel] = h*oay;
		z0_[ilevel] = h*oaz;
		
		xl_[ilevel] = h*nx;
		yl_[ilevel] = h*ny;
		zl_[ilevel] = h*nz;
		
		if( ilevel < levelmax_ )
		{
			ox_[ilevel+1] += dx;
			oy_[ilevel+1] += dy;
			oz_[ilevel+1] += dz;
		}
		
		find_new_levelmin();
		
	}
	
	/*! determine level for which grid extends over entire domain */
	void find_new_levelmin( bool print=false )
	{
		unsigned old_levelmin( levelmin_ );

		for( unsigned i=0; i<=levelmax(); ++i )
		{
			unsigned n = 1<<i;

			if(	oax_[i]==0 && oay_[i]==0 && oaz_[i]==0
			   && nx_[i]==n && ny_[i]==n && nz_[i]==n )
			{
				levelmin_=i;
			}
		}

		// Don't promote past the configured TF base level. A refinement
		// level can transiently span the full domain after TF cubic-inflate
		// padding (e.g. multibox unions that already cover most of the box)
		// without being a legitimate new base — promoting would cause rng
		// generation to skip the true coarsest level.
		if( levelmin_ > levelmin_tf_ )
			levelmin_ = levelmin_tf_;

		if( (old_levelmin != levelmin_) && print)
			LOGINFO("refinement_hierarchy: set new levelmin to %d", levelmin_ );
	}
	
	//! get absolute grid offset for a specified level along a specified dimension (in fine grid units)
	unsigned offset_abs( unsigned ilevel, int dim ) const
	{
		if( dim==0 )
			return oax_.at(ilevel);
		if( dim==1 )
			return oay_.at(ilevel);
		return oaz_.at(ilevel);
	}
	
	//! get relative grid offset for a specified level along a specified dimension (in coarser grid units)
	int offset( unsigned ilevel, int dim ) const
	{
		if( dim==0 )
			return ox_.at(ilevel);
		if( dim==1 )
			return oy_.at(ilevel);
		return oz_.at(ilevel);
	}
	
	//! get grid size for a specified level along a specified dimension
	size_t size( unsigned ilevel, int dim ) const
	{
		if( dim==0 )
			return nx_.at(ilevel);
		if( dim==1 )
			return ny_.at(ilevel);
		return nz_.at(ilevel);
	}

	//-------- multibox per-box accessors (Phase D.2) ---------------------
	//! Number of disjoint sub-meshes at the given level. 1 for single-box
	//! plugins; N for region=multibox with N connected components.
	size_t num_boxes( unsigned ilevel ) const
	{
	    if( ilevel >= level_boxes_.size() ) return 0;
	    return level_boxes_[ilevel].size();
	}

	//! Read-only access to one LevelBox at (level, box_id).
	const LevelBox & level_box( unsigned ilevel, size_t box_id ) const
	{
	    return level_boxes_.at(ilevel).at(box_id);
	}

	//! Per-box absolute offset (fine cells at this level), dim ∈ {0,1,2}.
	unsigned offset_abs( unsigned ilevel, size_t box_id, int dim ) const
	{
	    const LevelBox & b = level_boxes_.at(ilevel).at(box_id);
	    if( dim==0 ) return b.oax;
	    if( dim==1 ) return b.oay;
	    return b.oaz;
	}

	//! Per-box size in fine cells at this level, dim ∈ {0,1,2}.
	size_t size( unsigned ilevel, size_t box_id, int dim ) const
	{
	    const LevelBox & b = level_boxes_.at(ilevel).at(box_id);
	    if( dim==0 ) return b.nx;
	    if( dim==1 ) return b.ny;
	    return b.nz;
	}

	//! Per-box offset relative to parent box (in coarse cells at level-1).
	int offset( unsigned ilevel, size_t box_id, int dim ) const
	{
	    const LevelBox & b = level_boxes_.at(ilevel).at(box_id);
	    if( dim==0 ) return b.ox;
	    if( dim==1 ) return b.oy;
	    return b.oz;
	}

	//! Index (in level_boxes_[ilevel-1]) of this box's parent.
	size_t parent_box_index( unsigned ilevel, size_t box_id ) const
	{
	    return level_boxes_.at(ilevel).at(box_id).parent_idx;
	}

	//! Raw view of the per-level per-box bookkeeping (used by
	//! GridHierarchy::populate_per_box_meshes to avoid a forward-decl).
	const std::vector< std::vector<LevelBox> > & get_level_boxes( void ) const
	{   return level_boxes_; }
	//---------------------------------------------------------------------

	//! get minimum grid level (the level for which the grid covers the entire domain)
	unsigned levelmin( void ) const
	{	return levelmin_;	}
	
	//! get maximum grid level
	unsigned levelmax( void ) const
	{	return levelmax_;	}
	
	//! get the total shift of the coordinate system in units of coarse cells
	int get_shift( int idim ) const
	{	return xshift_[idim];  }
	
    //! get the total shift of the coordinate system in box coordinates
    const double* get_coord_shift( void ) const
    {   return rshift_; }
    
	//! write refinement hierarchy to stdout
	void output( void ) const
	{
		std::cout << "-------------------------------------------------------------\n";
		
		if( xshift_[0]!=0||xshift_[1]!=0||xshift_[2]!=0 )
			std::cout << " - Domain will be shifted by (" << xshift_[0] << ", " << xshift_[1] << ", " << xshift_[2] << ")\n" << std::endl;

		std::cout << " - Grid structure:\n";
		
		for( unsigned ilevel=levelmin_; ilevel<=levelmax_; ++ilevel )
		{
			std::cout 
			<< "     Level " << std::setw(3) << ilevel << " :   offset = (" << std::setw(5) << ox_[ilevel] << ", " << std::setw(5) << oy_[ilevel] << ", " << std::setw(5) << oz_[ilevel] << ")\n"
			<< "               offset_abs = (" << std::setw(5) << oax_[ilevel] << ", " << std::setw(5) << oay_[ilevel] << ", " << std::setw(5) << oaz_[ilevel] << ")\n"
			<< "                   size   = (" << std::setw(5) << nx_[ilevel] << ", " << std::setw(5) << ny_[ilevel] << ", " << std::setw(5) << nz_[ilevel] << ")\n";
		}
		std::cout << "-------------------------------------------------------------\n";
	}
	
	void output_log( void ) const
	{
		LOGUSER("   Domain shifted by      (%5d,%5d,%5d)",xshift_[0],xshift_[1],xshift_[2]);		
		for( unsigned ilevel=levelmin_; ilevel<=levelmax_; ++ilevel )
		{	
			LOGUSER("   Level %3d :   offset = (%5d,%5d,%5d)",ilevel,ox_[ilevel],oy_[ilevel],oz_[ilevel]);
			LOGUSER("                   size = (%5d,%5d,%5d)",nx_[ilevel],ny_[ilevel],nz_[ilevel]);
		}
	}
    
    
};


typedef GridHierarchy<real_t> grid_hierarchy;
typedef MeshvarBnd<real_t> meshvar_bnd;
typedef Meshvar<real_t> meshvar;


#endif

