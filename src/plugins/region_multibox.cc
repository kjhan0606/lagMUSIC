#include <vector>
#include <queue>
/*

 region_multibox.cc - A plugin for MUSIC -
 a code to generate multi-scale initial conditions
 for cosmological simulations

 Copyright (C) 2015 Ben Keller

 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>
#include <cctype>
#include <algorithm>

#include "region_generator.hh"
#include "point_file_reader.hh"


typedef std::vector<unsigned> col;
typedef std::vector<col> slice;
typedef std::vector<slice> grid;

typedef struct {
    unsigned x;
    unsigned y;
    unsigned z;
} coord;

typedef std::vector<coord> region;

//! A single disjoint refinement box on the multibox refgrid (cell indices in [0, res)).
//! hi is exclusive: the box covers cells [lo[d], hi[d]).
struct cc_box {
    unsigned lo[3];
    unsigned hi[3];
};

class region_multibox_plugin : public region_generator_plugin{
private:
    double cen[3];
    unsigned res;
    double vfac_;
    grid refgrid;
    //! disjoint refinement boxes per level (cell indices on the level-(`region_point_level`) refgrid).
    //! boxes_per_level_[L] is the list of connected components of {cells : refgrid[i][j][k] >= L},
    //! for L in [levelmin_+1, levelmax_]. Computed once in compute_components_per_level().
    std::vector< std::vector<cc_box> > boxes_per_level_;
    //! union bounding box per level (used by get_AABB to preserve legacy single-box behaviour).
    std::vector<cc_box> union_box_per_level_;
    
    region where(unsigned level)
    {
        slice curslice;
        col curcol;
        region equal_region;
        coord cp;
        for(unsigned i=0;i<refgrid.size();i++)
        {
            cp.x = i;
            curslice = refgrid[i];
            for(unsigned j=0;j<curslice.size();j++)
            {
                cp.y = j;
                curcol = curslice[j];
                for(unsigned k=0;k<curcol.size();k++)
                {
                    cp.z = k;
                    if(curcol[k] == level)
                    {
                        equal_region.push_back(cp);
                    }
                }
            }
        }
        return equal_region;
    }
    //This function takes the grid, a vector of doubles containing the particle
    //coordinates, and sets each particle-containing gridcell to levelmax_.
    void deposit_particles(std::vector<double> pp, unsigned res)
    {
        unsigned i,j,k;
        std::vector<double>::iterator cp = pp.begin();
        while(cp != pp.end())
        {
            i = (*(cp))*res;
            cp++;
            j = (*(cp))*res;
            cp++;
            k = (*(cp))*res;
            cp++;
            refgrid[i][j][k] = levelmax_;
        }
    }

    //This function takes the grid, which has been already had particles
    //deposited onto it, and set to the maximum refinement level.  It then
    //fills the remaining refinement levels.
    //
    //Neighbour lookups wrap periodically — clusters that touch a box edge
    //would otherwise index refgrid[-1] / refgrid[res] and segfault.
    void build_refgrid()
    {
        region curregion;
        //Build an extra layer about the maxlevel layer.
        curregion = where(levelmax_);
        for(region::iterator cp= curregion.begin(); cp != curregion.end(); ++cp)
        {
            for(int i=-1; i<2; i++)
            {
                unsigned ii = (cp->x + (int)res + i) % res;
                for(int j=-1; j<2; j++)
                {
                    unsigned jj = (cp->y + (int)res + j) % res;
                    for(int k=-1; k<2; k++)
                    {
                        unsigned kk = (cp->z + (int)res + k) % res;
                        if(refgrid[ii][jj][kk] == levelmin_)
                        {
                            refgrid[ii][jj][kk] = levelmax_;
                        }
                    }
                }
            }
        }
        for(unsigned curlevel=levelmax_; curlevel>(levelmin_+1); curlevel--)
        {
            curregion = where(curlevel);
            for(region::iterator cp= curregion.begin(); cp != curregion.end(); ++cp)
            {
                for(int i=-1; i<2; i++)
                {
                    unsigned ii = (cp->x + (int)res + i) % res;
                    for(int j=-1; j<2; j++)
                    {
                        unsigned jj = (cp->y + (int)res + j) % res;
                        for(int k=-1; k<2; k++)
                        {
                            unsigned kk = (cp->z + (int)res + k) % res;
                            if(refgrid[ii][jj][kk] == levelmin_)
                            {
                                refgrid[ii][jj][kk] = curlevel-1;
                            }
                        }
                    }
                }
            }
        }
    }

    //! Label connected components (26-connected, periodic) of cells with refgrid >= L
    //! for each refinement level L in (levelmin_, levelmax_]. Populates boxes_per_level_
    //! and union_box_per_level_. The refgrid cell indices live on a grid of resolution
    //! `res` (set from setup.region_point_level, default levelmin_-1); AABBs returned
    //! to the caller via get_AABB_box are normalised to [0,1) box coordinates.
    void compute_components_per_level()
    {
        boxes_per_level_.assign(levelmax_+1, std::vector<cc_box>());
        union_box_per_level_.assign(levelmax_+1, cc_box());

        // visited mask, reused across levels (cleared each level)
        std::vector<unsigned char> visited(res*res*res, 0);

        for( unsigned L = levelmin_+1; L <= levelmax_; ++L )
        {
            std::fill(visited.begin(), visited.end(), 0);

            unsigned umin[3] = {res, res, res};
            unsigned umax[3] = {0, 0, 0};
            bool any_cell = false;

            for( unsigned i=0; i<res; ++i )
            for( unsigned j=0; j<res; ++j )
            for( unsigned k=0; k<res; ++k )
            {
                size_t idx = ((size_t)i*res + j)*res + k;
                if( visited[idx] ) continue;
                if( refgrid[i][j][k] < L ) continue;

                // BFS for one connected component (26-connected, periodic).
                cc_box bb;
                bb.lo[0] = i; bb.hi[0] = i+1;
                bb.lo[1] = j; bb.hi[1] = j+1;
                bb.lo[2] = k; bb.hi[2] = k+1;
                any_cell = true;

                std::queue<coord> q;
                coord c0 = {i,j,k};
                q.push(c0);
                visited[idx] = 1;

                while(!q.empty())
                {
                    coord c = q.front(); q.pop();

                    // Track AABB. Note: 26-connected periodic BFS can yield a
                    // component that wraps the periodic boundary; in that case
                    // [lo,hi) on a single axis is ambiguous. We accept the
                    // limitation (same as the existing get_AABB(union) path)
                    // and just report the min/max cell indices.
                    if( c.x   < bb.lo[0] ) bb.lo[0] = c.x;
                    if( c.x+1 > bb.hi[0] ) bb.hi[0] = c.x+1;
                    if( c.y   < bb.lo[1] ) bb.lo[1] = c.y;
                    if( c.y+1 > bb.hi[1] ) bb.hi[1] = c.y+1;
                    if( c.z   < bb.lo[2] ) bb.lo[2] = c.z;
                    if( c.z+1 > bb.hi[2] ) bb.hi[2] = c.z+1;

                    for( int di=-1; di<=1; ++di )
                    for( int dj=-1; dj<=1; ++dj )
                    for( int dk=-1; dk<=1; ++dk )
                    {
                        if( di==0 && dj==0 && dk==0 ) continue;
                        unsigned ni = (c.x + (int)res + di) % res;
                        unsigned nj = (c.y + (int)res + dj) % res;
                        unsigned nk = (c.z + (int)res + dk) % res;
                        size_t nidx = ((size_t)ni*res + nj)*res + nk;
                        if( visited[nidx] ) continue;
                        if( refgrid[ni][nj][nk] < L ) continue;
                        visited[nidx] = 1;
                        coord nc = {ni,nj,nk};
                        q.push(nc);
                    }
                }

                boxes_per_level_[L].push_back(bb);

                // accumulate union AABB
                for( int d=0; d<3; ++d ) {
                    if( bb.lo[d] < umin[d] ) umin[d] = bb.lo[d];
                    if( bb.hi[d] > umax[d] ) umax[d] = bb.hi[d];
                }
            }

            if( any_cell ) {
                for( int d=0; d<3; ++d ) {
                    union_box_per_level_[L].lo[d] = umin[d];
                    union_box_per_level_[L].hi[d] = umax[d];
                }
            } else {
                // No cells at this level — emit a degenerate empty box so callers see size 0.
                for( int d=0; d<3; ++d ) {
                    union_box_per_level_[L].lo[d] = 0;
                    union_box_per_level_[L].hi[d] = 0;
                }
            }

            LOGINFO("Multibox level %u: detected %zu disjoint refinement box(es)",
                    L, boxes_per_level_[L].size());
        }
    }
    
public:
    explicit region_multibox_plugin( config_file& cf )
    : region_generator_plugin( cf )
    {
        res = 1<<(levelmin_-1);
        //check parameters
        if ( !cf.containsKey("setup", "region_point_file"))
        {
            LOGERR("Missing parameter \'region_point_file\' needed for region=multibox");
            throw std::runtime_error("Missing parameter for region=multibox");
        }
        if ( cf.containsKey("setup", "region_point_level"))
        {
        res = 1<<(levelmin_-1);
        res = 1<<(cf.getValue<int>("setup","region_point_level")-1);
        }
        vfac_ = cf.getValue<double>("cosmology","vfact");
        if (levelmin_ >= levelmax_)
        {
            LOGERR("Why are you specifying a region if you aren't refining?");
            throw std::runtime_error("region=multibox needs levelmax>levelmin");
        }
        std::vector<double> pp;
        point_reader pfr;
        pfr.read_points_from_file(cf.getValue<std::string>("setup", "region_point_file"), vfac_, pp);
        LOGINFO("Multibox Resolution: %d", res);
        //Initialize the grid with zeros, the base level
        refgrid = grid(res,slice(res,col(res,levelmin_)));
        //Build the highest refinement region
        deposit_particles(pp, res);
        LOGINFO("Deposited Multigrid Particles");
        //Build the intermediate refinement regions
        build_refgrid();
        LOGINFO("Built Multigrid");
        //Detect disjoint connected components per level (used by get_AABB_box / get_num_boxes).
        compute_components_per_level();
        get_center(cen);
        if( cf.getValueSafe<bool>("setup", "multibox_dump_grid", false) )
            dump_grid();
    }
    
    
    void get_AABB( double *left, double *right, unsigned level )
    {
        left[0] = left[1] = left[2] = 1.0;
        right[0] = right[1] = right[2] = 0.0;
        if( level <= levelmin_ )
        {
            left[0] = left[1] = left[2] = 0.0;
            right[0] = right[1] = right[2] = 1.0;
            return;
        }
        region myres = where(level);
        std::vector<coord>::iterator cp = myres.begin();
        while (cp != myres.end())
        {
            //Check left side
            if (float(cp->x)/res < left[0])
                left[0] = float(cp->x)/res;
            if (float(cp->y)/res < left[1])
                left[1] = float(cp->y)/res;
            if (float(cp->z)/res < left[2])
                left[2] = float(cp->z)/res;
            //Check right side. A refgrid cell at index N covers [N/res, (N+1)/res);
            //the AABB right edge must use (N+1)/res so the union mesh actually
            //contains all marked cells. (Pre-D.3.3 the union was undersized by
            //one refgrid-cell on each axis — at L=levelmax this typically lost
            //nresL/res fine cells per axis, leaving per-box AABBs from
            //get_AABB_box extending beyond the union.)
            if (float(cp->x+1)/res > right[0])
                right[0] = float(cp->x+1)/res;
            if (float(cp->y+1)/res > right[1])
                right[1] = float(cp->y+1)/res;
            if (float(cp->z+1)/res > right[2])
                right[2] = float(cp->z+1)/res;
            cp++;
        }
    }
    //! Number of disjoint refinement boxes at the requested level.
    //! Returns 1 for the base level and for levels outside [levelmin_+1, levelmax_].
    size_t get_num_boxes( unsigned level )
    {
        if( level <= levelmin_ ) return 1;
        if( level > levelmax_ )  return 0;
        if( level >= boxes_per_level_.size() ) return 0;
        return boxes_per_level_[level].size();
    }

    //! AABB of the box_id-th disjoint refinement box at the requested level, in
    //! normalised box coordinates [0,1).
    void get_AABB_box( double *left, double *right, unsigned level, size_t box_id )
    {
        if( level <= levelmin_ ) {
            if( box_id != 0 )
                throw std::runtime_error("region_multibox: box_id>0 at base level");
            left[0] = left[1] = left[2] = 0.0;
            right[0] = right[1] = right[2] = 1.0;
            return;
        }
        if( level >= boxes_per_level_.size() || box_id >= boxes_per_level_[level].size() )
            throw std::runtime_error("region_multibox: get_AABB_box: box_id out of range");
        const cc_box & b = boxes_per_level_[level][box_id];
        const double inv = 1.0 / (double)res;
        for( int d=0; d<3; ++d ) {
            left[d]  = (double)b.lo[d] * inv;
            right[d] = (double)b.hi[d] * inv;
        }
    }

    void dump_grid()
    {
        FILE * gridfile = fopen("griddump.dat", "w");
        for(unsigned i=0;i<res;i++)
        {
            for(unsigned j=0; j<res; j++)
            {
                for(unsigned k=0;k<res;k++)
                {
                    fprintf(gridfile, "%03d %03d %03d %1d\n", i-int(res*cen[0]),j-int(res*cen[1]),k-int(res*cen[2]), refgrid[i][j][k]);
                }
            }
        }
        fclose(gridfile);
    }
  
    void update_AABB( double *left, double *right )
    {
        //no need for this I think?
    }
  
    bool query_point( double *x, int level )
    {
        for(int i=0; i<3; ++i)
        {
            if(x[i] >= 1.0 || x[i] <= 0.0) return false;
        }
        return (level <= int(refgrid[(x[0])*res][(x[1])*res][(x[2])*res]));
    }
    
    bool is_grid_dim_forced( size_t* ndims )
    {   
        return false; //is this true?
    }
    
    void get_center( double *xc )
    {
        double xc2[3];
        get_AABB(xc, xc2, levelmax_);
        for(int i=0; i<3; ++i)
        {
            xc[i] += xc2[i];
            xc[i] *= 0.5;
        }
    }

  void get_center_unshifted( double *xc )
  {
      get_center(xc);
  }

};

namespace{
    region_generator_plugin_creator_concrete< region_multibox_plugin > creator("multibox");
}
