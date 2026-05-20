#ifndef __MPI_HELPER_HH
#define __MPI_HELPER_HH

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace MUSIC { namespace mpi {

#ifdef USE_MPI
inline MPI_Comm& world() { static MPI_Comm c = MPI_COMM_WORLD; return c; }
inline int rank() { int r = 0; MPI_Comm_rank(world(), &r); return r; }
inline int size() { int s = 1; MPI_Comm_size(world(), &s); return s; }
inline void barrier() { MPI_Barrier(world()); }
#else
inline int rank() { return 0; }
inline int size() { return 1; }
inline void barrier() {}
#endif

inline bool is_root() { return rank() == 0; }

}} // namespace MUSIC::mpi

#endif // __MPI_HELPER_HH
