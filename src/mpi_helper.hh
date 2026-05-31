#ifndef __MPI_HELPER_HH
#define __MPI_HELPER_HH

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <cstdio>
#include <string>
#include <vector>

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

// Per-rank phase profiling helpers.
//
// gather_to_root(local) -> on rank-0: vector<double> of size N (rank-ordered);
// on workers: empty vector. np<=1 / non-MPI builds: rank-0 returns {local}.
// Collective (every rank must call).
inline std::vector<double> gather_to_root(double local)
{
#ifdef USE_MPI
	const int n = size();
	if (n <= 1) {
		std::vector<double> r;
		if (is_root()) r.push_back(local);
		return r;
	}
	if (is_root()) {
		std::vector<double> all((size_t)n, 0.0);
		MPI_Gather(&local, 1, MPI_DOUBLE, all.data(), 1, MPI_DOUBLE, 0, world());
		return all;
	}
	MPI_Gather(&local, 1, MPI_DOUBLE, (void*)NULL, 1, MPI_DOUBLE, 0, world());
	return std::vector<double>();
#else
	std::vector<double> r; r.push_back(local); return r;
#endif
}

inline std::vector<double> gather_vec_to_root(const std::vector<double>& local)
{
#ifdef USE_MPI
	const int n = size();
	const int k = (int)local.size();
	if (n <= 1) {
		if (is_root()) return local;
		return std::vector<double>();
	}
	if (is_root()) {
		std::vector<double> all((size_t)n * (size_t)k, 0.0);
		MPI_Gather(local.data(), k, MPI_DOUBLE,
		           all.data(),   k, MPI_DOUBLE, 0, world());
		return all;
	}
	MPI_Gather(local.data(), k, MPI_DOUBLE,
	           (void*)NULL, k, MPI_DOUBLE, 0, world());
	return std::vector<double>();
#else
	return local;
#endif
}

// Format per-rank vector (length N == ranks) as one compact line:
//   "rk0=12.34 rk1=11.98 ... max=12.34 min=11.98 imb=1.03x"
// Only meaningful on rank-0 (workers got empty vector from gather_to_root).
inline std::string format_per_rank(const std::vector<double>& all,
                                   const char* fmt /* e.g. "%.3f" */)
{
	if (all.empty()) return std::string();
	std::string out;
	char buf[64];
	double mn = all[0], mx = all[0];
	for (size_t i = 0; i < all.size(); ++i) {
		if (all[i] < mn) mn = all[i];
		if (all[i] > mx) mx = all[i];
		std::snprintf(buf, sizeof(buf), "rk%zu=", i);
		out += buf;
		std::snprintf(buf, sizeof(buf), fmt, all[i]);
		out += buf;
		out += ' ';
	}
	std::snprintf(buf, sizeof(buf), "max=");
	out += buf;
	std::snprintf(buf, sizeof(buf), fmt, mx);
	out += buf;
	out += " min=";
	std::snprintf(buf, sizeof(buf), fmt, mn);
	out += buf;
	out += " imb=";
	const double imb = (mn > 1e-12) ? (mx / mn) : 0.0;
	std::snprintf(buf, sizeof(buf), "%.2fx", imb);
	out += buf;
	return out;
}

}} // namespace MUSIC::mpi

#endif // __MPI_HELPER_HH
