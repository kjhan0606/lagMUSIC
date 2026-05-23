/*

 mg_profile.hh — minimal per-level/per-phase wall-time profiler for the
 multigrid V-cycle. Used to identify the n>=4 plateau bottleneck before
 deciding on task #11 Phase B (level-1 distribute) and Phase C (interp_cf
 distribute). Rank-0-only timing; tiny overhead per phase wrap.

 Usage:
   double t0 = MUSIC::mg_profile::tic();
   ... do work ...
   MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_LOCAL, t0);

   At end of solve():
   MUSIC::mg_profile::report("zoom-V-cycle", levelmin, levelmax);
   MUSIC::mg_profile::reset();

*/

#ifndef __MG_PROFILE_HH
#define __MG_PROFILE_HH

#include "general.hh"
#include "log.hh"
#include <map>
#include <vector>
#include <cstdio>

namespace MUSIC { namespace mg_profile {

enum Phase {
    P_INTERP_CF = 0,      // interp().interp_coarse_fine(level, uc, uf) — Phase C target
    P_SMOOTH_LOCAL,       // serial GS/Jacobi/SOR sweep
    P_SMOOTH_DIST,        // mg_scatter + mg_gs_sweep + mg_gather (finest only)
    P_PERIODIC,           // make_periodic(uf|uc|cc) at any site
    P_RESTRICT_U,         // m_gridop.restrict(uf, uc)
    P_APPLY_REST_LOCAL,   // serial apply+restrict (Lu/tLu) double-loop
    P_APPLY_REST_DIST,    // mg_apply_restrict (finest only)
    P_RESTRICT_F,         // m_gridop.restrict(ff, fc) + fc update loop
    P_PROLONG_CC,         // cc compute + m_gridop.prolong_add(cc, uf)
    P_MG_BEGIN_END,       // mg_begin / mg_end overhead (finest only)
    P_N
};

inline const char* phase_name(int p){
    static const char* names[] = {
        "interp_cf",
        "smooth_loc",
        "smooth_dist",
        "make_period",
        "restrict_u",
        "apply_loc",
        "apply_dist",
        "restrict_f",
        "prolong_cc",
        "mg_be"
    };
    return (p>=0 && p<P_N) ? names[p] : "?";
}

inline std::map<int, std::vector<double> >& accum(){
    static std::map<int, std::vector<double> > a;
    return a;
}

inline double tic(){
#ifdef WITH_MPI
    return MPI_Wtime();
#else
    return omp_get_wtime();
#endif
}

inline void add(int level, int phase, double t0){
    if( phase < 0 || phase >= P_N ) return;
    std::vector<double>& v = accum()[level];
    if( (int)v.size() < P_N ) v.assign(P_N, 0.0);
    v[phase] += tic() - t0;
}

inline void reset(){
    accum().clear();
}

inline void report(const char* tag, unsigned levelmin, unsigned levelmax){
#ifdef WITH_MPI
    int rk = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rk);
    if( rk != 0 ) return;
#endif
    std::map<int, std::vector<double> >& a = accum();
    if( a.empty() ){
        LOGINFO("mg_profile [%s]: (no samples)", tag);
        return;
    }

    // Per-phase totals across all levels
    double phase_total[P_N];
    for( int p=0; p<P_N; ++p ) phase_total[p] = 0.0;
    double grand_total = 0.0;
    for( std::map<int, std::vector<double> >::const_iterator it=a.begin(); it!=a.end(); ++it ){
        const std::vector<double>& v = it->second;
        for( int p=0; p<P_N && p<(int)v.size(); ++p ){
            phase_total[p] += v[p];
            grand_total   += v[p];
        }
    }

    LOGINFO("mg_profile [%s] — per-level wall time (s), excluding recurse:", tag);
    {
        char hdr[512];
        int n = snprintf(hdr, sizeof(hdr), "  L  ");
        for( int p=0; p<P_N; ++p )
            n += snprintf(hdr+n, sizeof(hdr)-n, "%11s", phase_name(p));
        n += snprintf(hdr+n, sizeof(hdr)-n, "%11s", "SUBTOTAL");
        LOGINFO("%s", hdr);
    }
    for( int L=(int)levelmax; L>=0; --L ){
        std::map<int, std::vector<double> >::const_iterator it = a.find(L);
        if( it == a.end() ) continue;
        const std::vector<double>& v = it->second;
        char row[512];
        int n = snprintf(row, sizeof(row), "  %-3d", L);
        double sub = 0.0;
        for( int p=0; p<P_N; ++p ){
            double x = (p<(int)v.size()) ? v[p] : 0.0;
            sub += x;
            n += snprintf(row+n, sizeof(row)-n, "%11.4f", x);
        }
        n += snprintf(row+n, sizeof(row)-n, "%11.4f", sub);
        LOGINFO("%s", row);
    }
    {
        char row[512];
        int n = snprintf(row, sizeof(row), "  TOT");
        for( int p=0; p<P_N; ++p )
            n += snprintf(row+n, sizeof(row)-n, "%11.4f", phase_total[p]);
        n += snprintf(row+n, sizeof(row)-n, "%11.4f", grand_total);
        LOGINFO("%s", row);
    }
    LOGINFO("mg_profile [%s] — phase fractions of accounted-for time (%%):", tag);
    {
        char row[512];
        int n = snprintf(row, sizeof(row), "  pct");
        for( int p=0; p<P_N; ++p )
            n += snprintf(row+n, sizeof(row)-n, "%10.1f%%",
                          grand_total>0 ? 100.0*phase_total[p]/grand_total : 0.0);
        LOGINFO("%s", row);
    }
    (void)levelmin;
}

}}  // namespace MUSIC::mg_profile

#endif // __MG_PROFILE_HH
