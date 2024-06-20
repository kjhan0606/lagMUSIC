/*

 output_gamer2.cc - This file is part of MUSIC -
 a code to generate multi-scale initial conditions
 for cosmological simulations

 Copyright (C) 2010-13  Oliver Hahn

 Plugin: Yuri Oku (yuri.oku.astro@gmail.com)

 */

#include "output.hh"

class gamer2_output_plugin : public output_plugin
{
  protected:
    enum iofields
    {
        id_dm_mass,
        id_dm_pos,
        id_dm_vel,
        id_dm_type,
        id_gas_rho,
        id_gas_vel,
        id_gas_temp
    };

    const float GAMER_DARK_MATTER_TYPE = 1;

    std::string par_ic_;
    std::string um_ic_;

    double boxlength_;
    double zstart_;
    double omegam_;
    double omegab_;
    double hubble_;
    bool bbaryons_;

    unsigned long npart_;
    unsigned long nmesh_;

    real_t gamer_unit_length_;
    real_t gamer_unit_time_;
    real_t gamer_unit_density_;
    real_t gamer_unit_mass_;
    real_t gamer_unit_velocity_;

    real_t music_unit_length_;
    real_t music_unit_density_;
    real_t music_unit_mass_;
    real_t music_unit_velocity_;

    size_t write2tempfile(std::string fname, const grid_hierarchy &gh, unsigned ilevel, real_t fac = 1.0,
                          real_t shift = 0.0)
    {
        const MeshvarBnd<real_t> *data = gh.get_grid(ilevel);

        int n0 = data->size(0), n1 = data->size(1), n2 = data->size(2);
        std::vector<real_t> vdata;
        vdata.reserve((unsigned)(n0) * (n1) * (n2));

        size_t count = 0;
        for (int i = 0; i < n0; ++i)
            for (int j = 0; j < n1; ++j)
                for (int k = 0; k < n2; ++k)
                    if (gh.is_in_mask(ilevel, i, j, k) && !gh.is_refined(ilevel, i, j, k))
                    {
                        vdata.push_back((*data)(i, j, k) * fac + shift);

                        count++;
                    }

        std::ofstream ofs_temp(fname, std::ios::binary | std::ios::trunc);
        ofs_temp.write((char *)&vdata[0], vdata.size() * sizeof(real_t));

        ofs_temp.flush();
        ofs_temp.close();

        return count;
    }

    size_t copy2outputfile(std::ofstream &ofs, std::string temp_fname)
    {
        std::ifstream ifs(temp_fname, std::ios::binary);
        if (!ifs)
        {
            std::cerr << "Error: Could not open file " << temp_fname << std::endl;
            exit(1);
        }

        ofs << ifs.rdbuf();

        /* count elements */
        ifs.seekg(0, std::ios::end);
        size_t count = ifs.tellg() / sizeof(real_t);

        ifs.close();
        return count;
    }

    void assemble_gamer2_file()
    {
        /* ----- write particle file ----- */
        std::ofstream ofs_par(par_ic_.c_str(), std::ios::binary | std::ios::trunc);
        std::cout << "=============================================================\n";
        LOGINFO("GAMER-2 plugin: writing %s", par_ic_.c_str());

        /* output mass */
        size_t npart = 0;
        for (unsigned ilevel = levelmax_; ilevel >= levelmin_; --ilevel)
        {
            std::string temp_fname = "___ic_temp_" + std::to_string(1000 * id_dm_mass + ilevel) + ".bin";

            size_t npart_ilevel = copy2outputfile(ofs_par, temp_fname);

            LOGINFO("Level %2d: %12llu particles", ilevel, npart_ilevel);
            npart += npart_ilevel;

            remove(temp_fname.c_str());
        }

        LOGINFO("Total: %12llu particles", npart);

        /* output position */
        for (unsigned coord = 0; coord < 3; ++coord)
        {
            for (unsigned ilevel = levelmax_; ilevel >= levelmin_; --ilevel)
            {
                std::string temp_fname =
                    "___ic_temp_" + std::to_string(1000 * id_dm_pos + 100 * coord + ilevel) + ".bin";

                copy2outputfile(ofs_par, temp_fname);

                remove(temp_fname.c_str());
            }
        }

        /* output velocity */
        for (unsigned coord = 0; coord < 3; ++coord)
        {
            for (unsigned ilevel = levelmax_; ilevel >= levelmin_; --ilevel)
            {
                std::string temp_fname =
                    "___ic_temp_" + std::to_string(1000 * id_dm_vel + 100 * coord + ilevel) + ".bin";

                copy2outputfile(ofs_par, temp_fname);

                remove(temp_fname.c_str());
            }
        }

        /* output particle type */
        for (unsigned ilevel = levelmax_; ilevel >= levelmin_; --ilevel)
        {
            std::string temp_fname = "___ic_temp_" + std::to_string(1000 * id_dm_type + ilevel) + ".bin";

            copy2outputfile(ofs_par, temp_fname);

            remove(temp_fname.c_str());
        }

        ofs_par.flush();
        ofs_par.close();

        /* ----- write mesh file ----- */
        std::ofstream ofs_um(um_ic_.c_str(), std::ios::binary | std::ios::trunc);

        /* output density */
        for (unsigned ilevel = levelmin_; ilevel <= levelmax_; ++ilevel)
        {
            std::string temp_fname = "___ic_temp_" + std::to_string(1000 * id_gas_rho + ilevel) + ".bin";

            remove(temp_fname.c_str());
        }

        /* output velocity */
        for (unsigned coord = 0; coord < 3; ++coord)
        {
            for (unsigned ilevel = levelmin_; ilevel <= levelmax_; ++ilevel)
            {
                std::string temp_fname =
                    "___ic_temp_" + std::to_string(1000 * id_gas_vel + 100 * coord + ilevel) + ".bin";

                remove(temp_fname.c_str());
            }
        }

        ofs_um.flush();
        ofs_um.close();
    }

  public:
    gamer2_output_plugin(config_file &cf) // std::string fname, Cosmology cosm, Parameters param )
        : output_plugin(cf)               // fname, cosm, param )
    {
        par_ic_ = cf.getValue<std::string>("output", "parfilename");
        um_ic_  = cf.getValue<std::string>("output", "filename");

        boxlength_ = cf.getValue<double>("setup", "boxlength");
        zstart_    = cf.getValue<double>("setup", "zstart");
        bbaryons_  = cf.getValue<bool>("setup", "baryons");
        omegam_    = cf.getValue<double>("cosmology", "Omega_m");
        omegab_    = cf.getValue<double>("cosmology", "Omega_b");
        hubble_    = cf.getValue<double>("cosmology", "H0") / 100.0;

        // "GAMER" physical constants (/include/PhysicalConstant.h), input
        // parameters (Input__Parameter), and COMOVING units
        // (src/Init/Init_Unit.cpp)
        double const_cm   = 1.0;
        double const_km   = 1.0e5 * const_cm;
        double const_pc   = 3.08567758149e18 * const_cm; // parsec
        double const_Mpc  = 1.0e6 * const_pc;
        double const_s    = 1.0;                                                // second
        double const_Msun = 1.9885e33;                                          // solar mass
        double const_G    = 6.6738e-8;                                          // gravitational constant in cgs
        double H0         = 100.0 * hubble_ * const_km / (const_s * const_Mpc); // H0 = 100*h*km/(s*Mpc)
        // see
        // https://github.com/gamer-project/gamer/wiki/Runtime-Parameters%3A-Units#units-in-cosmological-simulations
        gamer_unit_length_   = const_Mpc / hubble_;
        gamer_unit_time_     = 1.0 / H0;
        gamer_unit_density_  = 3.0 * omegam_ * H0 * H0 / (8.0 * M_PI * const_G);
        gamer_unit_mass_     = gamer_unit_density_ * gamer_unit_length_ * gamer_unit_length_ * gamer_unit_length_;
        gamer_unit_velocity_ = gamer_unit_length_ / gamer_unit_time_;

        // MUSIC units
        music_unit_length_   = const_Mpc / hubble_;
        music_unit_mass_     = const_Msun / hubble_;
        music_unit_density_  = music_unit_mass_ / (music_unit_length_ * music_unit_length_ * music_unit_length_);
        music_unit_velocity_ = 1.0e5 * boxlength_;
    }

    ~gamer2_output_plugin()
    {
    }

    void write_dm_density(const grid_hierarchy &gh)
    {
        // TODO: write refinement mask
    }

    void write_dm_mass(const grid_hierarchy &gh)
    {
        for (unsigned ilevel = levelmin_; ilevel <= levelmax_; ++ilevel)
        {
            /* uniform particle mass for each level */
            double dx   = boxlength_ / (double)(1 << ilevel);
            double dx3  = dx * dx * dx;
            double rhom = 2.77519737e11; // h-1 M_o / (h-1 Mpc)**3
            real_t cmass;

            if (bbaryons_)
                cmass = (omegam_ - omegab_) * rhom * dx3 * music_unit_mass_ / gamer_unit_mass_;
            else
                cmass = omegam_ * rhom * dx3 * music_unit_mass_ / gamer_unit_mass_;

            std::string temp_fname = "___ic_temp_" + std::to_string(1000 * id_dm_mass + ilevel) + ".bin";

            write2tempfile(temp_fname, gh, ilevel, 0, cmass);
        }

        for (unsigned ilevel = levelmin_; ilevel <= levelmax_; ++ilevel)
        {
            std::string temp_fname = "___ic_temp_" + std::to_string(1000 * id_dm_type + ilevel) + ".bin";

            /* generate uniform array of GAMER_DARK_MATTER_TYPE */
            write2tempfile(temp_fname, gh, ilevel, 0, GAMER_DARK_MATTER_TYPE);
        }
    }

    void write_dm_position(int coord, const grid_hierarchy &gh)
    {
        for (unsigned ilevel = levelmin_; ilevel <= levelmax_; ++ilevel)
        {
            const MeshvarBnd<real_t> *data = gh.get_grid(ilevel);

            int n0 = data->size(0), n1 = data->size(1), n2 = data->size(2);
            std::vector<real_t> vdata;
            vdata.reserve((unsigned)(n0) * (n1) * (n2));

            size_t count = 0;
            real_t fac   = music_unit_length_ / gamer_unit_length_;
            for (int i = 0; i < n0; ++i)
                for (int j = 0; j < n1; ++j)
                    for (int k = 0; k < n2; ++k)
                        if (gh.is_in_mask(ilevel, i, j, k) && !gh.is_refined(ilevel, i, j, k))
                        {
                            /* convert displacement to particle position */
                            double xx[3];
                            gh.cell_pos(ilevel, i, j, k, xx);
                            real_t pos = (xx[coord] + (*data)(i, j, k)) * boxlength_;

                            /* apply periodic boundary condition */
                            while (pos < 0)
                                pos += boxlength_;
                            while (pos >= boxlength_)
                                pos -= boxlength_;

                            vdata.push_back(pos * fac);

                            count++;
                        }

            std::string temp_fname = "___ic_temp_" + std::to_string(1000 * id_dm_pos + 100 * coord + ilevel) + ".bin";

            std::ofstream ofs_temp(temp_fname, std::ios::binary | std::ios::trunc);
            ofs_temp.write((char *)&vdata[0], vdata.size() * sizeof(real_t));

            ofs_temp.flush();
            ofs_temp.close();
        }
    }

    void write_dm_velocity(int coord, const grid_hierarchy &gh)
    {
        for (unsigned ilevel = levelmin_; ilevel <= levelmax_; ++ilevel)
        {
            std::string temp_fname = "___ic_temp_" + std::to_string(1000 * id_dm_vel + 100 * coord + ilevel) + ".bin";

            real_t fac = music_unit_velocity_ / gamer_unit_velocity_;
            write2tempfile(temp_fname, gh, ilevel, fac, 0);
        }
    }

    void write_dm_potential(const grid_hierarchy &gh)
    { /* skip */
    }

    void write_gas_potential(const grid_hierarchy &gh)
    { /* skip */
    }

    void write_gas_position(int coord, const grid_hierarchy &gh)
    { /* not used */
    }

    void write_gas_density(const grid_hierarchy &gh)
    {
        for (unsigned ilevel = levelmin_; ilevel <= levelmax_; ++ilevel)
        {
            std::string temp_fname = "___ic_temp_" + std::to_string(1000 * id_gas_rho + ilevel) + ".bin";

            real_t fac = music_unit_density_ / gamer_unit_density_;
            write2tempfile(temp_fname, gh, ilevel, fac, 0);
        }
    }

    void write_gas_velocity(int coord, const grid_hierarchy &gh)
    {
        for (unsigned ilevel = levelmin_; ilevel <= levelmax_; ++ilevel)
        {
            std::string temp_fname = "___ic_temp_" + std::to_string(1000 * id_gas_vel + 100 * coord + ilevel) + ".bin";

            real_t fac = music_unit_velocity_ / gamer_unit_velocity_;
            write2tempfile(temp_fname, gh, ilevel, fac, 0);
        }
    }

    void finalize(void)
    {
        assemble_gamer2_file();
    }
};

namespace
{
output_plugin_creator_concrete<gamer2_output_plugin> creator("gamer2");
}
