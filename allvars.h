#ifndef ALLVARS_H
#define ALLVARS_H

#include <stdio.h>
#include "hdf5.h"

extern int nbins_profile;
extern int nRvir_profile;

extern int *pos_totpar_c;
extern int *cpos_totpar_c;
extern int *pos_totpar_i;
extern int *binpos_totpar;
extern double gr_M200_minlim;
extern double gr_M200_maxlim;
extern float h_const;
extern float Hubble_Parameter;
extern float Omega_m;
extern float Omega_dm;
extern float Omega_b;
extern float Omega_Lambda;
extern float Omega_r;
extern float Omega_k;

extern int n_snaps;
extern int n_b_max;
extern int n_b_center;
extern int nfiles_par;
extern int n_partType;
extern int n_partType_all;
extern double dm_mass;
extern double *particle_mass_table;
extern int dm_partType;
extern float m_p;
extern float k_B;
extern float gamma_gas;
extern float X_H;

extern int *gal_bin_x;
extern int *gal_bin_y;
extern int *gal_bin_z;
extern int *n_par_in_gal;
extern int *np_bg_limit;

/* All 46 per-snapshot profile accumulation arrays, grouped into one struct.
 * Allocation: n_pt  = n_partType,  Nd = Ndim,  nb = nbins_profile,
 *             pd    = pos_dimension, nm = n_metals.
 * Flat index for family A/B:  i_pt*Nd*nb + numgal*nb + i_bin
 * Flat index for family C:    i_coord*n_pt*Nd*nb + i_pt*Nd*nb + numgal*nb + i_bin
 * Flat index for family D:    numgal*nb + i_bin
 * Flat index for family E:    k_metal*Nd*nb + numgal*nb + i_bin
 */
typedef struct {
    /* ---- Family A: int,    n_pt * Nd * nb ---- */
    int    *np_bg;
    int    *num_vrad_out;
    /* ---- Family B: double, n_pt * Nd * nb ---- */
    double *mass_bg;
    double *vrad_times_mass;
    double *vLOS_x;
    double *vLOS_y;
    double *vLOS_z;
    double *mass_out;
    double *mass_out_fast1;
    double *mass_out_fast2;
    double *mass_out_fast3;
    double *vrad_times_mass_out;
    double *vrad_times_mass_out_fast1;
    double *vrad_times_mass_out_fast2;
    double *vrad_times_mass_out_fast3;
    double *vLOS_x_out;
    double *vLOS_y_out;
    double *vLOS_z_out;
    /* ---- Family C: double, pd * n_pt * Nd * nb ---- */
    double *vLOS;
    double *matter_flow_vector;
    double *matter_outflow_vector;
    double *vLOS_out;
    /* ---- Family D: double, Nd * nb ---- */
    double *temp_bg;
    double *entropy;
    double *L_bol_bg;
    double *L_bol_times_mass_bg;
    double *N_e;
    double *N_H0;
    double *N_H;
    double *temp_out;
    double *temp_out_fast1;
    double *temp_out_fast2;
    double *temp_out_fast3;
    double *mass_bg_TempAboveLim1;
    double *mass_bg_TempAboveLim2;
    double *mass_bg_TempAboveLim3;
    double *mass_bg_out_TempAboveLim1;
    double *mass_bg_out_TempAboveLim2;
    double *mass_bg_out_TempAboveLim3;
    /* ---- Family E: double, nm * Nd * nb ---- */
    double *metals_all;
    double *metals_all_emission_weighted;
    double *metals_all_emission_weighted_e2;
    double *metals_all_out;
    double *metals_all_out_fast1;
    double *metals_all_out_fast2;
    double *metals_all_out_fast3;
} ProfileBins;

extern ProfileBins pb;

extern double Temp_lim1;
extern double Temp_lim2;
extern double Temp_lim3;
extern double temp_starForming_gas_cell;
extern double v_rad_lim1;
extern double v_rad_lim2;
extern double v_rad_lim3;
extern float x_g;
extern float y_g;
extern float z_g;
extern float x_g1;
extern float y_g1;
extern float z_g1;

extern float vx_g;
extern float vy_g;
extern float vz_g;
extern float sf;
extern float sfsqrt;
extern double sim_redshift;
extern int particle_for_loop_step;

extern double l_bin;
extern int nb;
extern int k1;
extern int k2;
extern int k3;
extern int np_in_bin;
extern int *gal_bin_x1;
extern int *gal_bin_y1;
extern int *gal_bin_z1;
extern int bin_idx;
extern int bin_idx1;
extern double box_size;
extern int par_in_r_size;
extern int nb_total;


extern char   SimulationDir[512];
extern char   OutputDir[512];
extern char   InputDir[512];



extern int    SnapNum;
extern int SnapNum_i;
extern int SnapNum_f;
extern int subfilenr;
extern int subfilenr_i;
extern int subfilenr_f;
extern int filenr_i;
extern int part_in_profile_method;
extern int *fileStat_sim;
extern int *first_free_part_filenr_this_snap;
extern long long *first_free_part_idx_in_file_this_snap;
extern int n_subfile;


extern long long n_galpar_counter;
extern int pos_dimension;
extern int gal_dimension;
extern int n_metals;
extern int projection_flag;



extern int calculate_all_gas_properties;
extern int is_hydro_sim; /* 1 for hydro sims, 0 for DMO */

/* Output-selection flags (all default to 1 = compute/save everything,
 * matching legacy behaviour). See read_parameters.c for the optional
 * parameter-file tags that control these. */
extern int output_np_mass;             /* np, mass (Family A/B core) */
extern int output_vrad;                /* vrad_times_mass */
extern int output_vLOS;                /* vLOS_x, vLOS_y, vLOS_z */
extern int output_outflow;              /* np_out, mass_out, vrad_times_mass_out, vLOS_*_out */
extern int output_outflow_fast;         /* *_fast1/2/3 velocity-threshold variants (requires output_outflow) */
extern int output_flow_vectors;         /* Family C: vLOS, matter_flow_vec */
extern int output_outflow_vectors;      /* Family C: vLOS_out, matter_outflow_vec */
extern int output_temperature;          /* temp, entropy, temp_out*, mass_*TempAboveLim* */
extern int output_xray;                 /* L_bol, L_bol_times_mass */
extern int output_ionization;           /* N_e, N_H0, N_H */
extern int output_metals;               /* metals_all, metals_all_out* */
extern int output_metals_xray_weighted; /* metals_all_emission_weighted* (requires output_metals && output_xray) */

/* Simulation identification and cosmology settings from the parameter file. */
extern char SimulationLabel[1000];     /* e.g. "IllustrisTNG/TNG_100" */
extern int  cluster_subfile_mode;      /* 1 for TNG-Cluster-like subfile splitting */
extern double cosmo_h;                 /* cosmology override (0 = use default) */
extern double cosmo_Omega_m;
extern double cosmo_Omega_b;
extern double cosmo_Omega_Lambda;

extern hid_t h5_file;
extern hid_t h5_group;
extern hid_t h5_dataset;
extern hid_t h5_space;
extern herr_t h5_status;
extern hid_t h5_header_group;
extern hid_t h5_attr_dummy;



#endif





