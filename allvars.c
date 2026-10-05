
#include "allvars.h"



int nbins_profile;
int nRvir_profile;

int *pos_totpar_c;
int *cpos_totpar_c;
int *pos_totpar_i;
int *binpos_totpar;
double gr_M200_minlim;
double gr_M200_maxlim;
float h_const;
float Hubble_Parameter;
float Omega_m;
float Omega_dm;
float Omega_b;
float Omega_Lambda;
float Omega_r;
float Omega_k;

int n_snaps;
int n_b_max;
int n_b_center;
int nfiles_par;
int n_partType;
int n_partType_all;
double dm_mass;
double *particle_mass_table;
int dm_partType;
float m_p;
float k_B;
float gamma_gas;
float X_H;

int *gal_bin_x;
int *gal_bin_y;
int *gal_bin_z;
int *n_par_in_gal;
int *np_bg_limit;

ProfileBins pb;

double Temp_lim1;
double Temp_lim2;
double Temp_lim3;
double temp_starForming_gas_cell;
double v_rad_lim1;
double v_rad_lim2;
double v_rad_lim3;
float x_g;
float y_g;
float z_g;
float x_g1;
float y_g1;
float z_g1;

float vx_g;
float vy_g;
float vz_g;
float sf;
float sfsqrt;
double sim_redshift;
int particle_for_loop_step;



double l_bin;
int nb;
int k1;
int k2;
int k3;
int np_in_bin;
int *gal_bin_x1;
int *gal_bin_y1;
int *gal_bin_z1;
int bin_idx;
int bin_idx1;
double box_size;

int par_in_r_size;
int nb_total;


long long n_galpar_counter;
int pos_dimension;
int gal_dimension;
int n_metals;
int projection_flag;

char SimulationDir[512];
char OutputDir[512];
char InputDir[512];


int SnapNum;
int SnapNum_i;
int SnapNum_f;
int subfilenr;
int subfilenr_i;
int subfilenr_f;
int filenr_i;
int part_in_profile_method;
int *fileStat_sim;
int *first_free_part_filenr_this_snap;
long long *first_free_part_idx_in_file_this_snap;
int n_subfile;


int calculate_all_gas_properties;
int is_hydro_sim; /* 1 for hydro sims, 0 for DMO */

int output_np_mass;
int output_vrad;
int output_vLOS;
int output_outflow;
int output_outflow_fast;
int output_flow_vectors;
int output_outflow_vectors;
int output_temperature;
int output_xray;
int output_ionization;
int output_metals;
int output_metals_xray_weighted;

/* Simulation identification and cosmology from the parameter file. */
char SimulationLabel[1000];
int  cluster_subfile_mode;
double cosmo_h;
double cosmo_Omega_m;
double cosmo_Omega_b;
double cosmo_Omega_Lambda;

hid_t h5_file;
hid_t h5_group;
hid_t h5_dataset;
hid_t h5_space;
herr_t h5_status;
hid_t h5_header_group;
hid_t h5_attr_dummy;