/*
- List of some useful functions defined in ma_functions.c and elsewhere.
- Written by M. Ayromlou
*/
#ifndef LIST_FUNCTIONS_H
#define LIST_FUNCTIONS_H

#include <stddef.h>
#include "allvars.h"
#include <stdbool.h>
#include "hdf5.h"

void project(double pos_totpar[][3], float vel_totpar[][3], float metals_totpar[][n_metals], float mass_totpar[], float u_totpar[], float rho_totpar[], float NeutralHydrogenAbundance_totpar[], float StarFormationRate_totpar[], float x_e_totpar[], double L_bol_totpar[], float r_smoothing[], float r_g[][3], float v_g[][3], int Ndim, int n, float sfsqrt, int i_partType);
void read_in_particle_data(void);
void read_parameter_file(const char *fname);
double calculate_mean_molecular_weight_Arepo(double ElectronAbundance);
double calculate_temperature_Arepo(double InternalEnergy, double ElectronAbundance);
double convert_units_Arepo(double data, const char* quantity);
double calculate_Xray_BolometricLum(double ElectronAbundance, double InternalEnergy, double Masses, double Density, double StarFormationRate);
void mkdir_p(const char *dir);


hid_t ma_H5Gopen(hid_t loc_id, const char *h5_group_name);
hid_t ma_H5Aopen(hid_t loc_id, const char *h5_attribute_name);
hid_t ma_H5Dopen2(hid_t loc_id, const char *h5_dataset_name);
herr_t ma_H5Gclose(hid_t group_id);
herr_t ma_H5Aclose(hid_t attr_id);
herr_t ma_H5Dclose(hid_t dataset_id);
size_t ma_H5Aread(void *ptr, hid_t attribute_id, hid_t memory_type, size_t buffer_bytes);
size_t ma_H5Dread(void *ptr, hid_t dataset_id, hid_t memory_type, size_t buffer_bytes);

double *h5_alloc_read_2d_double(size_t N, size_t M, hid_t dset);
float *h5_alloc_read_2d_float(size_t N, size_t M, hid_t dset);

void profile_alloc(ProfileBins *pb_p, int Ndim, int nbins, int n_pt, int n_met, int pos_dim);
void profile_zero(ProfileBins *pb_p, int Ndim, int nbins, int n_pt, int n_met, int pos_dim);
void profile_free(ProfileBins *pb_p);
void profile_save_hdf5(ProfileBins *pb_p, const char *fpath, int Ndim, int nbins, int n_pt, int n_met, int pos_dim, const float *r_smoothing);

#endif
