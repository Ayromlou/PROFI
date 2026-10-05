/*
- Some useful functions.
- Written by M. Ayromlou
*/
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "allvars.h"
#include "list_functions.h"
#include <omp.h>
#include "hdf5.h"

static void ma_h5_fatal(const char *operation, const char *object_type, const char *object_name)
{
	if(object_name)
		fprintf(stderr, "HDF5 error: failed to %s %s '%s'\n", operation, object_type, object_name);
	else
		fprintf(stderr, "HDF5 error: failed to %s %s\n", operation, object_type);
	fflush(stderr);
	H5Eprint2(H5E_DEFAULT, stderr);
	/* HDF5's exit handlers can retry a failed write/close on damaged state. */
	fflush(NULL);
	_Exit(1);
}

hid_t ma_H5Gopen(hid_t loc_id, const char *h5_group_name)
{
	hid_t h5_group_id;
	h5_group_id = H5Gopen(loc_id, h5_group_name, H5P_DEFAULT);
	if (h5_group_id < 0)
		ma_h5_fatal("open", "group", h5_group_name);
	return h5_group_id;
}

hid_t ma_H5Aopen(hid_t loc_id, const char *h5_attribute_name)
{
	hid_t h5_attribute_id;
	h5_attribute_id = H5Aopen(loc_id, h5_attribute_name, H5P_DEFAULT);
	if (h5_attribute_id < 0)
		ma_h5_fatal("open", "attribute", h5_attribute_name);
	return h5_attribute_id;
}

hid_t ma_H5Dopen2(hid_t loc_id, const char *h5_dataset_name)
{
	hid_t h5_dataset_id;
	h5_dataset_id = H5Dopen2(loc_id, h5_dataset_name, H5P_DEFAULT);
	if (h5_dataset_id < 0)
		ma_h5_fatal("open", "dataset", h5_dataset_name);
	return h5_dataset_id;
}

/**@brief Modified versions of H5's closure routines for groups, attributes and datasets (checks that they are closed properly). */

herr_t ma_H5Gclose(hid_t group_id)
{
	herr_t h5_err;
	char h5_group_name[200] = {0};
	h5_err = H5Gclose(group_id);
	if (h5_err < 0)
	{
		H5Iget_name(group_id, h5_group_name, 200);
		ma_h5_fatal("close", "group", h5_group_name);
	}
	return h5_err;
}

herr_t ma_H5Aclose(hid_t attr_id)
{
	herr_t h5_err;
	char h5_attribute_name[200] = {0};
	h5_err = H5Aclose(attr_id);
	if (h5_err < 0)
	{
		H5Aget_name(attr_id, 200, h5_attribute_name);
		ma_h5_fatal("close", "attribute", h5_attribute_name);
	}
	return h5_err;
}

herr_t ma_H5Dclose(hid_t dataset_id)
{
	herr_t h5_err;
	char h5_dataset_name[200] = {0};
	h5_err = H5Dclose(dataset_id);
	if (h5_err < 0)
	{
		H5Iget_name(dataset_id, h5_dataset_name, 200);
		ma_h5_fatal("close", "dataset", h5_dataset_name);
	}
	return h5_err;
}

/* Require the complete converted payload to match the destination buffer. */
static size_t ma_h5_read_bytes(hid_t space, hid_t memory_type, size_t buffer_bytes,
                             void *ptr, const char *object_type, const char *object_name)
{
	hssize_t n_points = H5Sget_simple_extent_npoints(space);
	size_t type_size = H5Tget_size(memory_type);
	if(n_points < 0 || type_size == 0 || (uintmax_t) n_points > SIZE_MAX / type_size)
		ma_h5_fatal("compute memory payload size for", object_type, object_name);

	size_t nread = (size_t) n_points * type_size;
	if(nread != buffer_bytes) {
		fprintf(stderr, "HDF5 size mismatch for %s '%s': payload needs %zu memory bytes, buffer has %zu\n",
		        object_type, object_name, nread, buffer_bytes);
		ma_h5_fatal("validate destination size for", object_type, object_name);
	}
	if(nread != 0 && ptr == NULL)
		ma_h5_fatal("use a NULL destination for", object_type, object_name);
	return nread;
}

/* Read using the caller's destination datatype; return memory bytes read.
 * memory_type is caller-owned, so predefined native datatypes are not closed. */
size_t ma_H5Aread(void *ptr, hid_t attribute_id, hid_t memory_type, size_t buffer_bytes)
{
	hid_t h5_data_space = H5Aget_space(attribute_id);
	if(h5_data_space < 0)
		ma_h5_fatal("query dataspace for", "attribute", NULL);
	char h5_attribute_name[200] = {0};
	H5Aget_name(attribute_id, sizeof h5_attribute_name, h5_attribute_name);
	size_t nread = ma_h5_read_bytes(h5_data_space, memory_type, buffer_bytes,
	                              ptr, "attribute", h5_attribute_name);
	if(nread != 0 && H5Aread(attribute_id, memory_type, ptr) < 0)
		ma_h5_fatal("read", "attribute", h5_attribute_name);
	if(H5Sclose(h5_data_space) < 0)
		ma_h5_fatal("close dataspace for", "attribute", h5_attribute_name);
	return nread;
}

size_t ma_H5Dread(void *ptr, hid_t dataset_id, hid_t memory_type, size_t buffer_bytes)
{
	hid_t h5_data_space = H5Dget_space(dataset_id);
	if(h5_data_space < 0)
		ma_h5_fatal("query dataspace for", "dataset", NULL);
	char h5_dataset_name[200] = {0};
	H5Iget_name(dataset_id, h5_dataset_name, sizeof h5_dataset_name);
	size_t nread = ma_h5_read_bytes(h5_data_space, memory_type, buffer_bytes,
	                              ptr, "dataset", h5_dataset_name);
	if(nread != 0 && H5Dread(dataset_id, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr) < 0)
		ma_h5_fatal("read", "dataset", h5_dataset_name);
	if(H5Sclose(h5_data_space) < 0)
		ma_h5_fatal("close dataspace for", "dataset", h5_dataset_name);
	return nread;
}


/* =====================================================================
 * ProfileBins helpers
 * ===================================================================== */

void profile_alloc(ProfileBins *pb_p, int Ndim, int nbins,
                   int n_pt, int n_met, int pos_dim)
{
    size_t sz_A  = (size_t)n_pt  * (size_t)Ndim * (size_t)nbins;
    size_t sz_C  = (size_t)pos_dim * sz_A;
    size_t sz_D  = (size_t)Ndim * (size_t)nbins;
    size_t sz_E  = (size_t)n_met * sz_D;

#define PALLOC_INT(field, count) \
    do { pb_p->field = malloc((count) * sizeof(int)); \
         if (!pb_p->field) { fprintf(stderr, "profile_alloc: out of memory for " #field "\n"); exit(1); } \
    } while (0)
#define PALLOC_DBL(field, count) \
    do { pb_p->field = malloc((count) * sizeof(double)); \
         if (!pb_p->field) { fprintf(stderr, "profile_alloc: out of memory for " #field "\n"); exit(1); } \
    } while (0)

    /* Family A */
    if (output_np_mass) { PALLOC_INT(np_bg, sz_A); } else { pb_p->np_bg = NULL; }
    if (output_outflow)  { PALLOC_INT(num_vrad_out, sz_A); } else { pb_p->num_vrad_out = NULL; }
    /* Family B */
    if (output_np_mass) { PALLOC_DBL(mass_bg, sz_A); } else { pb_p->mass_bg = NULL; }
    if (output_vrad)    { PALLOC_DBL(vrad_times_mass, sz_A); } else { pb_p->vrad_times_mass = NULL; }
    if (output_vLOS) {
        PALLOC_DBL(vLOS_x, sz_A);
        PALLOC_DBL(vLOS_y, sz_A);
        PALLOC_DBL(vLOS_z, sz_A);
    } else {
        pb_p->vLOS_x = pb_p->vLOS_y = pb_p->vLOS_z = NULL;
    }
    if (output_outflow) {
        PALLOC_DBL(mass_out,             sz_A);
        PALLOC_DBL(vrad_times_mass_out,  sz_A);
        PALLOC_DBL(vLOS_x_out,           sz_A);
        PALLOC_DBL(vLOS_y_out,           sz_A);
        PALLOC_DBL(vLOS_z_out,           sz_A);
    } else {
        pb_p->mass_out = pb_p->vrad_times_mass_out = NULL;
        pb_p->vLOS_x_out = pb_p->vLOS_y_out = pb_p->vLOS_z_out = NULL;
    }
    if (output_outflow_fast) {
        PALLOC_DBL(mass_out_fast1,            sz_A);
        PALLOC_DBL(mass_out_fast2,            sz_A);
        PALLOC_DBL(mass_out_fast3,            sz_A);
        PALLOC_DBL(vrad_times_mass_out_fast1, sz_A);
        PALLOC_DBL(vrad_times_mass_out_fast2, sz_A);
        PALLOC_DBL(vrad_times_mass_out_fast3, sz_A);
    } else {
        pb_p->mass_out_fast1 = pb_p->mass_out_fast2 = pb_p->mass_out_fast3 = NULL;
        pb_p->vrad_times_mass_out_fast1 = pb_p->vrad_times_mass_out_fast2 = pb_p->vrad_times_mass_out_fast3 = NULL;
    }
    /* Family C */
    if (output_flow_vectors) {
        PALLOC_DBL(vLOS,               sz_C);
        PALLOC_DBL(matter_flow_vector, sz_C);
    } else {
        pb_p->vLOS = pb_p->matter_flow_vector = NULL;
    }
    if (output_outflow_vectors) {
        PALLOC_DBL(matter_outflow_vector, sz_C);
        PALLOC_DBL(vLOS_out,              sz_C);
    } else {
        pb_p->matter_outflow_vector = pb_p->vLOS_out = NULL;
    }

    /* Family D/E (hydro only, and only if the caller actually wants these
     * gas properties computed — otherwise leave NULL so profile_zero,
     * profile_save_hdf5, and project() all correctly skip them and we
     * don't allocate/write large all-zero arrays for nothing). */
    int want_temperature = is_hydro_sim && calculate_all_gas_properties && output_temperature;
    int want_xray        = is_hydro_sim && calculate_all_gas_properties && output_xray;
    int want_ionization   = is_hydro_sim && calculate_all_gas_properties && output_ionization;
    int want_metals       = is_hydro_sim && calculate_all_gas_properties && output_metals;
    int want_metals_xray_weighted = want_metals && want_xray && output_metals_xray_weighted;

    if (want_temperature) {
        PALLOC_DBL(temp_bg,                 sz_D);
        PALLOC_DBL(entropy,                 sz_D);
        PALLOC_DBL(temp_out,                sz_D);
        PALLOC_DBL(temp_out_fast1,          sz_D);
        PALLOC_DBL(temp_out_fast2,          sz_D);
        PALLOC_DBL(temp_out_fast3,          sz_D);
        PALLOC_DBL(mass_bg_TempAboveLim1,   sz_D);
        PALLOC_DBL(mass_bg_TempAboveLim2,   sz_D);
        PALLOC_DBL(mass_bg_TempAboveLim3,   sz_D);
        PALLOC_DBL(mass_bg_out_TempAboveLim1, sz_D);
        PALLOC_DBL(mass_bg_out_TempAboveLim2, sz_D);
        PALLOC_DBL(mass_bg_out_TempAboveLim3, sz_D);
    } else {
        pb_p->temp_bg = pb_p->entropy = NULL;
        pb_p->temp_out = pb_p->temp_out_fast1 = pb_p->temp_out_fast2 = pb_p->temp_out_fast3 = NULL;
        pb_p->mass_bg_TempAboveLim1 = pb_p->mass_bg_TempAboveLim2 = pb_p->mass_bg_TempAboveLim3 = NULL;
        pb_p->mass_bg_out_TempAboveLim1 = pb_p->mass_bg_out_TempAboveLim2 = pb_p->mass_bg_out_TempAboveLim3 = NULL;
    }
    if (want_xray) {
        PALLOC_DBL(L_bol_bg,             sz_D);
        PALLOC_DBL(L_bol_times_mass_bg,  sz_D);
    } else {
        pb_p->L_bol_bg = pb_p->L_bol_times_mass_bg = NULL;
    }
    if (want_ionization) {
        PALLOC_DBL(N_e,  sz_D);
        PALLOC_DBL(N_H0, sz_D);
        PALLOC_DBL(N_H,  sz_D);
    } else {
        pb_p->N_e = pb_p->N_H0 = pb_p->N_H = NULL;
    }
    if (want_metals) {
        PALLOC_DBL(metals_all,           sz_E);
        PALLOC_DBL(metals_all_out,       sz_E);
        PALLOC_DBL(metals_all_out_fast1, sz_E);
        PALLOC_DBL(metals_all_out_fast2, sz_E);
        PALLOC_DBL(metals_all_out_fast3, sz_E);
    } else {
        pb_p->metals_all = pb_p->metals_all_out = NULL;
        pb_p->metals_all_out_fast1 = pb_p->metals_all_out_fast2 = pb_p->metals_all_out_fast3 = NULL;
    }
    if (want_metals_xray_weighted) {
        PALLOC_DBL(metals_all_emission_weighted,    sz_E);
        PALLOC_DBL(metals_all_emission_weighted_e2, sz_E);
    } else {
        pb_p->metals_all_emission_weighted = pb_p->metals_all_emission_weighted_e2 = NULL;
    }

#undef PALLOC_INT
#undef PALLOC_DBL
}

void profile_zero(ProfileBins *pb_p, int Ndim, int nbins,
                  int n_pt, int n_met, int pos_dim)
{
    size_t sz_A = (size_t)n_pt   * (size_t)Ndim * (size_t)nbins;
    size_t sz_C = (size_t)pos_dim * sz_A;
    size_t sz_D = (size_t)Ndim   * (size_t)nbins;
    size_t sz_E = (size_t)n_met  * sz_D;

    /* Any field may be NULL now (not allocated because its output_* flag
     * was off), so guard every memset individually. */
#define ZERO_INT(field, count) \
    do { if (pb_p->field != NULL) memset(pb_p->field, 0, (count) * sizeof(int)); } while (0)
#define ZERO_DBL(field, count) \
    do { if (pb_p->field != NULL) memset(pb_p->field, 0, (count) * sizeof(double)); } while (0)

    ZERO_INT(np_bg,       sz_A);
    ZERO_INT(num_vrad_out, sz_A);

    ZERO_DBL(mass_bg,                   sz_A);
    ZERO_DBL(vrad_times_mass,           sz_A);
    ZERO_DBL(vLOS_x,                    sz_A);
    ZERO_DBL(vLOS_y,                    sz_A);
    ZERO_DBL(vLOS_z,                    sz_A);
    ZERO_DBL(mass_out,                  sz_A);
    ZERO_DBL(mass_out_fast1,            sz_A);
    ZERO_DBL(mass_out_fast2,            sz_A);
    ZERO_DBL(mass_out_fast3,            sz_A);
    ZERO_DBL(vrad_times_mass_out,       sz_A);
    ZERO_DBL(vrad_times_mass_out_fast1, sz_A);
    ZERO_DBL(vrad_times_mass_out_fast2, sz_A);
    ZERO_DBL(vrad_times_mass_out_fast3, sz_A);
    ZERO_DBL(vLOS_x_out,               sz_A);
    ZERO_DBL(vLOS_y_out,               sz_A);
    ZERO_DBL(vLOS_z_out,               sz_A);

    ZERO_DBL(vLOS,                 sz_C);
    ZERO_DBL(matter_flow_vector,   sz_C);
    ZERO_DBL(matter_outflow_vector, sz_C);
    ZERO_DBL(vLOS_out,             sz_C);

    /* Family D */
    ZERO_DBL(temp_bg,                   sz_D);
    ZERO_DBL(entropy,                   sz_D);
    ZERO_DBL(L_bol_bg,                  sz_D);
    ZERO_DBL(L_bol_times_mass_bg,       sz_D);
    ZERO_DBL(N_e,                       sz_D);
    ZERO_DBL(N_H0,                      sz_D);
    ZERO_DBL(N_H,                       sz_D);
    ZERO_DBL(temp_out,                  sz_D);
    ZERO_DBL(temp_out_fast1,            sz_D);
    ZERO_DBL(temp_out_fast2,            sz_D);
    ZERO_DBL(temp_out_fast3,            sz_D);
    ZERO_DBL(mass_bg_TempAboveLim1,     sz_D);
    ZERO_DBL(mass_bg_TempAboveLim2,     sz_D);
    ZERO_DBL(mass_bg_TempAboveLim3,     sz_D);
    ZERO_DBL(mass_bg_out_TempAboveLim1, sz_D);
    ZERO_DBL(mass_bg_out_TempAboveLim2, sz_D);
    ZERO_DBL(mass_bg_out_TempAboveLim3, sz_D);

    /* Family E */
    ZERO_DBL(metals_all,                     sz_E);
    ZERO_DBL(metals_all_emission_weighted,   sz_E);
    ZERO_DBL(metals_all_emission_weighted_e2, sz_E);
    ZERO_DBL(metals_all_out,                 sz_E);
    ZERO_DBL(metals_all_out_fast1,           sz_E);
    ZERO_DBL(metals_all_out_fast2,           sz_E);
    ZERO_DBL(metals_all_out_fast3,           sz_E);

#undef ZERO_INT
#undef ZERO_DBL
}

void profile_free(ProfileBins *pb_p)
{
    free(pb_p->np_bg);
    free(pb_p->num_vrad_out);
    free(pb_p->mass_bg);
    free(pb_p->vrad_times_mass);
    free(pb_p->vLOS_x);
    free(pb_p->vLOS_y);
    free(pb_p->vLOS_z);
    free(pb_p->mass_out);
    free(pb_p->mass_out_fast1);
    free(pb_p->mass_out_fast2);
    free(pb_p->mass_out_fast3);
    free(pb_p->vrad_times_mass_out);
    free(pb_p->vrad_times_mass_out_fast1);
    free(pb_p->vrad_times_mass_out_fast2);
    free(pb_p->vrad_times_mass_out_fast3);
    free(pb_p->vLOS_x_out);
    free(pb_p->vLOS_y_out);
    free(pb_p->vLOS_z_out);
    free(pb_p->vLOS);
    free(pb_p->matter_flow_vector);
    free(pb_p->matter_outflow_vector);
    free(pb_p->vLOS_out);
    free(pb_p->temp_bg);
    free(pb_p->entropy);
    free(pb_p->L_bol_bg);
    free(pb_p->L_bol_times_mass_bg);
    free(pb_p->N_e);
    free(pb_p->N_H0);
    free(pb_p->N_H);
    free(pb_p->temp_out);
    free(pb_p->temp_out_fast1);
    free(pb_p->temp_out_fast2);
    free(pb_p->temp_out_fast3);
    free(pb_p->mass_bg_TempAboveLim1);
    free(pb_p->mass_bg_TempAboveLim2);
    free(pb_p->mass_bg_TempAboveLim3);
    free(pb_p->mass_bg_out_TempAboveLim1);
    free(pb_p->mass_bg_out_TempAboveLim2);
    free(pb_p->mass_bg_out_TempAboveLim3);
    free(pb_p->metals_all);
    free(pb_p->metals_all_emission_weighted);
    free(pb_p->metals_all_emission_weighted_e2);
    free(pb_p->metals_all_out);
    free(pb_p->metals_all_out_fast1);
    free(pb_p->metals_all_out_fast2);
    free(pb_p->metals_all_out_fast3);
}

/* Write an N-D dataset into an open HDF5 file. */
static herr_t write_nd_dset(hid_t fid, const char *name,
                             hid_t dtype, const void *data,
                             int ndims, const hsize_t *dims)
{
    hid_t   sid = H5Screate_simple(ndims, dims, NULL);
    if (sid < 0)
        ma_h5_fatal("create dataspace for", "dataset", name);
    hid_t   did = H5Dcreate2(fid, name, dtype, sid,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (did < 0)
        ma_h5_fatal("create", "dataset", name);
    herr_t  ret = H5Dwrite(did, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    if (ret < 0)
        ma_h5_fatal("write", "dataset", name);
    if (H5Dclose(did) < 0)
        ma_h5_fatal("close", "dataset", name);
    if (H5Sclose(sid) < 0)
        ma_h5_fatal("close dataspace for", "dataset", name);
    return ret;
}

/* Transpose 3-D array: (d0, d1, d2) -> (d1, d0, d2).
 * Returns malloc'd copy; caller must free(). */
static void *trans_swap01_3d(const void *src,
                              hsize_t d0, hsize_t d1, hsize_t d2,
                              size_t esz)
{
    void *out = malloc(d0 * d1 * d2 * esz);
    if (!out) { fputs("trans_swap01_3d: OOM\n", stderr); exit(8); }
    const char *s = (const char *)src;
    char       *o = (char *)out;
    for (hsize_t i = 0; i < d0; i++)
        for (hsize_t j = 0; j < d1; j++) {
            size_t src_row = (i * d1 + j) * d2;
            size_t dst_row = (j * d0 + i) * d2;
            memcpy(o + dst_row * esz, s + src_row * esz, d2 * esz);
        }
    return out;
}

/* Transpose 4-D array: (d0, d1, d2, d3) -> (d2, d0, d1, d3).
 * Family C: (pos_dim, n_pt, n_halos, nbins) -> (n_halos, pos_dim, n_pt, nbins).
 * Returns malloc'd copy; caller must free(). */
static void *trans_C_4d(const void *src,
                         hsize_t d0, hsize_t d1, hsize_t d2, hsize_t d3,
                         size_t esz)
{
    void *out = malloc(d0 * d1 * d2 * d3 * esz);
    if (!out) { fputs("trans_C_4d: OOM\n", stderr); exit(8); }
    const char *s = (const char *)src;
    char       *o = (char *)out;
    for (hsize_t a = 0; a < d0; a++)
        for (hsize_t b = 0; b < d1; b++)
            for (hsize_t c = 0; c < d2; c++) {
                size_t src_row = ((a * d1 + b) * d2 + c) * d3;
                size_t dst_row = ((c * d0 + a) * d1 + b) * d3;
                memcpy(o + dst_row * esz, s + src_row * esz, d3 * esz);
            }
    return out;
}

/* Wrappers: transpose into a temporary buffer, write, free. */
static void write_AB(hid_t gpid, const char *name, hid_t dtype, size_t esz,
                     const void *data, int n_pt, int Ndim, int nbins,
                     const hsize_t *dims)
{
    void *tmp = trans_swap01_3d(data, (hsize_t)n_pt, (hsize_t)Ndim, (hsize_t)nbins, esz);
    write_nd_dset(gpid, name, dtype, tmp, 3, dims);
    free(tmp);
}
static void write_C(hid_t gpid, const char *name,
                    const void *data, int pos_dim, int n_pt, int Ndim, int nbins,
                    const hsize_t *dims)
{
    void *tmp = trans_C_4d(data, (hsize_t)pos_dim, (hsize_t)n_pt, (hsize_t)Ndim, (hsize_t)nbins, sizeof(double));
    write_nd_dset(gpid, name, H5T_NATIVE_DOUBLE, tmp, 4, dims);
    free(tmp);
}
static void write_E(hid_t gpid, const char *name,
                    const void *data, int n_met, int Ndim, int nbins,
                    const hsize_t *dims)
{
    void *tmp = trans_swap01_3d(data, (hsize_t)n_met, (hsize_t)Ndim, (hsize_t)nbins, sizeof(double));
    write_nd_dset(gpid, name, H5T_NATIVE_DOUBLE, tmp, 3, dims);
    free(tmp);
}

/* Write a scalar int attribute on an open HDF5 object. */
static herr_t write_int_attr(hid_t oid, const char *name, int value)
{
    hid_t   sid = H5Screate(H5S_SCALAR);
    if (sid < 0)
        ma_h5_fatal("create dataspace for", "attribute", name);
    hid_t   aid = H5Acreate2(oid, name, H5T_NATIVE_INT, sid,
                              H5P_DEFAULT, H5P_DEFAULT);
    if (aid < 0)
        ma_h5_fatal("create", "attribute", name);
    herr_t  ret = H5Awrite(aid, H5T_NATIVE_INT, &value);
    if (ret < 0)
        ma_h5_fatal("write", "attribute", name);
    if (H5Aclose(aid) < 0)
        ma_h5_fatal("close", "attribute", name);
    if (H5Sclose(sid) < 0)
        ma_h5_fatal("close dataspace for", "attribute", name);
    return ret;
}

/* Write a scalar double attribute on an open HDF5 object. */
static herr_t write_double_attr(hid_t oid, const char *name, double value)
{
    hid_t   sid = H5Screate(H5S_SCALAR);
    if (sid < 0)
        ma_h5_fatal("create dataspace for", "attribute", name);
    hid_t   aid = H5Acreate2(oid, name, H5T_NATIVE_DOUBLE, sid,
                              H5P_DEFAULT, H5P_DEFAULT);
    if (aid < 0)
        ma_h5_fatal("create", "attribute", name);
    herr_t  ret = H5Awrite(aid, H5T_NATIVE_DOUBLE, &value);
    if (ret < 0)
        ma_h5_fatal("write", "attribute", name);
    if (H5Aclose(aid) < 0)
        ma_h5_fatal("close", "attribute", name);
    if (H5Sclose(sid) < 0)
        ma_h5_fatal("close dataspace for", "attribute", name);
    return ret;
}

void profile_save_hdf5(ProfileBins *pb_p, const char *fpath,
                        int Ndim, int nbins, int n_pt, int n_met, int pos_dim,
                        const float *r_smoothing)
{
    hid_t fid = H5Fcreate(fpath, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0) {
        ma_h5_fatal("create", "file", fpath);
    }

    /* Dimension attributes on the file root so readers can verify shapes. */
    write_int_attr(fid, "Ndim",           Ndim);
    write_int_attr(fid, "nbins_profile",  nbins);
    write_int_attr(fid, "n_partType",     n_pt);
    write_int_attr(fid, "n_metals",       n_met);
    write_int_attr(fid, "pos_dimension",  pos_dim);

    /* /Header group — mirrors the structure written by the Python converter. */
    hid_t hgid = H5Gcreate2(fid, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (hgid < 0)
        ma_h5_fatal("create", "group", "Header");
    write_int_attr   (hgid, "Ngroups_ThisFile",       Ndim);
    write_int_attr   (hgid, "NumFilesPerSnapshot",    n_subfile);
    write_int_attr   (hgid, "is_hydro_sim",           is_hydro_sim);
    write_double_attr(hgid, "BoxSize",                box_size);
    write_double_attr(hgid, "Redshift",               sim_redshift);
    write_double_attr(hgid, "Group_M_Crit200_minlim", gr_M200_minlim);
    if (H5Gclose(hgid) < 0)
        ma_h5_fatal("close", "group", "Header");

    /* /GroupProfile group — one dataset per profile array, plus Radius. */
    hid_t gpid = H5Gcreate2(fid, "GroupProfile", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (gpid < 0)
        ma_h5_fatal("create", "group", "GroupProfile");

    /* Shape descriptors — leading axis is always n_halos (=Ndim here). */
    hsize_t dims_AB[3] = { (hsize_t)Ndim,    (hsize_t)n_pt,    (hsize_t)nbins };
    hsize_t dims_C[4]  = { (hsize_t)Ndim,    (hsize_t)pos_dim, (hsize_t)n_pt, (hsize_t)nbins };
    hsize_t dims_D[2]  = { (hsize_t)Ndim,    (hsize_t)nbins };
    hsize_t dims_E[3]  = { (hsize_t)Ndim,    (hsize_t)n_met,   (hsize_t)nbins };

    /* Radius: shape (Ndim, nbins_profile).
     * Formula: R[g][i] = r_smoothing[g] * (nbins - i) * nRvir_profile / nbins
     * Matches: R_profile[:, i] = groupR200crit * (nbins_profile - i) * bin_size
     *          where bin_size = nR200 / nbins_profile. */
    {
        double *radius = malloc((size_t)Ndim * (size_t)nbins * sizeof(double));
        if (!radius) { fprintf(stderr, "profile_save_hdf5: OOM for Radius\n"); exit(8); }
        for (int g = 0; g < Ndim; g++)
            for (int i = 0; i < nbins; i++)
                radius[g * nbins + i] = (double)r_smoothing[g]
                                        * (nbins - i)
                                        * (double)nRvir_profile / nbins;
        write_nd_dset(gpid, "Radius", H5T_NATIVE_DOUBLE, radius, 2, dims_D);
        free(radius);
    }

    /* Family A (int, n_halos × n_partType × nbins_profile) */
    if (pb_p->np_bg != NULL)
        write_AB(gpid, "np",     H5T_NATIVE_INT,    sizeof(int),    pb_p->np_bg,        n_pt, Ndim, nbins, dims_AB);
    if (pb_p->num_vrad_out != NULL)
        write_AB(gpid, "np_out", H5T_NATIVE_INT,    sizeof(int),    pb_p->num_vrad_out, n_pt, Ndim, nbins, dims_AB);
    /* Family B (double, n_halos × n_partType × nbins_profile) */
    if (pb_p->mass_bg != NULL)
        write_AB(gpid, "mass",                      H5T_NATIVE_DOUBLE, sizeof(double), pb_p->mass_bg,                   n_pt, Ndim, nbins, dims_AB);
    if (pb_p->vrad_times_mass != NULL)
        write_AB(gpid, "vrad_times_mass",            H5T_NATIVE_DOUBLE, sizeof(double), pb_p->vrad_times_mass,           n_pt, Ndim, nbins, dims_AB);
    if (pb_p->vLOS_x != NULL) {
        write_AB(gpid, "vLOS_x",                    H5T_NATIVE_DOUBLE, sizeof(double), pb_p->vLOS_x,                    n_pt, Ndim, nbins, dims_AB);
        write_AB(gpid, "vLOS_y",                    H5T_NATIVE_DOUBLE, sizeof(double), pb_p->vLOS_y,                    n_pt, Ndim, nbins, dims_AB);
        write_AB(gpid, "vLOS_z_times_mass",          H5T_NATIVE_DOUBLE, sizeof(double), pb_p->vLOS_z,                    n_pt, Ndim, nbins, dims_AB);
    }
    if (pb_p->mass_out != NULL) {
        write_AB(gpid, "mass_out",                  H5T_NATIVE_DOUBLE, sizeof(double), pb_p->mass_out,                  n_pt, Ndim, nbins, dims_AB);
        write_AB(gpid, "vrad_times_mass_out",        H5T_NATIVE_DOUBLE, sizeof(double), pb_p->vrad_times_mass_out,       n_pt, Ndim, nbins, dims_AB);
        write_AB(gpid, "vLOS_x_out",                H5T_NATIVE_DOUBLE, sizeof(double), pb_p->vLOS_x_out,               n_pt, Ndim, nbins, dims_AB);
        write_AB(gpid, "vLOS_y_out",                H5T_NATIVE_DOUBLE, sizeof(double), pb_p->vLOS_y_out,               n_pt, Ndim, nbins, dims_AB);
        write_AB(gpid, "vLOS_z_times_mass_out",      H5T_NATIVE_DOUBLE, sizeof(double), pb_p->vLOS_z_out,               n_pt, Ndim, nbins, dims_AB);
    }
    if (pb_p->mass_out_fast1 != NULL) {
        write_AB(gpid, "mass_out_fast1",             H5T_NATIVE_DOUBLE, sizeof(double), pb_p->mass_out_fast1,            n_pt, Ndim, nbins, dims_AB);
        write_AB(gpid, "mass_out_fast2",             H5T_NATIVE_DOUBLE, sizeof(double), pb_p->mass_out_fast2,            n_pt, Ndim, nbins, dims_AB);
        write_AB(gpid, "mass_out_fast3",             H5T_NATIVE_DOUBLE, sizeof(double), pb_p->mass_out_fast3,            n_pt, Ndim, nbins, dims_AB);
        write_AB(gpid, "vrad_times_mass_out_fast1",  H5T_NATIVE_DOUBLE, sizeof(double), pb_p->vrad_times_mass_out_fast1, n_pt, Ndim, nbins, dims_AB);
        write_AB(gpid, "vrad_times_mass_out_fast2",  H5T_NATIVE_DOUBLE, sizeof(double), pb_p->vrad_times_mass_out_fast2, n_pt, Ndim, nbins, dims_AB);
        write_AB(gpid, "vrad_times_mass_out_fast3",  H5T_NATIVE_DOUBLE, sizeof(double), pb_p->vrad_times_mass_out_fast3, n_pt, Ndim, nbins, dims_AB);
    }
    /* Family C (double, n_halos × pos_dimension × n_partType × nbins_profile) */
    if (pb_p->vLOS != NULL) {
        write_C(gpid, "vLOS_times_mass",      pb_p->vLOS,                  pos_dim, n_pt, Ndim, nbins, dims_C);
        write_C(gpid, "matter_flow_vec",       pb_p->matter_flow_vector,    pos_dim, n_pt, Ndim, nbins, dims_C);
    }
    if (pb_p->matter_outflow_vector != NULL) {
        write_C(gpid, "matter_outflow_vec",    pb_p->matter_outflow_vector, pos_dim, n_pt, Ndim, nbins, dims_C);
        write_C(gpid, "vLOS_out",              pb_p->vLOS_out,              pos_dim, n_pt, Ndim, nbins, dims_C);
    }
    /* Family D (double, Ndim × nbins_profile) — hydro only */
    if (pb_p->temp_bg != NULL) {
        write_nd_dset(gpid, "temp",                      H5T_NATIVE_DOUBLE, pb_p->temp_bg,                   2, dims_D);
        write_nd_dset(gpid, "entropy",                   H5T_NATIVE_DOUBLE, pb_p->entropy,                   2, dims_D);
        write_nd_dset(gpid, "temp_out",                  H5T_NATIVE_DOUBLE, pb_p->temp_out,                  2, dims_D);
        write_nd_dset(gpid, "temp_out_fast1",             H5T_NATIVE_DOUBLE, pb_p->temp_out_fast1,            2, dims_D);
        write_nd_dset(gpid, "temp_out_fast2",             H5T_NATIVE_DOUBLE, pb_p->temp_out_fast2,            2, dims_D);
        write_nd_dset(gpid, "temp_out_fast3",             H5T_NATIVE_DOUBLE, pb_p->temp_out_fast3,            2, dims_D);
        write_nd_dset(gpid, "mass_temp1",                H5T_NATIVE_DOUBLE, pb_p->mass_bg_TempAboveLim1,     2, dims_D);
        write_nd_dset(gpid, "mass_temp2",                H5T_NATIVE_DOUBLE, pb_p->mass_bg_TempAboveLim2,     2, dims_D);
        write_nd_dset(gpid, "mass_temp3",                H5T_NATIVE_DOUBLE, pb_p->mass_bg_TempAboveLim3,     2, dims_D);
        write_nd_dset(gpid, "mass_out_temp1",            H5T_NATIVE_DOUBLE, pb_p->mass_bg_out_TempAboveLim1, 2, dims_D);
        write_nd_dset(gpid, "mass_out_temp2",            H5T_NATIVE_DOUBLE, pb_p->mass_bg_out_TempAboveLim2, 2, dims_D);
        write_nd_dset(gpid, "mass_out_temp3",            H5T_NATIVE_DOUBLE, pb_p->mass_bg_out_TempAboveLim3, 2, dims_D);
    }
    if (pb_p->L_bol_bg != NULL) {
        write_nd_dset(gpid, "L_bol",                     H5T_NATIVE_DOUBLE, pb_p->L_bol_bg,                  2, dims_D);
        write_nd_dset(gpid, "L_bol_times_mass",           H5T_NATIVE_DOUBLE, pb_p->L_bol_times_mass_bg,       2, dims_D);
    }
    if (pb_p->N_e != NULL) {
        write_nd_dset(gpid, "N_e",                       H5T_NATIVE_DOUBLE, pb_p->N_e,                       2, dims_D);
        write_nd_dset(gpid, "N_H0",                      H5T_NATIVE_DOUBLE, pb_p->N_H0,                      2, dims_D);
        write_nd_dset(gpid, "N_H",                       H5T_NATIVE_DOUBLE, pb_p->N_H,                       2, dims_D);
    }
    /* Family E (double, n_halos × n_metals × nbins_profile) — hydro only */
    if (pb_p->metals_all != NULL) {
        write_E(gpid, "metals_all",                      pb_p->metals_all,                      n_met, Ndim, nbins, dims_E);
        write_E(gpid, "metals_all_out",                  pb_p->metals_all_out,                  n_met, Ndim, nbins, dims_E);
        write_E(gpid, "metals_all_out_fast1",            pb_p->metals_all_out_fast1,            n_met, Ndim, nbins, dims_E);
        write_E(gpid, "metals_all_out_fast2",            pb_p->metals_all_out_fast2,            n_met, Ndim, nbins, dims_E);
        write_E(gpid, "metals_all_out_fast3",            pb_p->metals_all_out_fast3,            n_met, Ndim, nbins, dims_E);
    }
    if (pb_p->metals_all_emission_weighted != NULL) {
        write_E(gpid, "metals_all_emission_weighted",    pb_p->metals_all_emission_weighted,    n_met, Ndim, nbins, dims_E);
        write_E(gpid, "metals_all_emission_weighted_e2", pb_p->metals_all_emission_weighted_e2, n_met, Ndim, nbins, dims_E);
    }

    if (H5Gclose(gpid) < 0)
        ma_h5_fatal("close", "group", "GroupProfile");
    if (H5Fclose(fid) < 0)
        ma_h5_fatal("close", "file", fpath);
}
