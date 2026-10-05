/* PROFI computes radial profiles for halos in cosmological simulations. */

#include <stdio.h>
#include <limits.h>
#include <stdint.h>
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

static void input_h5_fatal(const char *operation, const char *object_name)
{
	fprintf(stderr, "HDF5 input error: failed to %s '%s'\n", operation, object_name);
	H5Eprint2(H5E_DEFAULT, stderr);
	fflush(NULL);
	_Exit(1);
}

static hid_t open_hdf5_input(const char *path)
{
	hid_t fid = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (fid < 0)
		input_h5_fatal("open file", path);
	return fid;
}

static herr_t close_hdf5_input(hid_t fid, const char *path)
{
	herr_t status = H5Fclose(fid);
	if (status < 0)
		input_h5_fatal("close file", path);
	return status;
}

static htri_t input_h5_exists(hid_t loc_id, const char *object_name)
{
	htri_t exists = H5Lexists(loc_id, object_name, H5P_DEFAULT);
	if (exists < 0)
		input_h5_fatal("query object", object_name);
	return exists;
}

static void get_mem_gb(double *rss_gb, double *hwm_gb)
{
	FILE *f = fopen("/proc/self/status", "r");
	*rss_gb = *hwm_gb = -1.0;
	if (!f) return;
	char line[128];
	long rss = -1, hwm = -1;
	while (fgets(line, sizeof(line), f)) {
		if (rss < 0) sscanf(line, "VmRSS: %ld kB", &rss);
		if (hwm < 0) sscanf(line, "VmHWM: %ld kB", &hwm);
		if (rss >= 0 && hwm >= 0) break;
	}
	fclose(f);
	*rss_gb = rss / (1024.0 * 1024.0);
	*hwm_gb = hwm / (1024.0 * 1024.0);
}

static void print_usage_and_exit(const char *program)
{
	fprintf(stderr,
		"\nusage: %s <parameterfile> <snapnum_i> <snapnum_f> <subfilenr_i> <subfilenr_f>\n"
		"       [<part_in_profile_method>]   (default 0)\n\n"
		"All simulation-specific settings (SimulationLabel, cosmology, n_partType, ...)\n"
		"are read from the parameter file.\n\n",
		program);
	exit(1);
}


int main(int argc, char **argv)
{

	setvbuf(stdout, NULL, _IONBF, 0);

	time_t start = time(NULL);
	clock_t begin = clock();

	dm_partType = 1; // Standard AREPO particle type for dark matter.
	pos_dimension = 3;
	gal_dimension = 3;

	// Gas temperature constants
	m_p = 1.6726219e-24; // grams
	k_B = 1.38064852e-16; // Boltzmann constant in erg/K (CGS)
	gamma_gas = 5.0/3;
	X_H = 0.76; // Hydrogen mass fraction

	char buf[1000];
	FILE *fd;


	if(argc < 6 || argc > 7)
    	{
		print_usage_and_exit(argv[0]);
    	}


	read_parameter_file(argv[1]);
	/* Apply cosmology before converting mass limits to simulation units. */
	h_const       = cosmo_h;
	Omega_m       = cosmo_Omega_m;
	Omega_b       = cosmo_Omega_b;
	Omega_Lambda  = cosmo_Omega_Lambda;
	Omega_dm      = Omega_m - Omega_b;
	Omega_r       = 0;
	Omega_k       = 0;

	gr_M200_minlim = gr_M200_minlim * h_const; // converting to 1e10 Msun/h
	gr_M200_maxlim = gr_M200_maxlim * h_const; // converting to 1e10 Msun/h
	printf("1) parameter file has been read\n");

	/* Disable outputs whose dependencies are disabled. */
	if (!output_np_mass || !output_vLOS) {
		if (output_flow_vectors) {
			printf("WARNING: output_flow_vectors requires output_np_mass && output_vLOS, forcing off\n");
			output_flow_vectors = 0;
		}
	}
	if (!output_outflow) {
		if (output_outflow_fast) {
			printf("WARNING: output_outflow_fast requires output_outflow, forcing off\n");
			output_outflow_fast = 0;
		}
		if (output_outflow_vectors) {
			printf("WARNING: output_outflow_vectors requires output_outflow, forcing off\n");
			output_outflow_vectors = 0;
		}
	}
	if (!is_hydro_sim || !calculate_all_gas_properties) {
		if (output_temperature) {
			printf("WARNING: output_temperature requires hydro sim + calculate_all_gas_properties, forcing off\n");
			output_temperature = 0;
		}
		if (output_xray) {
			printf("WARNING: output_xray requires hydro sim + calculate_all_gas_properties, forcing off\n");
			output_xray = 0;
		}
		if (output_ionization) {
			printf("WARNING: output_ionization requires hydro sim + calculate_all_gas_properties, forcing off\n");
			output_ionization = 0;
		}
		if (output_metals) {
			printf("WARNING: output_metals requires hydro sim + calculate_all_gas_properties, forcing off\n");
			output_metals = 0;
		}
	}
	if (!output_metals || !output_xray) {
		if (output_metals_xray_weighted) {
			printf("WARNING: output_metals_xray_weighted requires output_metals && output_xray, forcing off\n");
			output_metals_xray_weighted = 0;
		}
	}

	if (!is_hydro_sim)
	{
		n_partType = 1; // DMO counterpart: DM only
	}

	SnapNum_i = atoi(argv[2]);
	SnapNum_f = atoi(argv[3]);
	subfilenr_i = atoi(argv[4]);
	subfilenr_f = atoi(argv[5]);

	if (argc == 7)
		part_in_profile_method = atoi(argv[6]);

	n_partType_all = 6;  /* 0=gas, 1=DM, 2,3, 4=stars, 5=BH */

	printf("\n\n********************\n Running PROFI\n for the %s simulation\n snapshots %d-%d\n subfilenr %d-%d\n part_in_profile_method = %d\n n_partType = %d\n\n",
	       SimulationLabel, SnapNum_i, SnapNum_f, subfilenr_i, subfilenr_f, part_in_profile_method, n_partType);

	for (SnapNum=SnapNum_i;SnapNum<SnapNum_f+1;SnapNum++)
	{
		snprintf(buf, sizeof(buf), "%s/output/snapdir_%03d/snap_%03d.%d.hdf5", SimulationDir, SnapNum, SnapNum, 0);
		if (access(buf, F_OK) != 0)
			snprintf(buf, sizeof(buf), "%s/output/snap_%03d.hdf5", SimulationDir, SnapNum);
		h5_file = open_hdf5_input(buf);
		h5_header_group = ma_H5Gopen(h5_file, "/Header");
		h5_attr_dummy = ma_H5Aopen(h5_header_group, "BoxSize");
		ma_H5Aread(&box_size, h5_attr_dummy, H5T_NATIVE_DOUBLE, sizeof box_size);
		ma_H5Aclose(h5_attr_dummy);
		printf("box size = %f\n",box_size);

		particle_mass_table = malloc(n_partType_all*sizeof(double));
		h5_attr_dummy = ma_H5Aopen(h5_header_group, "MassTable");
		ma_H5Aread(particle_mass_table, h5_attr_dummy, H5T_NATIVE_DOUBLE,
		           (size_t)n_partType_all * sizeof *particle_mass_table);
		ma_H5Aclose(h5_attr_dummy);
		dm_mass = particle_mass_table[dm_partType]; // dark matter is assigned as particle type 1
		free(particle_mass_table);
		if (dm_mass == 0.0) {
			/* MassTable[1]=0 convention (GIZMO/SIMBA): masses stored per-particle.
			   Read the first element to get the uniform DM particle mass. */
			hid_t ds_tmp = ma_H5Dopen2(h5_file, "/PartType1/Masses");
			hid_t sp_tmp = H5Dget_space(ds_tmp);
			if (sp_tmp < 0)
				input_h5_fatal("query dataspace for", "/PartType1/Masses");
			hsize_t dims_tmp[1];
			if (H5Sget_simple_extent_ndims(sp_tmp) != 1
			    || H5Sget_simple_extent_dims(sp_tmp, dims_tmp, NULL) < 0
			    || dims_tmp[0] == 0)
				input_h5_fatal("find a DM mass in", "/PartType1/Masses");
			hsize_t start_tmp[1] = {0}, count_tmp[1] = {1};
			if (H5Sselect_hyperslab(sp_tmp, H5S_SELECT_SET, start_tmp, NULL, count_tmp, NULL) < 0)
				input_h5_fatal("select the first DM mass in", "/PartType1/Masses");
			hid_t ms_tmp = H5Screate_simple(1, count_tmp, NULL);
			if (ms_tmp < 0)
				input_h5_fatal("create memory dataspace for", "/PartType1/Masses");
			float m1 = 0.0f;
			if (H5Dread(ds_tmp, H5T_NATIVE_FLOAT, ms_tmp, sp_tmp, H5P_DEFAULT, &m1) < 0)
				input_h5_fatal("read the first DM mass in", "/PartType1/Masses");
			if (H5Sclose(ms_tmp) < 0 || H5Sclose(sp_tmp) < 0)
				input_h5_fatal("close dataspace for", "/PartType1/Masses");
			ma_H5Dclose(ds_tmp);
			dm_mass = (double)m1;
		}
		printf("dm_mass = %f\n",dm_mass);

		h5_attr_dummy = ma_H5Aopen(h5_header_group, "NumFilesPerSnapshot");
		ma_H5Aread(&nfiles_par, h5_attr_dummy, H5T_NATIVE_INT, sizeof nfiles_par);
		ma_H5Aclose(h5_attr_dummy);
		if (nfiles_par < 1)
			input_h5_fatal("read a positive NumFilesPerSnapshot from", buf);
		if (cluster_subfile_mode)
		{
			n_subfile = nfiles_par/2;
			particle_for_loop_step = n_subfile;
		}
		else
		{
			n_subfile = nfiles_par;
			particle_for_loop_step = 1;
		}
		printf("n_subfile = %d\n",n_subfile);
		printf("nfiles_par = %d\n",nfiles_par);
		h5_attr_dummy = ma_H5Aopen(h5_header_group, "Redshift");
		ma_H5Aread(&sim_redshift, h5_attr_dummy, H5T_NATIVE_DOUBLE, sizeof sim_redshift);
		ma_H5Aclose(h5_attr_dummy);
		printf("redshift = %f \n********************\n\n\n\n\n\n", sim_redshift);
		sf = 1/(1+sim_redshift);
		sfsqrt = sqrt(sf);
		ma_H5Gclose(h5_header_group);
		h5_status = close_hdf5_input(h5_file, buf);


		if (part_in_profile_method==3)
		{
		// First file and particle index containing particles not bound to a halo.
		first_free_part_filenr_this_snap = malloc(n_partType_all * sizeof(int));
		first_free_part_idx_in_file_this_snap = malloc(n_partType_all * sizeof(long long));



		char buf[1000], buf2[1000];
		int k_partType;

		/* Read the indices of unbound particles */
		snprintf(buf, sizeof(buf), "%s/%s/Unbound_particles_indices/unbound_idx.hdf5",
		        InputDir, SimulationLabel);
		h5_file = open_hdf5_input(buf);
		snprintf(buf2, sizeof(buf2), "/PartIdx");
		h5_group = ma_H5Gopen(h5_file, buf2);

		snprintf(buf2, sizeof(buf2), "/PartIdx/last_file_with_group_particles");
		h5_dataset = ma_H5Dopen2(h5_group, buf2);
		int first_free_part_filenr[n_snaps][n_partType_all];
		h5_status = ma_H5Dread(first_free_part_filenr, h5_dataset, H5T_NATIVE_INT,
		                       sizeof first_free_part_filenr);
		h5_status = ma_H5Dclose(h5_dataset);

		for (k_partType=0 ; k_partType<n_partType_all; k_partType++)
		{
			first_free_part_filenr_this_snap[k_partType] = first_free_part_filenr[SnapNum][k_partType];
			printf("first_free_part_filenr_this_snap[%d] = %d\n", k_partType, first_free_part_filenr_this_snap[k_partType]);
		}

		snprintf(buf2, sizeof(buf2), "/PartIdx/part_idx_in_last_file");
		h5_dataset = ma_H5Dopen2(h5_group, buf2);
		long long first_free_part_idx_in_file[n_snaps][n_partType_all];
		h5_status = ma_H5Dread(first_free_part_idx_in_file, h5_dataset, H5T_NATIVE_LLONG,
		                       sizeof first_free_part_idx_in_file);
		h5_status = ma_H5Dclose(h5_dataset);

		for (k_partType=0 ; k_partType<n_partType_all; k_partType++)
		{
			first_free_part_idx_in_file_this_snap[k_partType] = first_free_part_idx_in_file[SnapNum][k_partType];
			printf("first_free_part_idx_in_file_this_snap[%d] = %lld\n", k_partType, first_free_part_idx_in_file_this_snap[k_partType]);
		}

		ma_H5Gclose(h5_group);
		h5_status = close_hdf5_input(h5_file, buf);
		printf("done reading unbound particles' indices\n");
		}

		// Read the status of files (if they contain subhalos or they are empty)
		fileStat_sim = calloc(n_subfile, sizeof(int));
		snprintf(buf, sizeof(buf), "%s/%s/files_with_subhalo_status/file_status_snap%03d.dat",
		        InputDir, SimulationLabel, SnapNum);
		if(!(fd = fopen(buf, "r")))
		{
			printf("can't open file `%s'\n", buf);
			printf("setting all fileStat_sim values to 1 and continuing\n");
			for (subfilenr = 0; subfilenr < n_subfile; subfilenr++)
			{
				fileStat_sim[subfilenr] = 1;
			}
		}
		else
		{
			printf("reading file `%s'\n", buf);
			if (fread(fileStat_sim, sizeof(int), n_subfile, fd) != (size_t)n_subfile) {
				fprintf(stderr, "ERROR: cannot read all %d file-status entries from '%s'\n", n_subfile, buf);
				exit(1);
			}
			if (fclose(fd) != 0) {
				perror(buf);
				exit(1);
			}
		}

		// Clamp subfile range to actual n_subfile
		if (subfilenr_f >= n_subfile) {
			printf("WARNING: subfilenr_f=%d >= n_subfile=%d, clamping\n", subfilenr_f, n_subfile);
			subfilenr_f = n_subfile - 1;
		}
		for (subfilenr=subfilenr_i;subfilenr<subfilenr_f+1;subfilenr++)
		{
			if (fileStat_sim[subfilenr]==1)
			{
				printf("filestat = %d: running the code for snapnum=%d and subfilenr=%d\n\n", fileStat_sim[subfilenr],SnapNum,subfilenr);
				read_in_particle_data();
			}
			else
			{
				printf("snapnum=%d and subfilenr=%d is empty\n\n",SnapNum,subfilenr);
				printf("fileStat_sim = %d",fileStat_sim[subfilenr]);
			}
		}
		free(fileStat_sim);
		if (part_in_profile_method == 3) {
			free(first_free_part_filenr_this_snap);
			free(first_free_part_idx_in_file_this_snap);
		}
	}

	printf("\n Real execution time = %.2f\n", (double)(time(NULL) - start));
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("CPU execution time = %f\n",time_spent);
	return 0;
}

void read_in_particle_data(void)
{
	int filenr, i_partType, j_partType;
	int i, n;
	long long i1, j1, k1;
	char buf[1000], buf2[1000];

	n_galpar_counter = 0;

	/* A requested non-empty subfile requires its FoF catalogue. */
	snprintf(buf, sizeof(buf), "%s/output/groups_%03d/fof_subhalo_tab_%03d.%d.hdf5", SimulationDir, SnapNum, SnapNum, subfilenr);
	if (access(buf, F_OK) != 0) {
		snprintf(buf, sizeof(buf), "%s/output/fof_subhalo_tab_%03d.hdf5", SimulationDir, SnapNum);
		if (access(buf, F_OK) != 0) {
			fprintf(stderr, "ERROR: snapshot %d subfile %d: required FoF catalogue not found\n", SnapNum, subfilenr);
			exit(1);
		}
	}
	h5_file = open_hdf5_input(buf);

	h5_dataset = ma_H5Dopen2(h5_file, "/Group/GroupPos");
	h5_space = H5Dget_space(h5_dataset);
	if (h5_space < 0)
		input_h5_fatal("query dataspace for", "/Group/GroupPos");
	int ndims = H5Sget_simple_extent_ndims(h5_space);
	if (ndims != 2)
		input_h5_fatal("read a rank-2 dataset from", "/Group/GroupPos");
	hsize_t dims[2];
	if (H5Sget_simple_extent_dims(h5_space, dims, NULL) < 0
	    || dims[1] != 3 || dims[0] > INT_MAX)
		input_h5_fatal("read supported position dimensions from", "/Group/GroupPos");
	if (H5Sclose(h5_space) < 0)
		input_h5_fatal("close dataspace for", "/Group/GroupPos");
	int Ndim_total = dims[0];
	float (*r_g)[3] = malloc(Ndim_total * sizeof(*r_g));
	/* HDF5 converts dataset values to match the float32 destination. */
	ma_H5Dread(r_g, h5_dataset, H5T_NATIVE_FLOAT, (size_t)Ndim_total * sizeof *r_g);
	h5_status = ma_H5Dclose(h5_dataset);

	h5_dataset = ma_H5Dopen2(h5_file, "/Group/GroupVel");
	float (*v_g)[3] = malloc(Ndim_total * sizeof(*v_g));
	ma_H5Dread(v_g, h5_dataset, H5T_NATIVE_FLOAT, (size_t)Ndim_total * sizeof *v_g);
	h5_status = ma_H5Dclose(h5_dataset);

	h5_dataset = ma_H5Dopen2(h5_file, "/Group/Group_R_Crit200");
	float *r_smoothing = malloc(Ndim_total * sizeof(float));
	/* Request float32 so HDF5 converts values stored at higher precision. */
	ma_H5Dread(r_smoothing, h5_dataset, H5T_NATIVE_FLOAT,
	           (size_t)Ndim_total * sizeof *r_smoothing);
	h5_status = ma_H5Dclose(h5_dataset);

	h5_dataset = ma_H5Dopen2(h5_file, "/Group/Group_M_Crit200");
	float *gr_M200 = malloc(Ndim_total * sizeof(float));
	ma_H5Dread(gr_M200, h5_dataset, H5T_NATIVE_FLOAT, (size_t)Ndim_total * sizeof *gr_M200);
	h5_status = ma_H5Dclose(h5_dataset);
	h5_status = close_hdf5_input(h5_file, buf);

	/* Count halos above the mass threshold and compact FoF arrays in-place. */
	int Ndim = 0;
	for (int g = 0; g < Ndim_total; g++) {
		if (gr_M200[g] > gr_M200_minlim && gr_M200[g] < gr_M200_maxlim) {
			if (Ndim != g) {
				r_g[Ndim][0] = r_g[g][0]; r_g[Ndim][1] = r_g[g][1]; r_g[Ndim][2] = r_g[g][2];
				v_g[Ndim][0] = v_g[g][0]; v_g[Ndim][1] = v_g[g][1]; v_g[Ndim][2] = v_g[g][2];
				r_smoothing[Ndim] = r_smoothing[g];
				gr_M200[Ndim] = gr_M200[g];
			}
			Ndim++;
		}
	}
	printf("Ndim (total FoF groups) = %d, above mass limit = %d\n", Ndim_total, Ndim);

	Hubble_Parameter = h_const*100 * sqrt(Omega_m * pow(sf,-3) + Omega_Lambda); // Hubble parameter at redshift z

	np_bg_limit = malloc(Ndim * sizeof(int));

	profile_alloc(&pb, Ndim, nbins_profile, n_partType, n_metals, pos_dimension);
	profile_zero(&pb, Ndim, nbins_profile, n_partType, n_metals, pos_dimension);


	for(i = 0; i < Ndim; i++)
	{
		np_bg_limit[i] = 0;
	}

	gal_bin_x = malloc(Ndim * sizeof(int));
	gal_bin_y = malloc(Ndim * sizeof(int));
	gal_bin_z = malloc(Ndim * sizeof(int));




	// Bin halo centers for the spatial particle lookup.
	n_par_in_gal = malloc(Ndim * sizeof(int));
	for(i = 0; i<Ndim; i++)
	{
		n_par_in_gal[i] = 0;
	}
	int numgal;


	if (projection_flag == 1)
	{
		for (numgal = 0; numgal<Ndim; numgal++)
		{
			r_g[numgal][2] = 0; // This ensures that in the case of projection in the z-direction, all galaxies are projected in the same plane 
		}
	}

	for (numgal = 0; numgal<Ndim; numgal++)
	{
		gal_bin_x[numgal] = floor(r_g[numgal][0]/l_bin);
		gal_bin_y[numgal] = floor(r_g[numgal][1]/l_bin);
		gal_bin_z[numgal] = floor(r_g[numgal][2]/l_bin);
	}




	printf("Number of FOF halos = %d\n",Ndim);

	for(i_partType = 0; i_partType < n_partType; i_partType++)
	{
		if (is_hydro_sim == 0)
		{
			j_partType = 1; /* DMO: the only loop iteration maps to DM (type 1) */
		}
		else if (i_partType==0)
		{
			j_partType = 0;
		}
		else if (i_partType==1)
		{
			j_partType = 1;
		}
		else if (i_partType==2)
		{
			j_partType = 4;
		}
		else if (i_partType==3)
		{
			j_partType = 5;
		}
		else if (i_partType==4)
		{
			j_partType = 2;
		}

		if (part_in_profile_method==3)
		{
			filenr_i = first_free_part_filenr_this_snap[j_partType];
		}
		else
		{
			filenr_i = 0;
		}

		for(filenr = filenr_i; filenr < nfiles_par; filenr+=particle_for_loop_step)
		{

			snprintf(buf, sizeof(buf), "%s/output/snapdir_%03d/snap_%03d.%d.hdf5", SimulationDir, SnapNum, SnapNum, filenr);
			if (access(buf, F_OK) != 0) {
				if (nfiles_par != 1) {
					fprintf(stderr, "ERROR: snapshot %d subfile %d particle file %d: required snapshot chunk '%s' not found\n", SnapNum, subfilenr, filenr, buf);
					exit(1);
				}
				snprintf(buf, sizeof(buf), "%s/output/snap_%03d.hdf5", SimulationDir, SnapNum);
				if (access(buf, F_OK) != 0) {
					fprintf(stderr, "ERROR: snapshot %d subfile %d particle file %d: required snapshot '%s' not found\n", SnapNum, subfilenr, filenr, buf);
					exit(1);
				}
			}
			{ double _rss, _hwm; get_mem_gb(&_rss, &_hwm);
			printf("Snap = %d, subF = %d, parF = %d, ParType = %d -- [mem: %.2f GB RSS, %.2f GB peak]\n",
				SnapNum, subfilenr, filenr, j_partType, _rss, _hwm); }
			h5_file = open_hdf5_input(buf);

			snprintf(buf2, sizeof(buf2), "/PartType%d", j_partType);
			if(input_h5_exists(h5_file, buf2))
			{
				h5_group = ma_H5Gopen(h5_file, buf2);

				snprintf(buf2, sizeof(buf2), "/PartType%d/Coordinates", j_partType);
				if(input_h5_exists(h5_group, buf2))
				{
					h5_dataset = ma_H5Dopen2(h5_group, buf2);
					h5_space = H5Dget_space(h5_dataset);
					if (h5_space < 0)
						input_h5_fatal("query dataspace for", buf2);
					const int ndims_particles = H5Sget_simple_extent_ndims(h5_space);
					if (ndims_particles != 2)
						input_h5_fatal("read a rank-2 dataset from", buf2);
					hsize_t dims_particles[2];
					if (H5Sget_simple_extent_dims(h5_space, dims_particles, NULL) < 0
					    || dims_particles[1] != 3 || dims_particles[0] > INT_MAX)
						input_h5_fatal("read supported position dimensions from", buf2);
					if (H5Sclose(h5_space) < 0)
						input_h5_fatal("close dataspace for", buf2);



					/* Restrict mode 3 to unbound particles in the first free-particle file. */
					if (part_in_profile_method==3 && filenr == first_free_part_filenr_this_snap[j_partType] )
					{
						n = dims_particles[0] - first_free_part_idx_in_file_this_snap[j_partType];
					}
					else
					{
						n = dims_particles[0];
					}

					double *flat_pos = h5_alloc_read_2d_double(dims_particles[0], dims_particles[1], h5_dataset);
					double (*pos_totpar)[dims_particles[1]] = (double (*)[dims_particles[1]])flat_pos;
					
					h5_status = ma_H5Dclose(h5_dataset);

					if (projection_flag == 1)
					{
						for (i=0; i<(int)dims_particles[0]; i++)
						{
							pos_totpar[i][2] = 0; // This is to project the particles onto the xy plane
						}
					}

					snprintf(buf2, sizeof(buf2), "/PartType%d/Velocities", j_partType);
					h5_dataset = ma_H5Dopen2(h5_group, buf2);
					
					float *flat_vel = h5_alloc_read_2d_float(dims_particles[0], dims_particles[1], h5_dataset);
					float (*vel_totpar)[dims_particles[1]] = (float (*)[dims_particles[1]])flat_vel;
					
					h5_status = ma_H5Dclose(h5_dataset);

					float *mass_totpar = malloc(dims_particles[0] * sizeof(float));
					if (j_partType!=1)
					{
						snprintf(buf2, sizeof(buf2), "/PartType%d/Masses", j_partType);
						if (input_h5_exists(h5_group, buf2))
						{
							h5_dataset = ma_H5Dopen2(h5_group, buf2);
						}
						else if (j_partType==5)
						{
							// Some formats store black hole masses under this name.
							snprintf(buf2, sizeof(buf2), "/PartType%d/BH_Mass", j_partType);
							if (input_h5_exists(h5_group, buf2))
								h5_dataset = ma_H5Dopen2(h5_group, buf2);
							else
								h5_dataset = -1;
						}
						else
						{
							h5_dataset = -1;
						}
						if (h5_dataset >= 0)
						{
							h5_status = ma_H5Dread(mass_totpar, h5_dataset, H5T_NATIVE_FLOAT,
							                       dims_particles[0] * sizeof *mass_totpar);
							h5_status = ma_H5Dclose(h5_dataset);
						}
						else
						{
							input_h5_fatal("find required particle masses at", buf2);
						}
					}

					float *u_totpar = malloc(dims_particles[0] * sizeof(float));
					if (j_partType==0)
					{
						// Check which internal energy field exists (use H5Lexists
						// because ma_H5Dopen2 calls exit() on failure)
						snprintf(buf2, sizeof(buf2), "/PartType%d/InternalEnergy", j_partType);
						if (input_h5_exists(h5_group, buf2))
						{
							h5_dataset = ma_H5Dopen2(h5_group, buf2);
						}
						else
						{
							snprintf(buf2, sizeof(buf2), "/PartType%d/InternalEnergyOld", j_partType);
							if (input_h5_exists(h5_group, buf2))
							{
								h5_dataset = ma_H5Dopen2(h5_group, buf2);
							}
							else
							{
								h5_dataset = -1;
							}
						}
						if (h5_dataset >= 0)
						{
							h5_status = ma_H5Dread(u_totpar, h5_dataset, H5T_NATIVE_FLOAT,
							                       dims_particles[0] * sizeof *u_totpar);
							h5_status = ma_H5Dclose(h5_dataset);
						}
						else
						{
							printf("Warning: no InternalEnergy or InternalEnergyOld for PartType0 — setting u to zero\n");
							for (i1 = 0; i1 < (long long)dims_particles[0]; i1++)
								u_totpar[i1] = 0.0f;
						}
					}

					float *flat_metals_totpar = NULL;
					float (*metals_totpar)[n_metals] = NULL;

					if (j_partType == 0) {
							// The metals dataset is optional in some formats.
							snprintf(buf2, sizeof(buf2), "/PartType%d/GFM_Metals", j_partType);
							if (input_h5_exists(h5_group, buf2))
							{
								h5_dataset = ma_H5Dopen2(h5_group, buf2);

								flat_metals_totpar = h5_alloc_read_2d_float(dims_particles[0], n_metals, h5_dataset);

								metals_totpar = (float (*)[n_metals])flat_metals_totpar;

								ma_H5Dclose(h5_dataset);
							}
							else
							{
								printf("Warning: GFM_Metals not found for PartType0 — metals set to NULL\n");
							}
					}

					float *rho_totpar = malloc(dims_particles[0] * sizeof(float));
					if (j_partType==0)
					{
						snprintf(buf2, sizeof(buf2), "/PartType%d/Density", j_partType);
						h5_dataset = ma_H5Dopen2(h5_group, buf2);
						h5_status = ma_H5Dread(rho_totpar, h5_dataset, H5T_NATIVE_FLOAT,
						                       dims_particles[0] * sizeof *rho_totpar);
						h5_status = ma_H5Dclose(h5_dataset);
					}
					
					float *NeutralHydrogenAbundance_totpar = malloc(dims_particles[0] * sizeof(float));
					if (j_partType==0)
					{
						snprintf(buf2, sizeof(buf2), "/PartType%d/NeutralHydrogenAbundance", j_partType);
						if (input_h5_exists(h5_group, buf2))
						{
							h5_dataset = ma_H5Dopen2(h5_group, buf2);
							printf("NeutralHydrogenAbundance exists in the dataset\n");
							h5_status = ma_H5Dread(NeutralHydrogenAbundance_totpar, h5_dataset, H5T_NATIVE_FLOAT,
							                       dims_particles[0] * sizeof *NeutralHydrogenAbundance_totpar);
							h5_status = ma_H5Dclose(h5_dataset);
						}
						else
						{
							printf("NeutralHydrogenAbundance not found — setting to zero\n");
							for (i1 = 0; i1 < (long long)dims_particles[0]; i1++)
								NeutralHydrogenAbundance_totpar[i1] = 0;
						}
					}

					float *StarFormationRate_totpar = malloc(dims_particles[0] * sizeof(float));
					if (j_partType==0)
					{
						snprintf(buf2, sizeof(buf2), "/PartType%d/StarFormationRate", j_partType);
						if (input_h5_exists(h5_group, buf2))
						{
							h5_dataset = ma_H5Dopen2(h5_group, buf2);
							h5_status = ma_H5Dread(StarFormationRate_totpar, h5_dataset, H5T_NATIVE_FLOAT,
							                       dims_particles[0] * sizeof *StarFormationRate_totpar);
							h5_status = ma_H5Dclose(h5_dataset);
						}
						else
						{
							printf("StarFormationRate not found — setting to zero\n");
							for (i1 = 0; i1 < (long long)dims_particles[0]; i1++)
								StarFormationRate_totpar[i1] = 0;
						}
					}

					float *x_e_totpar = malloc(dims_particles[0] * sizeof(float));
					if (j_partType==0)
					{
						snprintf(buf2, sizeof(buf2), "/PartType%d/ElectronAbundance", j_partType);
						if (input_h5_exists(h5_group, buf2))
						{
							h5_dataset = ma_H5Dopen2(h5_group, buf2);
							h5_status = ma_H5Dread(x_e_totpar, h5_dataset, H5T_NATIVE_FLOAT,
							                       dims_particles[0] * sizeof *x_e_totpar);
							h5_status = ma_H5Dclose(h5_dataset);
						}
						else
						{
							printf("ElectronAbundance not found — setting to zero\n");
							for (i1 = 0; i1 < (long long)dims_particles[0]; i1++)
								x_e_totpar[i1] = 0;
						}
					}
					
					ma_H5Gclose(h5_group);
					h5_status = close_hdf5_input(h5_file, buf);


					double *L_bol_totpar = malloc(dims_particles[0] * sizeof(double));
					if (j_partType==0)
					{
						for (i1 = 0; i1 < (long long)dims_particles[0]; i1++)
						{
							L_bol_totpar[i1] = calculate_Xray_BolometricLum(x_e_totpar[i1], u_totpar[i1], mass_totpar[i1], rho_totpar[i1], StarFormationRate_totpar[i1]);
						}
					}

					if (part_in_profile_method==3 && filenr == first_free_part_filenr_this_snap[j_partType] )
					{
						for(i1 = 0; i1 < n; i1++)
						{
							j1 = i1+first_free_part_idx_in_file_this_snap[j_partType];
							for (k1 = 0; k1 < 3; k1++)
							{
								pos_totpar[i1][k1] = pos_totpar[j1][k1];
								vel_totpar[i1][k1] = vel_totpar[j1][k1];
							}
							mass_totpar[i1] = mass_totpar[j1];
							if (j_partType==0)
							{
								u_totpar[i1] = u_totpar[j1];
								rho_totpar[i1] = rho_totpar[j1];
								NeutralHydrogenAbundance_totpar[i1] = NeutralHydrogenAbundance_totpar[j1];
								StarFormationRate_totpar[i1] = StarFormationRate_totpar[j1];
								x_e_totpar[i1] = x_e_totpar[j1];
								L_bol_totpar[i1] = L_bol_totpar[j1];
								if (metals_totpar != NULL)
									for (k1 = 0; k1< n_metals; k1++)
										metals_totpar[i1][k1] = metals_totpar[j1][k1];
							}
						}
					}

					n_b_max = 501;
					n_b_center = 250;
					gal_bin_x1 = malloc(n_b_max * sizeof(int));
					gal_bin_y1 = malloc(n_b_max * sizeof(int));
					gal_bin_z1 = malloc(n_b_max * sizeof(int));

					nb = ceil(box_size/l_bin);


					project(pos_totpar, vel_totpar, metals_totpar, mass_totpar, u_totpar, rho_totpar, NeutralHydrogenAbundance_totpar, StarFormationRate_totpar, x_e_totpar, L_bol_totpar, r_smoothing, r_g, v_g, Ndim, n, sfsqrt, i_partType);

					free(gal_bin_x1);
					free(gal_bin_y1);
					free(gal_bin_z1);

					free(flat_pos);
					free(flat_vel);
					free(mass_totpar);
					free(u_totpar);
					free(rho_totpar);
					free(NeutralHydrogenAbundance_totpar);
					free(StarFormationRate_totpar);
					free(x_e_totpar);
					free(L_bol_totpar);
					free(flat_metals_totpar);

				}
				else
				{
					input_h5_fatal("find required coordinates at", buf2);
				}
			}
			else
			{
				printf("Group %s does not exist\n", buf2);
				h5_status = close_hdf5_input(h5_file, buf);
			}

		}
	}
	free(gal_bin_x);
	free(gal_bin_y);
	free(gal_bin_z);
	printf("Saving the output\n");
	char OutputDir_snap[1000];
	snprintf(OutputDir_snap, sizeof(OutputDir_snap), "%s/snap%d", OutputDir, SnapNum);
	struct stat st = {0};
	if (stat(OutputDir_snap, &st) == -1)
	{
		mkdir_p(OutputDir_snap);
	}

	/* Use only the last component of SimulationLabel for the filename. */
	const char *last = strrchr(SimulationLabel, '/');
	const char *name = last ? last + 1 : SimulationLabel;
	snprintf(buf, sizeof(buf), "%s/snap%d/profile_%s_snap%03d_%d.h5",
	        OutputDir, SnapNum, name, SnapNum, subfilenr);
	profile_save_hdf5(&pb, buf, Ndim, nbins_profile, n_partType, n_metals, pos_dimension, r_smoothing);

	profile_free(&pb);
	free(np_bg_limit);
	free(n_par_in_gal);
	free(r_g);
	free(v_g);
	free(r_smoothing);
	free(gr_M200);
}
void project(double pos_totpar[][3], float vel_totpar[][3], float metals_totpar[][n_metals], float mass_totpar[], float u_totpar[], float rho_totpar[], float NeutralHydrogenAbundance_totpar[], float StarFormationRate_totpar[], float x_e_totpar[], double L_bol_totpar[], float r_smoothing[], float r_g[][3], float v_g[][3], int Ndim, int n, float sfsqrt, int i_partType)
{
	{ double _rss, _hwm; get_mem_gb(&_rss, &_hwm);
	  printf("project has been started [mem: %.2f GB RSS, %.2f GB peak]\n", _rss, _hwm); }
	int i, n_b_1D, n_bin_x, n_bin_y, n_bin_z, k1_metal;
	long long numgal;
	double xx, yy, zz, r2, r2_s_o, r_s_o, r_s_o_z, vLOS_x_this_particle, vLOS_y_this_particle, vLOS_z_this_particle;
	double vparx, vpary, vparz, masspar_weight = 0.0;
	double temp_par = 0.0, entropy_par, L_bol_par = 0.0, mu, N_H0_par, N_e_par, N_H_par, sfr_par;
	double r2_s_o_profile, v_rad;
	int i_bin;


    if (i_partType==1 || is_hydro_sim==0) // If DM, or if DMO run, use the constant DM mass
	{
		masspar_weight = dm_mass;
	}

	nb_total = nb*nb*nb;
	pos_totpar_c = calloc(nb_total, sizeof(int));
	cpos_totpar_c = calloc(nb_total, sizeof(int));
	if (!pos_totpar_c || !cpos_totpar_c) {
		fprintf(stderr, "ERROR: project: calloc of %d ints failed (box=%g, BinSize=%g → nb=%d, nb_total=%d). Increase BinSize or reduce box.\n",
		        nb_total, box_size, l_bin, nb, nb_total);
		exit(1);
	}
	long long n_oob = 0; // out-of-box particle counter
	for(i = 0; i<n; i++)
	{
		k1 = floor(pos_totpar[i][0]/l_bin);
		k2 = floor(pos_totpar[i][1]/l_bin);
		k3 = floor(pos_totpar[i][2]/l_bin);
		if(k1 < 0 || k1 >= nb || k2 < 0 || k2 >= nb || k3 < 0 || k3 >= nb) { n_oob++; continue; }
		bin_idx = k1+nb*(k2+nb*k3);
		pos_totpar_c[bin_idx]+=1;
	}
	if(n_oob > 0)
		printf("Warning: %lld / %d particles outside box — skipped (wind particles?)\n", n_oob, n);

	// Build prefix sums to index particles within each spatial bin.
	for(i = 0; i<nb_total-1; i++)
	{
		cpos_totpar_c[i+1] = cpos_totpar_c[i] + pos_totpar_c[i];
	}
	np_in_bin = cpos_totpar_c[nb_total-1] + pos_totpar_c[nb_total-1];

	pos_totpar_i = malloc(nb_total * sizeof(int));
	if (!pos_totpar_i) {
		fprintf(stderr, "ERROR: project: malloc pos_totpar_i (%d ints) failed\n", nb_total);
		exit(1);
	}
	for (i = 0; i < nb_total; i++)
	{
		pos_totpar_i[i] = 0;
	}

	long long ggdds = np_in_bin;
	binpos_totpar = malloc(ggdds * sizeof(int));
	if (!binpos_totpar) {
		fprintf(stderr, "ERROR: project: malloc binpos_totpar (%lld ints) failed\n", ggdds);
		exit(1);
	}
	for(i = 0; i < ggdds; i++)
	{
		binpos_totpar[i] = -1;
	}
	{ double _rss, _hwm; get_mem_gb(&_rss, &_hwm);
	  printf("project binning arrays allocated (nb=%d, np_in_bin=%lld) [mem: %.2f GB RSS, %.2f GB peak]\n",
	         nb, ggdds, _rss, _hwm); }

	for(i = 0; i < n; i++)
	{
		double px = (double)pos_totpar[i][0];
		double py = (double)pos_totpar[i][1];
		double pz = (double)pos_totpar[i][2];
		k1 = (int)floor(px/l_bin);
		k2 = (int)floor(py/l_bin);
		k3 = (int)floor(pz/l_bin);
		if(k1 < 0 || k1 >= nb || k2 < 0 || k2 >= nb || k3 < 0 || k3 >= nb) continue;
		bin_idx = k1+nb*(k2+nb*k3);
		bin_idx1 = cpos_totpar_c[bin_idx] + pos_totpar_i[bin_idx];
		binpos_totpar[bin_idx1] = i;
		pos_totpar_i[bin_idx]+=1;
	}
	free(pos_totpar_i);


	// Per-thread scratch arrays: gal_bin_x1/y1/z1 are global pointers shared across
	// the halo loop iterations, so allocate one independent copy per OpenMP thread.
	int nthreads_omp = omp_get_max_threads();
	int **gbl_x_thr = malloc(nthreads_omp * sizeof(int *));
	int **gbl_y_thr = malloc(nthreads_omp * sizeof(int *));
	int **gbl_z_thr = malloc(nthreads_omp * sizeof(int *));
	for (int _t = 0; _t < nthreads_omp; _t++) {
		gbl_x_thr[_t] = malloc(n_b_max * sizeof(int));
		gbl_y_thr[_t] = malloc(n_b_max * sizeof(int));
		gbl_z_thr[_t] = malloc(n_b_max * sizeof(int));
	}

	long long print_every = (Ndim > 0) ? (Ndim / 100) : 1;
	if (print_every < 1) print_every = 1;

#pragma omp parallel for schedule(dynamic, 4) \
	private(n_b_1D, n_bin_x, n_bin_y, n_bin_z, k1_metal, i_bin, \
	        xx, yy, zz, r2, r2_s_o, r_s_o, r_s_o_z, r2_s_o_profile, v_rad, \
	        vparx, vpary, vparz, \
	        vLOS_x_this_particle, vLOS_y_this_particle, vLOS_z_this_particle, \
	        entropy_par, mu, N_H0_par, N_e_par, N_H_par, sfr_par, \
	        par_in_r_size, bin_idx, bin_idx1, \
	        x_g, y_g, z_g, x_g1, y_g1, z_g1, vx_g, vy_g, vz_g) \
	firstprivate(masspar_weight, temp_par, L_bol_par)
	for (numgal = 0; numgal<Ndim; numgal++)
	{
		// Shadow the global gal_bin_x1/y1/z1 with thread-local arrays.
		int _tid = omp_get_thread_num();
		int *gal_bin_x1 = gbl_x_thr[_tid];
		int *gal_bin_y1 = gbl_y_thr[_tid];
		int *gal_bin_z1 = gbl_z_thr[_tid];

		if (numgal % print_every == 0)
		{
			#pragma omp critical
			{
				double _rss, _hwm; get_mem_gb(&_rss, &_hwm);
				printf("halo number = %lld / %d (%.0f%%) [mem: %.2f GB RSS, %.2f GB peak]\n",
				       numgal, Ndim, 100.0*numgal/Ndim, _rss, _hwm);
			}
		}
			x_g = r_g[numgal][0];
			y_g = r_g[numgal][1];
			z_g = r_g[numgal][2];
			vx_g = v_g[numgal][0]/sf;
			vy_g = v_g[numgal][1]/sf;
			vz_g = v_g[numgal][2]/sf;
			par_in_r_size = n_par_in_gal[numgal];

			gal_bin_x1[n_b_center] = gal_bin_x[numgal]; // Center of the local bin-index window.
			gal_bin_y1[n_b_center] = gal_bin_y[numgal];
			gal_bin_z1[n_b_center] = gal_bin_z[numgal];


			if(gal_bin_x1[n_b_center]==nb)
			{
				gal_bin_x1[n_b_center]-=1;
			}
			if(gal_bin_y1[n_b_center]==nb)
			{
				gal_bin_y1[n_b_center]-=1;
			}
			if(gal_bin_z1[n_b_center]==nb)
			{
				gal_bin_z1[n_b_center]-=1;
			}
			r_s_o = r_smoothing[numgal]*nRvir_profile;

			if (projection_flag == 0)
			{
				r_s_o_z = r_s_o;
			}
			else
			{
				r_s_o_z = -1; // Sentinel that suppresses extra z-bin searches in projection mode.
			}

			r2_s_o = r_s_o*r_s_o;

			// Search enough neighboring bins to cover the profile radius.
			x_g1 = x_g; // initially only, will change in few lines below
			y_g1 = y_g;
			z_g1 = z_g;
			if (x_g1-r_s_o>gal_bin_x1[n_b_center]*l_bin && x_g1+r_s_o<(gal_bin_x1[n_b_center]+1)*l_bin && y_g1-r_s_o>gal_bin_y1[n_b_center]*l_bin && y_g1+r_s_o<(gal_bin_y1[n_b_center]+1)*l_bin && z_g1-r_s_o_z>gal_bin_z1[n_b_center]*l_bin && z_g1+r_s_o_z<(gal_bin_z1[n_b_center]+1)*l_bin)
			{
				n_b_1D = 0;
			}
			else
			{
				n_b_1D = floor(r_s_o/l_bin)+1;
			}
			int z_i, y_i, x_i;
			for(z_i = -n_b_1D; z_i<n_b_1D+1; z_i++)
			{
				for(y_i = -n_b_1D; y_i<n_b_1D+1; y_i++)
				{
					for(x_i = -n_b_1D; x_i<n_b_1D+1; x_i++)
					{
						n_bin_x = n_b_center + x_i;
						n_bin_y = n_b_center + y_i;
						n_bin_z = n_b_center + z_i;
						gal_bin_x1[n_bin_x] = gal_bin_x1[n_b_center] + x_i;
						gal_bin_y1[n_bin_y] = gal_bin_y1[n_b_center] + y_i;
						gal_bin_z1[n_bin_z] = gal_bin_z1[n_b_center] + z_i;
						if(gal_bin_x1[n_bin_x]<0)
						{
							gal_bin_x1[n_bin_x] = gal_bin_x1[n_bin_x] + nb;
							x_g1 = x_g + box_size;
						}
						else if(gal_bin_x1[n_bin_x]>nb-1)
						{
							gal_bin_x1[n_bin_x] = gal_bin_x1[n_bin_x] - nb;
							x_g1 = x_g - box_size;
						}
						else
						{
							x_g1 = x_g;
						}
						if(gal_bin_y1[n_bin_y]<0)
						{
							gal_bin_y1[n_bin_y] = gal_bin_y1[n_bin_y] + nb;
							y_g1 = y_g + box_size;
						}
						else if(gal_bin_y1[n_bin_y]>nb-1)
						{
							gal_bin_y1[n_bin_y] = gal_bin_y1[n_bin_y] - nb;
							y_g1 = y_g - box_size;
						}
						else
						{
							y_g1 = y_g;
						}
						if(gal_bin_z1[n_bin_z]<0)
						{
							gal_bin_z1[n_bin_z] = gal_bin_z1[n_bin_z] + nb;
							z_g1 = z_g + box_size;
						}
						else if(gal_bin_z1[n_bin_z]>nb-1)
						{
							gal_bin_z1[n_bin_z] = gal_bin_z1[n_bin_z] - nb;
							z_g1 = z_g - box_size;
						}
						else
						{
							z_g1 = z_g;
						}
						bin_idx = gal_bin_x1[n_bin_x]+nb*(gal_bin_y1[n_bin_y]+nb*gal_bin_z1[n_bin_z]);
						if(pos_totpar_c[bin_idx]>0)
						{
							int numpar;
							for (numpar = 0; numpar<pos_totpar_c[bin_idx]; numpar++)
							{
								bin_idx1 = cpos_totpar_c[bin_idx]+numpar;

								if(binpos_totpar[bin_idx1]<0)
								{
									break;
								}
								else
								{
									xx = pos_totpar[binpos_totpar[bin_idx1]][0] - x_g1;
									yy = pos_totpar[binpos_totpar[bin_idx1]][1] - y_g1;
									zz = pos_totpar[binpos_totpar[bin_idx1]][2] - z_g1;

									r2 = xx*xx + yy*yy + zz*zz;

									r2_s_o_profile = r2_s_o;
									i_bin = 0;
									if(r2 < r2_s_o)
									{
										while(r2 < r2_s_o_profile && i_bin<nbins_profile)
										{											
											if(i_partType!=1 && is_hydro_sim)
											{
												masspar_weight = mass_totpar[binpos_totpar[bin_idx1]];
											}

											if (output_np_mass) {
											  pb.np_bg[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.np_bg[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + 1;
											  pb.mass_bg[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.mass_bg[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + masspar_weight;
											}

											vparx = vel_totpar[binpos_totpar[bin_idx1]][0]*sfsqrt - vx_g;
											vpary = vel_totpar[binpos_totpar[bin_idx1]][1]*sfsqrt - vy_g;
											vparz = vel_totpar[binpos_totpar[bin_idx1]][2]*sfsqrt - vz_g;

											v_rad = (vparx*xx + vpary*yy + vparz*zz)/sqrt(r2);

											/* Hubble flow uses the physical separation in Mpc. */
											v_rad = v_rad + (sqrt(r2)*sf/1000/h_const) * Hubble_Parameter;
											/* LOS components use the signed displacement along each axis. */
											vLOS_x_this_particle = vparx + (xx*sf/1000/h_const) * Hubble_Parameter;
											vLOS_y_this_particle = vpary + (yy*sf/1000/h_const) * Hubble_Parameter;
											vLOS_z_this_particle = vparz + (zz*sf/1000/h_const) * Hubble_Parameter;

											if (v_rad>0)
											{
												if (output_outflow) {
												  pb.num_vrad_out[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.num_vrad_out[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + 1;
												  pb.mass_out[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.mass_out[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + masspar_weight;
												  pb.vrad_times_mass_out[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vrad_times_mass_out[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + v_rad*masspar_weight;
												  pb.vLOS_x_out[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vLOS_x_out[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + vLOS_x_this_particle*masspar_weight;
												  pb.vLOS_y_out[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vLOS_y_out[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + vLOS_y_this_particle*masspar_weight;
												  pb.vLOS_z_out[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vLOS_z_out[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + vLOS_z_this_particle*masspar_weight;
												}

												if (output_outflow_vectors) {
												  pb.vLOS_out[nbins_profile*Ndim*n_partType*0 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vLOS_out[nbins_profile*Ndim*n_partType*0 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + vLOS_x_this_particle*masspar_weight;
												  pb.vLOS_out[nbins_profile*Ndim*n_partType*1 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vLOS_out[nbins_profile*Ndim*n_partType*1 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + vLOS_y_this_particle*masspar_weight;
												  pb.vLOS_out[nbins_profile*Ndim*n_partType*2 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vLOS_out[nbins_profile*Ndim*n_partType*2 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + vLOS_z_this_particle*masspar_weight;

												  pb.matter_outflow_vector[nbins_profile*Ndim*n_partType*0 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.matter_outflow_vector[nbins_profile*Ndim*n_partType*0 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + v_rad*masspar_weight*xx;
												  pb.matter_outflow_vector[nbins_profile*Ndim*n_partType*1 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.matter_outflow_vector[nbins_profile*Ndim*n_partType*1 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + v_rad*masspar_weight*yy;
												  pb.matter_outflow_vector[nbins_profile*Ndim*n_partType*2 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.matter_outflow_vector[nbins_profile*Ndim*n_partType*2 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + v_rad*masspar_weight*zz;
												}

												if (output_outflow_fast) {
												  if (v_rad>v_rad_lim1)
												  {
													  pb.mass_out_fast1[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.mass_out_fast1[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + masspar_weight;
													  pb.vrad_times_mass_out_fast1[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vrad_times_mass_out_fast1[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + v_rad*masspar_weight;
													  if (v_rad>v_rad_lim2)
													  {
														  pb.mass_out_fast2[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.mass_out_fast2[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + masspar_weight;
														  pb.vrad_times_mass_out_fast2[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vrad_times_mass_out_fast2[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + v_rad*masspar_weight;
														  if (v_rad>v_rad_lim3)
														  {
															  pb.mass_out_fast3[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.mass_out_fast3[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + masspar_weight;
															  pb.vrad_times_mass_out_fast3[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vrad_times_mass_out_fast3[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + v_rad*masspar_weight;
														  }
													  }
												  }
												}
											}

											if (output_vrad)
											  pb.vrad_times_mass[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vrad_times_mass[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + v_rad*masspar_weight;
											if (output_vLOS) {
											  pb.vLOS_x[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vLOS_x[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + vLOS_x_this_particle*masspar_weight;
											  pb.vLOS_y[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vLOS_y[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + vLOS_y_this_particle*masspar_weight;
											  pb.vLOS_z[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vLOS_z[nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + vLOS_z_this_particle*masspar_weight;
											}

											if (output_flow_vectors) {
											  pb.vLOS[nbins_profile*Ndim*n_partType*0 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vLOS[nbins_profile*Ndim*n_partType*0 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + vLOS_x_this_particle*masspar_weight;
											  pb.vLOS[nbins_profile*Ndim*n_partType*1 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vLOS[nbins_profile*Ndim*n_partType*1 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + vLOS_y_this_particle*masspar_weight;
											  pb.vLOS[nbins_profile*Ndim*n_partType*2 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.vLOS[nbins_profile*Ndim*n_partType*2 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + vLOS_z_this_particle*masspar_weight;

											  pb.matter_flow_vector[nbins_profile*Ndim*n_partType*0 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.matter_flow_vector[nbins_profile*Ndim*n_partType*0 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + v_rad*masspar_weight*xx;
											  pb.matter_flow_vector[nbins_profile*Ndim*n_partType*1 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.matter_flow_vector[nbins_profile*Ndim*n_partType*1 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + v_rad*masspar_weight*yy;
											  pb.matter_flow_vector[nbins_profile*Ndim*n_partType*2 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] = pb.matter_flow_vector[nbins_profile*Ndim*n_partType*2 + nbins_profile*Ndim*i_partType + numgal*nbins_profile + i_bin] + v_rad*masspar_weight*zz;
											}

											/* Optional gas thermodynamic and composition profiles. */
											if(i_partType==0 && calculate_all_gas_properties==1 && is_hydro_sim)
											{
												int compute_temp   = output_temperature;
												int compute_xray   = output_xray;
												int compute_ion    = output_ionization;
												int compute_metals = output_metals && (metals_totpar != NULL);
												int compute_metals_xrw = output_metals_xray_weighted && (metals_totpar != NULL);

												if (compute_temp || compute_xray) {
													sfr_par = StarFormationRate_totpar[binpos_totpar[bin_idx1]];
													mu = 4*m_p/(1 + 3*X_H + 4*X_H*x_e_totpar[binpos_totpar[bin_idx1]]);
													temp_par = (gamma_gas-1)*(u_totpar[binpos_totpar[bin_idx1]]/k_B)*mu*pow(10,10);
													L_bol_par = L_bol_totpar[binpos_totpar[bin_idx1]];
													if (sfr_par>0) temp_par = temp_starForming_gas_cell;
												}

												if (compute_temp) {
													pb.temp_bg[numgal*nbins_profile + i_bin] = pb.temp_bg[numgal*nbins_profile + i_bin] + temp_par*masspar_weight;
													entropy_par = k_B*temp_par/pow(x_e_totpar[binpos_totpar[bin_idx1]]*X_H*rho_totpar[binpos_totpar[bin_idx1]]/m_p, 2.0/3.0);
													pb.entropy[numgal*nbins_profile + i_bin] = pb.entropy[numgal*nbins_profile + i_bin] + entropy_par*masspar_weight;
													if (temp_par>Temp_lim1) {
														pb.mass_bg_TempAboveLim1[numgal*nbins_profile + i_bin] += masspar_weight;
														if (temp_par>Temp_lim2) {
															pb.mass_bg_TempAboveLim2[numgal*nbins_profile + i_bin] += masspar_weight;
															if (temp_par>Temp_lim3)
																pb.mass_bg_TempAboveLim3[numgal*nbins_profile + i_bin] += masspar_weight;
														}
													}
												}

												if (compute_xray) {
													pb.L_bol_bg[numgal*nbins_profile + i_bin] += L_bol_par;
													pb.L_bol_times_mass_bg[numgal*nbins_profile + i_bin] += L_bol_par*masspar_weight;
												}

												if (compute_metals) {
													for (k1_metal = 0; k1_metal< n_metals; k1_metal++)
														pb.metals_all[nbins_profile*Ndim*k1_metal + numgal*nbins_profile + i_bin] += masspar_weight*metals_totpar[binpos_totpar[bin_idx1]][k1_metal];
												}

												if (compute_metals_xrw) {
													for (k1_metal = 0; k1_metal< n_metals; k1_metal++) {
														pb.metals_all_emission_weighted[nbins_profile*Ndim*k1_metal + numgal*nbins_profile + i_bin] += L_bol_par*metals_totpar[binpos_totpar[bin_idx1]][k1_metal];
														pb.metals_all_emission_weighted_e2[nbins_profile*Ndim*k1_metal + numgal*nbins_profile + i_bin] += L_bol_par*masspar_weight*metals_totpar[binpos_totpar[bin_idx1]][k1_metal];
													}
												}

												if (compute_ion) {
													N_H_par  = X_H*masspar_weight/m_p;
													N_H0_par = NeutralHydrogenAbundance_totpar[binpos_totpar[bin_idx1]]*N_H_par;
													pb.N_H0[numgal*nbins_profile + i_bin] += N_H0_par;
													pb.N_H[numgal*nbins_profile + i_bin]  += N_H_par;
													N_e_par = x_e_totpar[binpos_totpar[bin_idx1]]*N_H_par;
													pb.N_e[numgal*nbins_profile + i_bin] += N_e_par;
												}

												if (v_rad>0)
												{
													if (compute_temp) {
														pb.temp_out[numgal*nbins_profile + i_bin] += temp_par*masspar_weight;
														if (temp_par>Temp_lim1) {
															pb.mass_bg_out_TempAboveLim1[numgal*nbins_profile + i_bin] += masspar_weight;
															if (temp_par>Temp_lim2) {
																pb.mass_bg_out_TempAboveLim2[numgal*nbins_profile + i_bin] += masspar_weight;
																if (temp_par>Temp_lim3)
																	pb.mass_bg_out_TempAboveLim3[numgal*nbins_profile + i_bin] += masspar_weight;
															}
														}
													}

													if (compute_metals) {
														for (k1_metal = 0; k1_metal< n_metals; k1_metal++)
															pb.metals_all_out[nbins_profile*Ndim*k1_metal + numgal*nbins_profile + i_bin] += masspar_weight*metals_totpar[binpos_totpar[bin_idx1]][k1_metal];
													}

													if (v_rad>v_rad_lim1) {
														if (compute_temp)
															pb.temp_out_fast1[numgal*nbins_profile + i_bin] += temp_par*masspar_weight;
														if (compute_metals)
															for (k1_metal = 0; k1_metal< n_metals; k1_metal++)
																pb.metals_all_out_fast1[nbins_profile*Ndim*k1_metal + numgal*nbins_profile + i_bin] += masspar_weight*metals_totpar[binpos_totpar[bin_idx1]][k1_metal];
														if (v_rad>v_rad_lim2) {
															if (compute_temp)
																pb.temp_out_fast2[numgal*nbins_profile + i_bin] += temp_par*masspar_weight;
															if (compute_metals)
																for (k1_metal = 0; k1_metal< n_metals; k1_metal++)
																	pb.metals_all_out_fast2[nbins_profile*Ndim*k1_metal + numgal*nbins_profile + i_bin] += masspar_weight*metals_totpar[binpos_totpar[bin_idx1]][k1_metal];
															if (v_rad>v_rad_lim3) {
																if (compute_temp)
																	pb.temp_out_fast3[numgal*nbins_profile + i_bin] += temp_par*masspar_weight;
																if (compute_metals)
																	for (k1_metal = 0; k1_metal< n_metals; k1_metal++)
																		pb.metals_all_out_fast3[nbins_profile*Ndim*k1_metal + numgal*nbins_profile + i_bin] += masspar_weight*metals_totpar[binpos_totpar[bin_idx1]][k1_metal];
															}
														}
													}
												}
											}

											i_bin+=1;
											r2_s_o_profile = r2_s_o*pow((1-i_bin*1.0/nbins_profile),2);
										}

									}
								}
							}
						}
					}
				}
			}
	}

	for (int _t = 0; _t < nthreads_omp; _t++) {
		free(gbl_x_thr[_t]);
		free(gbl_y_thr[_t]);
		free(gbl_z_thr[_t]);
	}
	free(gbl_x_thr);
	free(gbl_y_thr);
	free(gbl_z_thr);

	{ double _rss, _hwm; get_mem_gb(&_rss, &_hwm);
	  printf("project done, freeing binning arrays [mem: %.2f GB RSS, %.2f GB peak]\n", _rss, _hwm); }
	free(cpos_totpar_c);
	free(pos_totpar_c);
	free(binpos_totpar);

}
double calculate_mean_molecular_weight_Arepo(double ElectronAbundance)
{
    double mu = 4 * m_p / (1 + 3 * X_H + 4 * X_H * ElectronAbundance);  // mean molecular weight
    return mu;
}

double calculate_temperature_Arepo(double InternalEnergy, double ElectronAbundance)
{
    double mu = calculate_mean_molecular_weight_Arepo(ElectronAbundance);
    double gas_temp = (gamma_gas - 1) * (InternalEnergy / k_B) * mu * pow(10, 10);  // kelvin
    return gas_temp;
}

double convert_units_Arepo(double data, const char* quantity)
{
    double mass_conversion = 1.989e33 * 1e10 / h_const;
    double scale_conversion = (3.086e24) / (1000 * h_const);
    double converted_data;

    if (strcmp(quantity, "mass") == 0)
	{
        converted_data = data * mass_conversion;
    }
	else if (strcmp(quantity, "density") == 0)
	{
        converted_data = data * mass_conversion / pow(scale_conversion, 3);
    } 
	else
	{
        printf("Quantity not recognized\n");
        return -1; // error
    }

    return converted_data;
}

static double kelvin_to_keV(double temperature)
{
    return temperature * 8.617333262145e-8; // Boltzmann constant in keV/K
}

double calculate_Xray_BolometricLum(double ElectronAbundance, double InternalEnergy, double Masses, double Density, double StarFormationRate)
{
    double mu = calculate_mean_molecular_weight_Arepo(ElectronAbundance);
    double temp = calculate_temperature_Arepo(InternalEnergy, ElectronAbundance);
	double log_temp = log10(temp);
	double temp_keV = kelvin_to_keV(temp);
    double mass_cgs = convert_units_Arepo(Masses, "mass");
    double rho_cgs = convert_units_Arepo(Density, "density");

    double L_x = (1.2e-24 / pow(mu, 2)) * mass_cgs * rho_cgs * sqrt(temp_keV);
    if (StarFormationRate > 0.0)
	{
        L_x = 0.0;
    }

    // Implement a linear ramp from log(T)=6.0 to log(T)=5.8 over which we clip to zero
    double clip_value = (log_temp - 5.8) / 0.2;
    if (clip_value < 0.0)
	{
        clip_value = 0.0;
    }
	else if (clip_value > 1.0)
	{
        clip_value = 1.0;
    }
    L_x *= clip_value;

    L_x = L_x / 1e30;
    return L_x;
}


void mkdir_p(const char *dir) {
    char tmp[256];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", dir);
    len = strlen(tmp);
    if(tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for(p = tmp + 1; *p; p++)
        if(*p == '/') {
            *p = 0;
            mkdir(tmp, S_IRWXU);
            *p = '/';
        }
    mkdir(tmp, S_IRWXU);
}



double *h5_alloc_read_2d_double(size_t N, size_t M, hid_t dset) {
    double *buf;
    if (M != 0 && N > SIZE_MAX / sizeof *buf / M) {
        fprintf(stderr, "ERROR: %zux%zu double allocation overflows size_t\n", N, M);
        exit(EXIT_FAILURE);
    }
    size_t buffer_bytes = N * M * sizeof *buf;
    buf = malloc(buffer_bytes);
    if (!buf && buffer_bytes != 0) {
        fprintf(stderr,
                "ERROR: alloc %zux%zu doubles failed\n", N, M);
        exit(EXIT_FAILURE);
    }
    ma_H5Dread(buf, dset, H5T_NATIVE_DOUBLE, buffer_bytes);
    return buf;
}

float *h5_alloc_read_2d_float(size_t N, size_t M, hid_t dset) {
	float *buf;
	if (M != 0 && N > SIZE_MAX / sizeof *buf / M) {
		fprintf(stderr, "ERROR: %zux%zu float allocation overflows size_t\n", N, M);
		exit(EXIT_FAILURE);
	}
	size_t buffer_bytes = N * M * sizeof *buf;
	buf = malloc(buffer_bytes);
	if (!buf && buffer_bytes != 0) {
		fprintf(stderr,
						"ERROR: alloc %zux%zu floats failed\n", N, M);
		exit(EXIT_FAILURE);
	}
	ma_H5Dread(buf, dset, H5T_NATIVE_FLOAT, buffer_bytes);
	return buf;
}
