#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "list_functions.h"


#define DOUBLE 1
#define STRING 2
#define INT 3
#define MAXTAGS 300
#define PARAM_STRING_MAX 512

void read_parameter_file(const char *fname)
{
  FILE *fd;
  char buf[400], buf1[400], buf2[400], buf3[400];
  int i, j, nt = 0;
  int id[MAXTAGS];
  int required[MAXTAGS]; /* 1 = must appear in file, 0 = optional (default already set below) */
  void *addr[MAXTAGS];
  char tag[MAXTAGS][50];
  int errorFlag = 0;

  for(i = 0; i < MAXTAGS; i++)
    required[i] = 1;

  printf("\nreading parameter file:\n\n");

  strcpy(tag[nt], "OutputDir");
  addr[nt] = OutputDir;
  id[nt++] = STRING;

  strcpy(tag[nt], "SimulationDir");
  addr[nt] = SimulationDir;
  id[nt++] = STRING;

  strcpy(tag[nt], "InputDir");
  addr[nt] = InputDir;
  id[nt++] = STRING;

  strcpy(tag[nt], "NumberOfSnapshots");
  addr[nt] = &n_snaps;
  id[nt++] = INT;

  strcpy(tag[nt], "BinSize");
  addr[nt] = &l_bin;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "M200Min");
  addr[nt] = &gr_M200_minlim;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "M200Max");
  addr[nt] = &gr_M200_maxlim;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "nbins_profile");
  addr[nt] = &nbins_profile;
  id[nt++] = INT;

  strcpy(tag[nt], "nRvir_profile");
  addr[nt] = &nRvir_profile;
  id[nt++] = INT;

  strcpy(tag[nt], "Temp_lim1"); // Kelvin
  addr[nt] = &Temp_lim1;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "Temp_lim2"); // Kelvin
  addr[nt] = &Temp_lim2;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "Temp_lim3"); // Kelvin
  addr[nt] = &Temp_lim3;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "temp_starForming_gas_cell"); // Kelvin. The temperature of star forming gas cells will set to this value.
  addr[nt] = &temp_starForming_gas_cell;
  id[nt++] = DOUBLE;
  
  strcpy(tag[nt], "v_rad_lim1"); // km/s
  addr[nt] = &v_rad_lim1;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "v_rad_lim2"); // km/s
  addr[nt] = &v_rad_lim2;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "v_rad_lim3"); // km/s
  addr[nt] = &v_rad_lim3;
  id[nt++] = DOUBLE;

  strcpy(tag[nt], "n_metals");
  addr[nt] = &n_metals;
  id[nt++] = INT;

  strcpy(tag[nt], "projection_flag");
  addr[nt] = &projection_flag;
  id[nt++] = INT;

  strcpy(tag[nt], "calculate_all_gas_properties");
  addr[nt] = &calculate_all_gas_properties;
  id[nt++] = INT;

  strcpy(tag[nt], "is_hydro_sim"); // 1 for hydro simulations, 0 for their DMO counterpart
  addr[nt] = &is_hydro_sim;
  id[nt++] = INT;

  /* ── Optional output-selection flags ──────────────────────────────────────
   * All default to 1 (compute/save everything), matching legacy behaviour.
   * Any subset may be overridden in the parameter file; they are NOT
   * mandatory tags, so old parameter files continue to work unchanged. */
  output_np_mass = 1;
  strcpy(tag[nt], "output_np_mass");
  addr[nt] = &output_np_mass;
  id[nt] = INT; required[nt] = 0; nt++;

  output_vrad = 1;
  strcpy(tag[nt], "output_vrad");
  addr[nt] = &output_vrad;
  id[nt] = INT; required[nt] = 0; nt++;

  output_vLOS = 1;
  strcpy(tag[nt], "output_vLOS");
  addr[nt] = &output_vLOS;
  id[nt] = INT; required[nt] = 0; nt++;

  output_outflow = 1;
  strcpy(tag[nt], "output_outflow");
  addr[nt] = &output_outflow;
  id[nt] = INT; required[nt] = 0; nt++;

  output_outflow_fast = 1;
  strcpy(tag[nt], "output_outflow_fast");
  addr[nt] = &output_outflow_fast;
  id[nt] = INT; required[nt] = 0; nt++;

  output_flow_vectors = 1;
  strcpy(tag[nt], "output_flow_vectors");
  addr[nt] = &output_flow_vectors;
  id[nt] = INT; required[nt] = 0; nt++;

  output_outflow_vectors = 1;
  strcpy(tag[nt], "output_outflow_vectors");
  addr[nt] = &output_outflow_vectors;
  id[nt] = INT; required[nt] = 0; nt++;

  output_temperature = 1;
  strcpy(tag[nt], "output_temperature");
  addr[nt] = &output_temperature;
  id[nt] = INT; required[nt] = 0; nt++;

  output_xray = 1;
  strcpy(tag[nt], "output_xray");
  addr[nt] = &output_xray;
  id[nt] = INT; required[nt] = 0; nt++;

  output_ionization = 1;
  strcpy(tag[nt], "output_ionization");
  addr[nt] = &output_ionization;
  id[nt] = INT; required[nt] = 0; nt++;

  output_metals = 1;
  strcpy(tag[nt], "output_metals");
  addr[nt] = &output_metals;
  id[nt] = INT; required[nt] = 0; nt++;

  output_metals_xray_weighted = 1;
  strcpy(tag[nt], "output_metals_xray_weighted");
  addr[nt] = &output_metals_xray_weighted;
  id[nt] = INT; required[nt] = 0; nt++;

  /* ── Simulation identification ─────────────────────────────────── */
  /* SimulationLabel identifies this simulation in output files and auxiliary
   * data paths (such as the file-status list).
   * Example: SimulationLabel   MySimulation */
  SimulationLabel[0] = 0;
  strcpy(tag[nt], "SimulationLabel");
  addr[nt] = SimulationLabel;
  id[nt] = STRING; required[nt] = 1; nt++;

  /* ── Particle-type control ──────────────────────────────────────── */
  n_partType = 3; /* hydro default; DMO will auto-downgrade to 1 */
  strcpy(tag[nt], "n_partType");
  addr[nt] = &n_partType;
  id[nt] = INT; required[nt] = 0; nt++;

  cluster_subfile_mode = 0;
  strcpy(tag[nt], "cluster_subfile_mode");
  addr[nt] = &cluster_subfile_mode;
  id[nt] = INT; required[nt] = 0; nt++;

  /* ── Particle-in-profile method (0=all, 1=this FOF, 2=other FOFs, 3=unbound) ─ */
  part_in_profile_method = 0;
  strcpy(tag[nt], "part_in_profile_method");
  addr[nt] = &part_in_profile_method;
  id[nt] = INT; required[nt] = 0; nt++;

  /* Cosmology defaults (Planck 2015). */
  cosmo_h = 0.6774;
  strcpy(tag[nt], "cosmo_h");
  addr[nt] = &cosmo_h;
  id[nt] = DOUBLE; required[nt] = 0; nt++;

  cosmo_Omega_m = 0.3089;
  strcpy(tag[nt], "cosmo_Omega_m");
  addr[nt] = &cosmo_Omega_m;
  id[nt] = DOUBLE; required[nt] = 0; nt++;

  cosmo_Omega_b = 0.0486;
  strcpy(tag[nt], "cosmo_Omega_b");
  addr[nt] = &cosmo_Omega_b;
  id[nt] = DOUBLE; required[nt] = 0; nt++;

  cosmo_Omega_Lambda = 0.6911;
  strcpy(tag[nt], "cosmo_Omega_Lambda");
  addr[nt] = &cosmo_Omega_Lambda;
  id[nt] = DOUBLE; required[nt] = 0; nt++;


  if((fd = fopen(fname, "r")))
    {
    while(fgets(buf, sizeof(buf), fd) != NULL)
	{
    buf1[0] = 0;
    buf2[0] = 0;
    buf3[0] = 0;
    if(sscanf(buf, "%399s %399s %399s", buf1, buf2, buf3) < 2)
	    continue;

	  if(buf1[0] == '%')
	    continue;

	  for(i = 0, j = -1; i < nt; i++)
	    if(strcmp(buf1, tag[i]) == 0)
	      {
		j = i;
		tag[i][0] = 0;
		break;
	      }

	  if(j >= 0)
	    {
	      printf("%35s\t%10s\n", buf1, buf2);

	      switch (id[j])
		{
		case DOUBLE:
		  *((double *) addr[j]) = atof(buf2);
		  break;
		case STRING:
      snprintf((char *) addr[j], PARAM_STRING_MAX, "%s", buf2);
		  break;
		case INT:
		  *((int *) addr[j]) = atoi(buf2);
		  break;
		}
	    }
	  else
	    {
	      printf("Error in file %s:   Tag '%s' not allowed or multiple defined.\n", fname, buf1);
	      errorFlag = 1;
	    }
	}
      fclose(fd);

      i = strlen(SimulationDir);
      if(i > 0)
	if(SimulationDir[i - 1] != '/')
	  strcat(SimulationDir, "/");
    }
  else
    {
      printf("Parameter file %s not found.\n", fname);
      errorFlag = 1;
    }


  for(i = 0; i < nt; i++)
    {
      if(*tag[i] && required[i])
	{
	  printf("Error. I miss a value for tag '%s' in parameter file '%s'.\n", tag[i], fname);
	  errorFlag = 1;
	}
    }

  if(errorFlag)
    exit(1);

}
