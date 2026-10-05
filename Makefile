# Makefile for PROFI — OpenMP + fast-math build

EXEC     = PROFI
OBJS     = main.o allvars.o read_parameters.o ma_functions.o
INCL     = allvars.h list_functions.h Makefile

# h5pcc is the HDF5 (possibly MPI) compiler wrapper; it adds -lhdf5, -lz, etc.
CC       = h5pcc

# ---- Flags ----
# Architecture tuning is off by default so the build stays portable across
# x86-64 hosts.  For maximum performance on the build machine, run:
#     make ARCH="-march=native -mtune=native"
ARCH     =
WARN     = -Wall -Wextra -Wformat=2 -Wstrict-prototypes -Wmissing-prototypes -Wno-unused-parameter
OPT      = -O3 $(ARCH) -ffast-math -fno-math-errno -fno-trapping-math
LTO      = -flto
OMP      = -fopenmp
DEFS     = -DNDEBUG                  # strip asserts in release builds
CSTD     = -std=gnu99

CFLAGS   = $(CSTD) -g $(WARN) $(OPT) $(LTO) $(OMP) $(DEFS) $(OPTIONS) $(EXTRA_CFLAGS)
LDFLAGS  = $(LTO) $(OMP) $(EXTRA_LDFLAGS)
LIBS     =                             # h5pcc injects needed libs automatically

.PHONY: all clean release debug asan pgo-gen pgo-use

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS) $(LIBS)

# Rebuild objects if any header or the Makefile changes
$(OBJS): $(INCL)

# Profile-Guided Optimization (two-stage; optional)
pgo-gen: clean
	$(MAKE) EXTRA_CFLAGS="-fprofile-generate" EXTRA_LDFLAGS="-fprofile-generate" all
	@echo ">> Run the program now with a representative workload to collect profiles."

pgo-use: clean
	$(MAKE) EXTRA_CFLAGS="-fprofile-use -fprofile-correction" EXTRA_LDFLAGS="-fprofile-use" all

# Aggressive math — only if numerical reproducibility is not strict
fastmath:
	$(MAKE) EXTRA_CFLAGS="-Ofast -ffast-math -fno-math-errno -fno-trapping-math -ffinite-math-only" EXTRA_LDFLAGS="$(LTO)" all

# Plain release rebuild without extras
release:
	$(MAKE) EXTRA_CFLAGS="" EXTRA_LDFLAGS="" all

# Debug build with conservative optimization for easier debugging
debug:
	$(MAKE) EXTRA_CFLAGS="-O0 -g3 -DDEBUG" EXTRA_LDFLAGS="" all

# AddressSanitizer/UndefinedBehaviorSanitizer build for memory bug hunting
asan:
	$(MAKE) EXTRA_CFLAGS="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined" EXTRA_LDFLAGS="-fsanitize=address,undefined" all

clean:
	rm -f $(OBJS) $(EXEC)