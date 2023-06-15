# HWWanomalous

## Setting up

```
source setup.sh
```

This will load the correct version of madgraph.

## Running madgraph

```
mg5_aMC script_SM.txt
```

To run the scripts `script_par.txt` and `script_impar.txt` first remember to change the respective coupling in the `SMEFTsim_U35_MwScheme_UFO/restric_SMlimit_massless.dat` card:

```
7 0.500000e+00 # cHW  for the CP-even coupling

4 1.200000e+00 # cHWtil for the CP-odd coupling
```

