# VAEs in the Presence of Missing Data

Code for the paper VAEs in the Presence of Missing Data, Mark Collier, Alfredo Nazabal, Christopher K.I. Williams, 2020.

The notebook `VAEs_in_the_Presence_of_Missing_Data.ipynb` was used to produce the results in the paper.

In order to run the MNIST MCAR experiments, set:

```
MISSINGNESS_TYPE = 'MCAR'
DATASET = 'MNIST'
LIKELIHOOD = 'BERNOULLI'
```

MNIST MNAR:

```
MISSINGNESS_TYPE = 'MNAR'
DATASET = 'MNIST'
LIKELIHOOD = 'BERNOULLI'
```
SVHN MCAR:

```
MISSINGNESS_TYPE = 'MCAR'
DATASET = 'SVHN'
LIKELIHOOD = 'LOGISTIC_MIXTURE'
```

SVHN MNAR:

```
MISSINGNESS_TYPE = 'MNAR'
DATASET = 'SVHN'
LIKELIHOOD = 'LOGISTIC_MIXTURE'
```

Please adjust `NUM_RUNS` to set the number of replicates to run.
