## Location Testing

The underlying component of the COVID-19 model is agent movement. To calibrate where agents go and at what rates, we have ran the following location test.

This consists of preparing input transitions, preparing the synthetic population, running the experiment, and making sure the targets match the output.


### Run the Experiment
To see how well the location movement matches the targets, run the experiment and create target values:

```
python model_runs/location/src/test_targets.py
```

#### How to compare target values:
- Steady state targets for each facility type should be within ~10% of the target. It is likely the hospital types (UNC, Large, Small) will be below the value. This is because we don't model births, which make up 5-10% of hospitalizations. 
	- The NH steady state will be off. We do not have good NH data.

- Community -> : should be the most accurate. All values should be within 5%
- UNC -> : Only UNC to NH is unknown. The rest should be close
- Large -> : Large to UNC and Large to NH might be off by more than 10%
- Small -> : Small to UNC and Small to NH might be off by more than 10%
- LT -> : All targets will be close. For such small numbers though, we don't mind values being 1/2 or 2x their targets
- NH -> : We have minimal data here. Targets will be off.
- Deaths : All deaths other than community and NH deaths should be close. We don't have a good estimate for either one of theses.
