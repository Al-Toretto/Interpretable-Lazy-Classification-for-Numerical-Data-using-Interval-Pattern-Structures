Firstly, you need to specify the environment name in the .sh files 

To run non-randomized experiments you need to specify the name of the dataset: glass, ionosphere, page-blocks, rice, sonar, spambase, waveform, wdbc, or winequality-red-bin.
```shell
bash classification.sh dataset
bash proximity.sh dataset
```
To run randomized experiments you need to specify the name of the dataset: glass, ionosphere, page-blocks, rice, sonar, spambase, waveform, wdbc or winequality-red-bin. Also you need to pass a method name. For randomization.sh: standard, standard-support or ratio-support. For randomization-proximity.sh: proximity, proximity-non-falsified or proximity-support.
```shell
bash randomization.sh dataset method
bash randomization-proximity.sh dataset method
```