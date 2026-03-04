# Effect Evaluation
This directory evaluates the effect of the defense on the poisoning attack under BiVFL's framework. Meanwhile, we also compare the performance of the defense with the typical baseline.

## Running Experiments
To run the experiments, you can use the following command:
```bash
python3 main.py --config configs/config.yaml
```
To change the configuration and the defense method, you can modify the `configs/config.yaml` file as the alternative given in the `configs/config_example.yaml` file. 

We recommend you to use the `config_[model].yaml` file as the template to create your own configuration file when testing on [model].