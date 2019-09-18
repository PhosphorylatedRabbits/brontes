# Brontes with MLflow

To use Brontes together with mlflow, you can adapt the `MLProject` file in the
root of the repo.
`MLProject` is a yaml file that describes how to run the experiment, together with
a description of the conda environment required:

```sh
mlflow run ./ -P batch_size=40 -P learning_rate=0.001
```

modifying the parameters as you wish

To visualize and compare the results of your experiment(s):

```sh
mlflow ui
```
