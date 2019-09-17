## Using Brontes with ML Flow

To use Brontes together with mlflow, you can adapt the `MLProject` file in the 
root of the repo.

To run an mlflow experiment, just use:

```sh
mlflow run git@github.com:afoncubierta/brontes.git -P batch_size=40
```

with the parameters you want to modify.

To visualize and compare the results of your experiment(s):

```sh
mlflow ui
```

## Version Control