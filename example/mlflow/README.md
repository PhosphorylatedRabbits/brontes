## Using Brontes with ML Flow

To use Brontes together with mlflow, you can adapt the `MLProject` file in the 
root of the repo.

To run an mlflow experiment directly from the current version of the git repo, just use:

```sh
mlflow run git@github.com:afoncubierta/brontes.git -P batch_size=40
```

or 

```
mldflow run ./ -P batch_size=40 learning_rate=0.001

```

modifying the parameters as you wish

To visualize and compare the results of your experiment(s):

```sh
mlflow ui
```

Don't forget to version control / store your mlruns folder.