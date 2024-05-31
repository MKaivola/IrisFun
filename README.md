# Basic classification program with database integration

This repo contains a simple classification program which uses a Postgres database as a data source.
These services are initalized using the `compose.yaml` file.

```bash
docker compose up .
```

This will create a service stack containing a Postgres database service and a Debian
environment where Python code is executed. Currently one has to enter the myapp container
directly in order to execute code.

- [ ] Implement the inference service as a server

The database is populated using the `data/db_creation.py` script. Currently, the script uses the
Vowel dataset from [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/).
Use `train_model.py` to train the sklearn Pipeline model and `predict_model.py` to predict for unlabelled
data. The "new" data is just the test data without the class labels.

Currently there is no functionality for adding additional data to the database beyond the `data/db_creation.py` script.

- [] Implement an API for adding new data to the existing database from the inference service
