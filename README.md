# DemoIris species classifier

A containerised web-hostable demo of a iris species classifier, trained from the [iris species data-set](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)


## Deployment

```sh
docker build -t <docker tag> .
docker run -p 8000:8000 <image tag>
```

## Development

```sh
pipenv install --dev
pipenv shell
```
