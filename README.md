# Serve Pre-Trained PyTorch Model

## Introduction

Plain and simple way to run small pre-trained PyTorch models for testing.

## Setup

1. Clone the repository

```sh
git clone https://github.com/bensooraj/mlops-serve-mobilenetv3-plain.git
```

2. Install the dependencies

```sh
uv sync
```

3. Start the FastAPI server

```sh
make run
```

## Test

There are two images under the `scripts` folder:

```sh
$ ls -alh scripts/*.jpg(N) scripts/*.webp(N)          
-rw-r--r--  1 xyz  staff   169K Aug  6 17:57 scripts/frog_image.jpg
-rw-r--r--  1 xyz  staff   199K Aug  6 11:53 scripts/red_goldfish.webp
```

Run the following `curl` commands invoke the `/predict` api:

```sh
$ curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@scripts/frog_image.jpg;type=image/jpeg'

{"class_index":31,"class_name":"tree frog"}
```

and,

```sh
$ curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@scripts/red_goldfish.webp;type=image/webp'

{"class_index":1,"class_name":"goldfish"}
```

## Resources

1. [saveloadrun_tutorial](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)
2. [saving_loading_models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
3. [Deploying PyTorch with FastAPI to Handle 120K+ requests a Day!](https://medium.com/data-science-collective/deploying-pytorch-with-fastapi-to-handle-120k-requests-a-day-d4573a5767af)
4. [terraform-custom-pytorch-model-sagemaker-endpoint](https://github.com/MlOpsWithAM/terraform-custom-pytorch-model-sagemaker-endpoint)
