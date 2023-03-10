# Triton Server Example 

How to use.
1. model pt 파일 제작 
2. model inference flow config 파일 작성 
3. pre, post processing environment 제작 
3. Triton docker image pull & container run 
4. Client test 

### Create .pt 
- `save_model.py` 참고 
- [torchscript tutorial](https://tutorials.pytorch.kr/beginner/Intro_to_TorchScript_tutorial.html)
- `test_model.py` 참고하여 inference 동작 체크 필요

### Write config file
- model config: inference flow configuration 
- 보통의 경우 4가지로 구성: `ensemble`, `core model(.pt file)`, `preprocessing`, `postprocessing`
- 각 model마다 input, output dimension, name, type 지정
- ensemble의 경우 inferencee flow 명시 

### pre_env, post_env 
- conda 환경을 통해 구성 
- conda env 생성 후, 필요한 requirements install 
- `conda-pack -n ${conda_env_name} -o pre_env.tar.gz` 와 같은 방식으로 conda environment 패키징 
- pre, post 알맞은 모델 폴더에 conda env 위치시키기 (config.pbtxt에 env 경로 명시해야 함)

### Triton run
- Triton image pull [image tag list](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)
- example: `docker pull nvcr.io/nvidia/tritonserver:21.10-py3`
- docker container run 
- *** MODEL_FOLDER_PATH: triton 전체 모델이 있는 base folder path (여기서는 `./triton`)
- example: `docker run --gpus='"device=0"' -it --rm --shm-size=8g -p 8005:8000  -v ${MODEL_FOLDER_PATH}:/model_dir  nvcr.io/nvidia/tritonserver:21.10 tritonserver --model-repository=/model_dir --strict-model-config=false --model-control-mode=poll --repository-poll-secs=10 --backend-config=tensorflow,version=2 --log-verbose=1`

### Client test
- `client.py` 참고
