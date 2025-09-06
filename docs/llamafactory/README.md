# 下载镜像
docker pull hiyouga/llamafactory:latest

# 启动,只暴露主机上的 GPU 1 给容器
1. 创建容器
docker create --gpus "device=1" --net=host \
  --shm-size=10g \
  --cap-add=SYS_ADMIN \
  -v "$PWD":/workspace/verl \
  -v /etc/localtime:/etc/localtime:ro \
  -v /etc/timezone:/etc/timezone:ro \
  --name factory \
  hiyouga/llamafactory:latest sleep infinity
2. 启动容器
docker start factory
docker exec -it factory bash