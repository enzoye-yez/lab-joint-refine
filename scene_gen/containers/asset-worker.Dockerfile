# 容器仅包含通用依赖；源码运行时只读挂载，不复制凭据或运行记录。
# Docker 官方 Python 镜像的 AWS 公共分发源；固定实际验证的多架构摘要。
FROM public.ecr.aws/docker/library/python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mujoco==3.3.7 Pillow==11.3.0 trimesh==4.7.4 defusedxml==0.7.1
ENV PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=osmesa
USER 65534:65534
WORKDIR /
