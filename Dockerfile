FROM python:3.9

LABEL maintainer="yooongchun@foxmail.com"

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
ENV USER=zoz

RUN groupadd -r --gid 1000 $USER && useradd -r --uid 1000 -g $USER $USER

WORKDIR /home/$USER

# Install pip requirements
COPY requirements-deploy.txt .
RUN python -m pip install --no-cache-dir --upgrade -r requirements-deploy.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host https://pypi.tuna.tsinghua.edu.cn \
   && rm -rf /root/.cache/* \
   && rm -rf ~/.cache/*

COPY src src
COPY assets/vocabulary.txt assets/
COPY output/checkpoint/inference inference
RUN chown -R $USER:$USER /home

WORKDIR /home/$USER/src
USER $USER
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]