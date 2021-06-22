# Base Images
## 从天池基础镜像构建
FROM myimage:v2

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]



# netCDF4-1.5.6-cp36-cp36m-manylinux2014_x86_64.whl
# numpy-1.19.2-cp36-cp36m-manylinux2010_x86_64.whl
# scikit_learn-0.23.1-cp36-cp36m-manylinux1_x86_64.whl
# joblib-1.0.1-py3-none-any.whl
# scipy-1.5.4-cp36-cp36m-manylinux1_x86_64.whl
# xarray


# Docker commands:
# docker build -t registry.cn-shenzhen.aliyuncs.com/gz_tianchi_submit/gz_tianchi_submit:1.5 .
# docker push registry.cn-shenzhen.aliyuncs.com/gz_tianchi_submit/gz_tianchi_submit:1.5