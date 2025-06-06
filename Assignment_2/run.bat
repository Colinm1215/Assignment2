@echo off
setlocal

set IMAGE_NAME=Assignment_2
set CONTAINER_NAME=Assignment_2_container
set WORKDIR=%cd%

docker build --no-cache -t %IMAGE_NAME% .
docker run --gpus all -it --rm --name %CONTAINER_NAME% -v "%WORKDIR%":/app %IMAGE_NAME%

endlocal