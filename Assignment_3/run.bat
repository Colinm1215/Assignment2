@echo off
setlocal

set IMAGE_NAME=assignment_3
set CONTAINER_NAME=assignment_3_container
set WORKDIR=%cd%

docker build --no-cache -t %IMAGE_NAME% .
docker run --gpus all -it --rm --name %CONTAINER_NAME% -v "%WORKDIR%":/app %IMAGE_NAME%

endlocal