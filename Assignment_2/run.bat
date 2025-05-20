@echo off
setlocal

set IMAGE_NAME=sentiment-bert
set CONTAINER_NAME=sentiment-bert-container
set WORKDIR=%cd%

docker build --no-cache -t %IMAGE_NAME% .
docker run --gpus all -it --rm --name %CONTAINER_NAME% -v "%WORKDIR%":/app %IMAGE_NAME%

endlocal