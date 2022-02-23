#!/bin/bash
imageName=streamlitpredictapp:latest
containerName=Feedback

docker build -t $imageName -f Dockerfile  .

echo Delete old container...
docker rm -f $containerName

echo Run new container...
docker run -d -p 8501:8501 --name $containerName $imageName