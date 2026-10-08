# Docker Commands Cheatsheet

A practical Docker command reference focused on learning Docker and using it for deployment.

---

# 1. Docker Basics

## Check Docker Version

```bash
docker --version
```

Example:

```text
Docker version 28.x.x
```

## Docker Information

```bash
docker info
```

Shows information about your Docker installation, containers, images, storage, etc.

## Docker Help

```bash
docker help
```

Help for a specific command:

```bash
docker run --help
```

---

# 2. Docker Images

Images are templates used to create containers.

## Download an Image

```bash
docker pull nginx
```

Download a specific version:

```bash
docker pull nginx:latest
```

```bash
docker pull nginx:1.27
```

## List Images

```bash
docker images
```

Alternative:

```bash
docker image ls
```

## Remove an Image

```bash
docker rmi nginx
```

Or:

```bash
docker image rm nginx
```

Force remove:

```bash
docker rmi -f nginx
```

---

# 3. Docker Containers

A container is a running instance of an image.

## Create and Run a Container

```bash
docker run nginx
```

This runs Nginx in the foreground.

## Run in Background

```bash
docker run -d nginx
```

`-d` = detached/background mode.

## Give Container a Name

```bash
docker run -d --name my-nginx nginx
```

Now you can use:

```bash
docker stop my-nginx
```

instead of using the container ID.

---

# 4. Port Mapping

Containers have their own network.

To access a container from your computer, map a port.

```bash
docker run -d --name my-nginx -p 8080:80 nginx
```

Format:

```text
-p HOST_PORT:CONTAINER_PORT
```

Example:

```text
8080:80
```

Means:

```text
localhost:8080 → container:80
```

Open:

```text
http://localhost:8080
```

---

# 5. List Containers

## Running Containers

```bash
docker ps
```

## All Containers

```bash
docker ps -a
```

This includes stopped containers.

---

# 6. Stop a Container

```bash
docker stop my-nginx
```

Or using container ID:

```bash
docker stop CONTAINER_ID
```

---

# 7. Start a Stopped Container

```bash
docker start my-nginx
```

---

# 8. Restart a Container

```bash
docker restart my-nginx
```

---

# 9. Remove a Container

First stop it:

```bash
docker stop my-nginx
```

Then remove it:

```bash
docker rm my-nginx
```

Check:

```bash
docker ps -a
```

The container should no longer appear.

## Force Remove

If the container is running:

```bash
docker rm -f my-nginx
```

`-f` = force.

---

# 10. Automatically Remove Container

Use `--rm`:

```bash
docker run --rm nginx
```

When the container stops, Docker automatically removes the container.

Useful for temporary containers.

---

# 11. Docker Logs

## Show Logs

```bash
docker logs my-nginx
```

## Follow Logs

```bash
docker logs -f my-nginx
```

`-f` = follow.

Press:

```text
Ctrl + C
```

to stop following logs.

Important:

```bash
docker logs -f my-nginx
```

Here `-f` means **follow logs**.

But:

```bash
docker rm -f my-nginx
```

Here `-f` means **force**.

The meaning of a flag depends on the command.

---

# 12. Execute Commands Inside a Container

## Open Shell

```bash
docker exec -it my-nginx bash
```

`-i` = interactive

`-t` = terminal

Together:

```bash
-it
```

means interactive terminal.

Some lightweight images don't have Bash.

Use:

```bash
docker exec -it my-nginx sh
```

Exit:

```bash
exit
```

---

# 13. Execute a Single Command

You don't always need to enter the container.

Example:

```bash
docker exec my-nginx ls
```

Check Nginx HTML:

```bash
docker exec my-nginx ls /usr/share/nginx/html
```

Read the page:

```bash
docker exec my-nginx cat /usr/share/nginx/html/index.html
```

---

# 14. Environment Variables

Pass an environment variable:

```bash
docker run -d \
  --name my-app \
  -e APP_ENV=production \
  my-image
```

Multiple variables:

```bash
docker run -d \
  --name my-app \
  -e APP_ENV=production \
  -e PORT=8080 \
  my-image
```

Check environment variables:

```bash
docker exec my-app env
```

---

# 15. Volumes

Volumes are used to persist data.

## Create Volume

```bash
docker volume create my-volume
```

## List Volumes

```bash
docker volume ls
```

## Use a Volume

```bash
docker run -d \
  --name my-nginx \
  -v my-volume:/data \
  nginx
```

Format:

```text
-v VOLUME_NAME:CONTAINER_PATH
```

---

# 16. Bind Mount

Mount a local directory into a container:

```bash
docker run -d \
  --name my-nginx \
  -v $(pwd)/html:/usr/share/nginx/html \
  -p 8080:80 \
  nginx
```

Format:

```text
-v HOST_PATH:CONTAINER_PATH
```

This is useful during development.

---

# 17. Inspect a Container

```bash
docker inspect my-nginx
```

This provides detailed information about:

* Container ID
* Image
* Network
* Ports
* Volumes
* Environment variables
* IP address
* Configuration

---

# 18. Container Resource Usage

```bash
docker stats
```

For one container:

```bash
docker stats my-nginx
```

Shows:

* CPU
* Memory
* Network
* Disk I/O
* Process information

Press:

```text
Ctrl + C
```

to exit.

---

# 19. Docker Networks

## List Networks

```bash
docker network ls
```

## Create Network

```bash
docker network create my-network
```

## Run Container on Network

```bash
docker run -d \
  --name my-nginx \
  --network my-network \
  nginx
```

## Inspect Network

```bash
docker network inspect my-network
```

---

# 20. Dockerfile

A Dockerfile describes how to build your image.

Example:

```dockerfile
FROM nginx:latest

COPY index.html /usr/share/nginx/html/index.html
```

Build it:

```bash
docker build -t my-nginx .
```

`-t` = tag/name the image.

The `.` means:

```text
Use the current directory as the build context.
```

---

# 21. Build Docker Image

Basic:

```bash
docker build .
```

Give it a name:

```bash
docker build -t my-app .
```

Give it a version:

```bash
docker build -t my-app:1.0 .
```

Build latest:

```bash
docker build -t my-app:latest .
```

List it:

```bash
docker images
```

---

# 22. Docker Image Tags

Tag an image:

```bash
docker tag my-app myusername/my-app:latest
```

Example:

```bash
docker tag my-app username/my-app:1.0
```

---

# 23. Docker Login

Login to Docker Hub:

```bash
docker login
```

Logout:

```bash
docker logout
```

---

# 24. Push Image to Registry

Example:

```bash
docker tag my-app username/my-app:latest
```

Then:

```bash
docker push username/my-app:latest
```

Another server can pull it:

```bash
docker pull username/my-app:latest
```

---

# 25. Docker Compose

Docker Compose is used to run multiple containers together.

Typical example:

```text
Application
    |
    +--- Backend
    |
    +--- Database
    |
    +--- Redis
```

---

# 26. Docker Compose File

Example `compose.yaml`:

```yaml
services:

  nginx:
    image: nginx:latest
    ports:
      - "8080:80"

  redis:
    image: redis:latest
```

Start:

```bash
docker compose up
```

Start in background:

```bash
docker compose up -d
```

---

# 27. Docker Compose Commands

## Start

```bash
docker compose up
```

## Start in Background

```bash
docker compose up -d
```

## Stop and Remove Containers

```bash
docker compose down
```

## Show Containers

```bash
docker compose ps
```

## Show Logs

```bash
docker compose logs
```

## Follow Logs

```bash
docker compose logs -f
```

## Build Images

```bash
docker compose build
```

## Build and Start

```bash
docker compose up -d --build
```

This is very commonly used during deployment.

---

# 28. Docker Compose Example for Deployment

```yaml
services:

  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      NODE_ENV: production

  redis:
    image: redis:latest
```

Start:

```bash
docker compose up -d --build
```

Check:

```bash
docker compose ps
```

Logs:

```bash
docker compose logs -f
```

Stop:

```bash
docker compose down
```

---

# 29. Docker Cleanup

## Remove Stopped Containers

```bash
docker container prune
```

## Remove Unused Images

```bash
docker image prune
```

Remove unused images:

```bash
docker image prune -a
```

## Remove Unused Volumes

```bash
docker volume prune
```

## Remove Unused Networks

```bash
docker network prune
```

## General Cleanup

```bash
docker system prune
```

More aggressive:

```bash
docker system prune -a
```

Be careful with cleanup commands because they can remove resources you may still need.

---

# 30. Nginx Hands-On Practice

This section gives you a complete Docker exercise using only Nginx.

---

## Step 1: Download Nginx

```bash
docker pull nginx
```

Check:

```bash
docker images
```

---

## Step 2: Run Nginx

```bash
docker run -d --name my-nginx -p 8080:80 nginx
```

Check:

```bash
docker ps
```

Open:

```text
http://localhost:8080
```

You should see the Nginx welcome page.

---

## Step 3: Check Logs

```bash
docker logs my-nginx
```

Follow logs:

```bash
docker logs -f my-nginx
```

Press:

```text
Ctrl + C
```

---

## Step 4: Enter the Container

```bash
docker exec -it my-nginx bash
```

Inside the container:

```bash
pwd
```

List files:

```bash
ls
```

Check Nginx configuration:

```bash
ls /etc/nginx
```

Read configuration:

```bash
cat /etc/nginx/nginx.conf
```

Check website files:

```bash
ls /usr/share/nginx/html
```

Read the HTML:

```bash
cat /usr/share/nginx/html/index.html
```

Exit:

```bash
exit
```

---

# 31. Change Nginx Website

Run:

```bash
docker exec my-nginx sh -c 'echo "Hello from Docker!" > /usr/share/nginx/html/index.html'
```

Check:

```bash
docker exec my-nginx cat /usr/share/nginx/html/index.html
```

Open:

```text
http://localhost:8080
```

You should now see:

```text
Hello from Docker!
```

---

# 32. Inspect Nginx

```bash
docker inspect my-nginx
```

You can use this to understand:

* Container configuration
* IP address
* Port mapping
* Mounts
* Environment
* Network
* Image

---

# 33. Check Nginx Resources

```bash
docker stats my-nginx
```

Press:

```text
Ctrl + C
```

---

# 34. Stop Nginx

```bash
docker stop my-nginx
```

Check:

```bash
docker ps
```

The container isn't running.

Check all containers:

```bash
docker ps -a
```

You should see:

```text
my-nginx    Exited
```

---

# 35. Start Nginx Again

```bash
docker start my-nginx
```

Check:

```bash
docker ps
```

Open:

```text
http://localhost:8080
```

---

# 36. Restart Nginx

```bash
docker restart my-nginx
```

Check:

```bash
docker ps
```

---

# 37. Final Remove

Stop:

```bash
docker stop my-nginx
```

Remove container:

```bash
docker rm my-nginx
```

Check:

```bash
docker ps -a
```

The container should be gone.

---

# 38. Remove Nginx Image

Check images:

```bash
docker images
```

Remove:

```bash
docker rmi nginx
```

Check:

```bash
docker images
```

Now both the Nginx container and image are removed.

---

# 39. Complete Nginx Practice Sequence

If you want to practice everything from beginning to end:

```bash
docker pull nginx
```

```bash
docker run -d --name my-nginx -p 8080:80 nginx
```

```bash
docker ps
```

Open:

```text
http://localhost:8080
```

```bash
docker logs my-nginx
```

```bash
docker logs -f my-nginx
```

Press:

```text
Ctrl + C
```

```bash
docker exec -it my-nginx bash
```

Inside:

```bash
pwd
ls
ls /etc/nginx
cat /etc/nginx/nginx.conf
ls /usr/share/nginx/html
cat /usr/share/nginx/html/index.html
exit
```

Change page:

```bash
docker exec my-nginx sh -c 'echo "Hello from Docker!" > /usr/share/nginx/html/index.html'
```

Check:

```bash
docker exec my-nginx cat /usr/share/nginx/html/index.html
```

Inspect:

```bash
docker inspect my-nginx
```

Check resources:

```bash
docker stats my-nginx
```

Press:

```text
Ctrl + C
```

Stop:

```bash
docker stop my-nginx
```

Check:

```bash
docker ps -a
```

Start:

```bash
docker start my-nginx
```

Restart:

```bash
docker restart my-nginx
```

Stop:

```bash
docker stop my-nginx
```

Remove:

```bash
docker rm my-nginx
```

Check:

```bash
docker ps -a
```

Remove image:

```bash
docker rmi nginx
```

Check:

```bash
docker images
```

---

# 40. Most Important Docker Flags

| Flag        | Meaning                           | Example                    |
| ----------- | --------------------------------- | -------------------------- |
| `-d`        | Detached/background               | `docker run -d nginx`      |
| `--name`    | Give container a name             | `--name my-nginx`          |
| `-p`        | Map ports                         | `-p 8080:80`               |
| `-v`        | Mount volume/directory            | `-v data:/data`            |
| `-e`        | Environment variable              | `-e NODE_ENV=production`   |
| `-it`       | Interactive terminal              | `-it nginx bash`           |
| `--rm`      | Auto-remove container             | `docker run --rm nginx`    |
| `-f`        | Follow/force depending on command | `logs -f`, `rm -f`         |
| `-t`        | Tag image                         | `docker build -t my-app .` |
| `--network` | Connect to network                | `--network my-network`     |
| `--restart` | Container restart policy          | `--restart=always`         |

---

# 41. Important Command Differences

## Stop vs Remove

```bash
docker stop my-container
```

Stops the container.

```bash
docker rm my-container
```

Removes the container.

---

## Remove Container vs Remove Image

Container:

```bash
docker rm my-container
```

Image:

```bash
docker rmi my-image
```

---

## Start vs Run

```bash
docker start my-container
```

Starts an **existing** container.

```bash
docker run nginx
```

Creates a **new container** from an image and starts it.

---

## Exec vs Run

```bash
docker exec my-container command
```

Runs a command inside an **existing running container**.

```bash
docker run nginx
```

Creates a **new container**.

---

# 42. Common Deployment Commands

Build application:

```bash
docker build -t my-app:latest .
```

Run:

```bash
docker run -d \
  --name my-app \
  -p 3000:3000 \
  my-app:latest
```

Check:

```bash
docker ps
```

Logs:

```bash
docker logs -f my-app
```

Enter container:

```bash
docker exec -it my-app sh
```

Stop:

```bash
docker stop my-app
```

Remove:

```bash
docker rm my-app
```

---

# 43. Typical Docker Deployment Flow

The basic deployment process is:

```text
Application
     |
     v
Dockerfile
     |
     v
docker build
     |
     v
Docker Image
     |
     v
Docker Registry
     |
     v
Production Server
     |
     v
docker pull
     |
     v
docker run
```

Typical commands:

```bash
docker build -t my-app:1.0 .
```

```bash
docker tag my-app:1.0 username/my-app:1.0
```

```bash
docker push username/my-app:1.0
```

On the production server:

```bash
docker pull username/my-app:1.0
```

Then:

```bash
docker run -d \
  --name my-app \
  -p 3000:3000 \
  username/my-app:1.0
```

Check:

```bash
docker ps
```

Check logs:

```bash
docker logs -f my-app
```

---

# 44. Docker Commands to Memorize First

If you're learning Docker for deployment, memorize these first:

```bash
docker pull
docker images
docker build
docker run
docker ps
docker ps -a
docker stop
docker start
docker restart
docker rm
docker rmi
docker logs
docker exec
docker inspect
docker stats
docker volume
docker network
docker compose
docker login
docker tag
docker push
docker pull
```

And especially understand these flags:

```text
-d
--name
-p
-v
-e
-it
--rm
-f
-t
--network
--restart
```

---

# 45. Recommended Learning Order

Learn Docker in this order:

```text
1. Images
      ↓
2. Containers
      ↓
3. docker run
      ↓
4. Ports
      ↓
5. Logs
      ↓
6. docker exec
      ↓
7. Environment variables
      ↓
8. Volumes
      ↓
9. Networks
      ↓
10. Dockerfile
      ↓
11. docker build
      ↓
12. Docker Compose
      ↓
13. Docker Registry
      ↓
14. Deployment
      ↓
15. CI/CD
      ↓
16. Kubernetes
```

---

# Quick Cheat Sheet

```bash
# Docker
docker --version
docker info
docker help

# Images
docker pull nginx
docker images
docker rmi nginx

# Containers
docker run nginx
docker run -d nginx
docker run -d --name my-nginx nginx
docker ps
docker ps -a
docker stop my-nginx
docker start my-nginx
docker restart my-nginx
docker rm my-nginx
docker rm -f my-nginx

# Ports
docker run -d --name my-nginx -p 8080:80 nginx

# Logs
docker logs my-nginx
docker logs -f my-nginx

# Execute
docker exec -it my-nginx bash
docker exec -it my-nginx sh
docker exec my-nginx ls

# Environment
docker run -d -e APP_ENV=production my-app

# Volumes
docker volume create my-volume
docker volume ls
docker run -d -v my-volume:/data my-app

# Networks
docker network ls
docker network create my-network
docker network inspect my-network

# Build
docker build -t my-app .
docker build -t my-app:1.0 .

# Registry
docker login
docker tag my-app username/my-app:latest
docker push username/my-app:latest
docker pull username/my-app:latest

# Compose
docker compose up
docker compose up -d
docker compose down
docker compose ps
docker compose logs
docker compose logs -f
docker compose build
docker compose up -d --build

# Inspect / Monitoring
docker inspect my-app
docker stats my-app

# Cleanup
docker container prune
docker image prune
docker volume prune
docker network prune
docker system prune
```

---

# Key Rule to Remember

```text
IMAGE → CONTAINER

docker run
    ↓
creates container from image

docker stop
    ↓
stops container

docker start
    ↓
starts existing container

docker rm
    ↓
removes container

docker rmi
    ↓
removes image
```

This distinction is one of the most important things to understand when learning Docker.
