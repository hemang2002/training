# Kubernetes Commands Cheatsheet

A practical Kubernetes (`kubectl`) command reference focused on learning Kubernetes and using it for deployment.

Works with the Kubernetes built into **Docker Desktop** (Settings → Kubernetes → Kubeadm). Every command is a single line, so it works the same in PowerShell, Git Bash and Linux.

---

# 0. Docker vs Kubernetes in One Picture

```text
DOCKER                                  KUBERNETES
------                                  ----------
You run containers yourself.            You describe what you want.
                                        Kubernetes runs it and keeps it running.

docker run nginx                        kubectl create deployment my-nginx --image=nginx
   |                                       |
   v                                       v
1 container                             Deployment  ("I want 2 copies")
                                           |
                                           v
                                        ReplicaSet  (counts the copies)
                                           |
                                           v
                                        Pod   Pod   (each pod wraps 1 container)
                                           ^
                                           |
                                        Service     (one fixed address in front of the pods)
```

If a container dies in Docker, it stays dead.
If a pod dies in Kubernetes, a new one is created automatically.

---

# 1. Kubernetes Basics

## Check kubectl Version

```bash
kubectl version
```

Example:

```text
Client Version: v1.36.1      ← kubectl on your computer
Server Version: v1.36.1      ← the cluster that answered
```

Only the client (no cluster needed):

```bash
kubectl version --client
```

## Cluster Information

```bash
kubectl cluster-info
```

Example:

```text
Kubernetes control plane is running at https://kubernetes.docker.internal:6443
CoreDNS is running at https://kubernetes.docker.internal:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
```

## kubectl Help

```bash
kubectl help
```

Help for a specific command:

```bash
kubectl create deployment --help
```

Explain any field of any object (built-in documentation):

```bash
kubectl explain deployment
```

```bash
kubectl explain deployment.spec.replicas
```

---

# 2. Contexts (Which Cluster Am I Talking To?)

`kubectl` can know many clusters. A **context** = which cluster + which user.

## List Contexts

```bash
kubectl config get-contexts
```

The one with `*` is active.

## Show Current Context

```bash
kubectl config current-context
```

## Switch Context

```bash
kubectl config use-context docker-desktop
```

Error `current-context is not set` or `connection refused`? Docker Desktop → Kubernetes must be **running** (green). If it is: Docker Desktop → Kubernetes → **Reset cluster**.

---

# 3. Nodes

A node is a machine (real or virtual) that runs pods.

Docker Desktop has one node, called `docker-desktop`.

## List Nodes

```bash
kubectl get nodes
```

Example:

```text
NAME             STATUS   ROLES           AGE   VERSION
docker-desktop   Ready    control-plane   19h   v1.36.1
```

## More Details (IP, OS, Container Runtime)

```bash
kubectl get nodes -o wide
```

## Everything About a Node

```bash
kubectl describe node docker-desktop
```

Shows CPU/memory capacity, which pods run on it, and events.

---

# 4. Namespaces

A namespace is a folder for Kubernetes objects. Deleting a namespace deletes everything inside it.

## List Namespaces

```bash
kubectl get namespaces
```

Short form:

```bash
kubectl get ns
```

Default namespaces:

```text
default           ← used when you don't say -n
kube-system       ← Kubernetes' own parts (DNS, proxy, ...)
kube-public
kube-node-lease
```

## Create a Namespace

```bash
kubectl create namespace practice
```

## Use a Namespace in a Command

```bash
kubectl get pods -n practice
```

`-n` = namespace. Without it, `kubectl` uses `default`.

## Pods in All Namespaces

```bash
kubectl get pods -A
```

`-A` = all namespaces.

## Make a Namespace the Default (Optional)

```bash
kubectl config set-context --current --namespace=practice
```

Back to normal:

```bash
kubectl config set-context --current --namespace=default
```

## Delete a Namespace

```bash
kubectl delete namespace practice
```

Careful: removes everything inside it.

---

# 5. Pods

A pod is the smallest thing Kubernetes runs: one container (sometimes a few) with its own IP address.

## Run a Single Pod

```bash
kubectl run my-pod --image=nginx
```

Good for quick tests. For real apps use a Deployment (next section), because a lone pod is **not** recreated if it dies.

## List Pods

```bash
kubectl get pods
```

With IP address and node:

```bash
kubectl get pods -o wide
```

Keep watching (Ctrl + C to stop):

```bash
kubectl get pods -w
```

`-w` = watch.

## Pod Status Meanings

```text
ContainerCreating   ← downloading the image / starting
Running  0/1        ← running, but not READY yet (still loading)
Running  1/1        ← running and ready for traffic
Completed           ← finished its job (normal for jobs)
ErrImagePull        ← can't download the image
ImagePullBackOff    ← gave up for now; will retry later (wrong name/tag?)
CrashLoopBackOff    ← the app keeps crashing at start
OOMKilled           ← used more memory than its limit
Terminating         ← being deleted
```

## Everything About a Pod

```bash
kubectl describe pod my-pod
```

Read the **Events** at the bottom first. They tell you why a pod is stuck.

## Delete a Pod

```bash
kubectl delete pod my-pod
```

---

# 6. Deployments

A Deployment keeps a number of identical pods running. If one dies, it makes a new one.

## Create a Deployment

```bash
kubectl create deployment my-nginx --image=nginx
```

With 2 copies:

```bash
kubectl create deployment my-nginx --image=nginx --replicas=2
```

## List Deployments

```bash
kubectl get deployments
```

Short form:

```bash
kubectl get deploy
```

Example:

```text
NAME       READY   UP-TO-DATE   AVAILABLE   AGE
my-nginx   2/2     2            2           6s
```

`2/2` = 2 ready out of 2 wanted.

## See the Whole Family

```bash
kubectl get deploy,rs,pods
```

The names show who owns whom:

```text
my-nginx                       ← Deployment
my-nginx-7c4d7fdf59            ← ReplicaSet (made by the Deployment)
my-nginx-7c4d7fdf59-7rp5t      ← Pod (made by the ReplicaSet)
```

## Everything About a Deployment

```bash
kubectl describe deployment my-nginx
```

## Delete a Deployment

```bash
kubectl delete deployment my-nginx
```

This also deletes its ReplicaSet and pods.

---

# 7. Scaling

## Change the Number of Pods

```bash
kubectl scale deployment my-nginx --replicas=3
```

Check:

```bash
kubectl get pods
```

Scale down:

```bash
kubectl scale deployment my-nginx --replicas=1
```

---

# 8. Self-Healing

Delete one pod of a Deployment:

```bash
kubectl get pods
```

```bash
kubectl delete pod my-nginx-7c4d7fdf59-7rp5t
```

(Use one of **your** pod names.)

Check again:

```bash
kubectl get pods
```

A new pod with a new name and a small AGE appears. The Deployment saw "want 2, have 1" and fixed it.

---

# 9. Updates and Rollback

## Change the Image Version

```bash
kubectl set image deployment/my-nginx nginx=nginx:1.27
```

Format:

```text
kubectl set image deployment/DEPLOYMENT CONTAINER_NAME=IMAGE:TAG
```

`kubectl create deployment` names the container after the image (`nginx`).

## Watch the Update

```bash
kubectl rollout status deployment/my-nginx
```

Kubernetes replaces pods one by one, so the app never goes down.

## Check Which Image Is Running

```bash
kubectl get deployment my-nginx -o wide
```

## History

```bash
kubectl rollout history deployment/my-nginx
```

## Undo (Go Back One Version)

```bash
kubectl rollout undo deployment/my-nginx
```

## Restart All Pods (Same Version)

```bash
kubectl rollout restart deployment/my-nginx
```

Useful after changing a ConfigMap (section 15).

---

# 10. Services

Pods come and go and get new IP addresses. A Service gives them **one fixed address and name**, and spreads requests over the ready pods.

```text
             Service my-nginx  (fixed IP + DNS name "my-nginx")
              /            \
        Pod 10.1.0.47    Pod 10.1.0.48      ← IPs change, the Service doesn't
```

## Create a Service for a Deployment

Inside the cluster only (ClusterIP, the default):

```bash
kubectl expose deployment my-nginx --port=80
```

Reachable from your computer too (NodePort):

```bash
kubectl expose deployment my-nginx --port=80 --type=NodePort
```

## List Services

```bash
kubectl get services
```

Short form:

```bash
kubectl get svc
```

Example:

```text
NAME       TYPE       CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE
my-nginx   NodePort   10.97.181.77   <none>        80:31369/TCP   1s
```

`80:31369` means: Service port 80, opened on your computer as port **31369** (random, between 30000 and 32767).

Open:

```text
http://localhost:31369
```

(Use **your** number.)

## Which Pods Are Behind a Service?

```bash
kubectl get endpoints my-nginx
```

Example:

```text
NAME       ENDPOINTS
my-nginx   10.1.0.47:80,10.1.0.48:80
```

Only **ready** pods appear here. `<none>` = no pod matches (check labels, section 12).

A line `Warning: v1 Endpoints is deprecated` is harmless.

## Delete a Service

```bash
kubectl delete service my-nginx
```

---


# 11. Logs

## Show Logs

```bash
kubectl logs my-pod
```

Logs of a Deployment (picks one of its pods):

```bash
kubectl logs deployment/my-nginx
```

Last lines only:

```bash
kubectl logs deployment/my-nginx --tail=20
```

## Follow Logs

```bash
kubectl logs -f deployment/my-nginx
```

Press:

```text
Ctrl + C
```

to stop.

## Logs of All Pods with a Label

```bash
kubectl logs -l app=my-nginx --prefix
```

`--prefix` shows which pod each line came from.

## Logs of the Previous (Crashed) Container

```bash
kubectl logs my-pod --previous
```

Very useful for `CrashLoopBackOff`.

---

# 12. ConfigMaps (Settings)

Settings live **outside** the image. Change the setting, not the image.

## Create a ConfigMap

```bash
kubectl create configmap my-config --from-literal=APP_ENV=production --from-literal=LOG_LEVEL=INFO
```

## List and Show

```bash
kubectl get configmaps
```

```bash
kubectl describe configmap my-config
```
---

# 13. Secrets (Passwords, Keys)

Same idea as a ConfigMap, for sensitive values.

## Create a Secret

```bash
kubectl create secret generic my-secret --from-literal=API_KEY=change-me
```

## List

```bash
kubectl get secrets
```

`describe` doesn't print the values:

```bash
kubectl describe secret my-secret
```

Important: Secrets are only **base64-encoded**, not encrypted. Never commit real secrets to git.

# 14. Debugging Checklist

When something doesn't work, in this order:

```bash
kubectl get pods
```

Read **Events** at the bottom.

```bash
kubectl logs POD_NAME
```

```bash
kubectl logs POD_NAME --previous
```

```bash
kubectl get endpoints SERVICE_NAME
```

```bash
kubectl get events --sort-by=.metadata.creationTimestamp
```

| You see                  | Usually means                              | Look at                          |
| ------------------------ | ------------------------------------------ | -------------------------------- |
| `ImagePullBackOff`       | wrong image name/tag, or image not built   | `describe pod` → Events          |
| `CrashLoopBackOff`       | app crashes at start                       | `logs --previous`                |
| `Running 0/1`            | readiness check failing / still loading    | `describe pod` → Events          |
| `Pending`                | no node has enough CPU/memory              | `describe pod` → Events          |
| `OOMKilled`              | memory limit too small                     | `describe pod` → Last State      |
| Service `ENDPOINTS <none>` | selector doesn't match pod labels        | `get pods --show-labels`         |

---

# 15. Cleanup

## Delete One Object

```bash
kubectl delete deployment my-nginx
```

```bash
kubectl delete service my-nginx
```

## Delete Several at Once

```bash
kubectl delete deployment,service my-nginx
```

## Delete Everything in a Namespace

```bash
kubectl delete namespace practice
```

## See What's Left

```bash
kubectl get all
```

`service/kubernetes` in `default` is Kubernetes' own Service. Never delete it.

Be careful with cleanup commands: there is no "undo" for a deleted namespace.


# 16. Docker Command → Kubernetes Command

| I want to…                 | Docker                          | Kubernetes                                          |
| -------------------------- | ------------------------------- | --------------------------------------------------- |
| start an app               | `docker run -d --name web nginx` | `kubectl create deployment web --image=nginx`      |
| run several copies         | run it several times            | `kubectl scale deployment web --replicas=3`         |
| see what's running         | `docker ps`                     | `kubectl get pods`                                  |
| open it from my computer   | `-p 8080:80`                    | `kubectl expose deployment web --port=80 --type=NodePort` |
| read logs                  | `docker logs -f web`            | `kubectl logs -f deployment/web`                    |
| run a command inside       | `docker exec -it web bash`      | `kubectl exec -it deploy/web -- bash`               |
| see details                | `docker inspect web`            | `kubectl describe pod POD_NAME`                     |
| resource usage             | `docker stats web`              | `kubectl top pods`                                  |
| environment variable       | `-e KEY=value`                  | `kubectl set env deployment/web KEY=value`          |
| change version             | stop, remove, run again         | `kubectl set image deployment/web nginx=nginx:1.27` |
| describe a whole app       | `compose.yaml` + `docker compose up -d` | `k8s/*.yaml` + `kubectl apply -k k8s/`      |
| remove it                  | `docker rm -f web`              | `kubectl delete deployment web`                     |
| if it crashes              | stays dead                      | replaced automatically                              |

---

# 17. Typical Kubernetes Deployment Flow

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
docker push  →  Docker Registry
                     |
                     v
            Kubernetes YAML (Deployment, Service, ConfigMap)
                     |
                     v
              kubectl apply
                     |
                     v
       Kubernetes pulls the image and runs the pods
                     |
                     v
       Service gives users one stable address
```