# Docker 容器基础

## 镜像和容器是什么关系？

**镜像是只读模板，容器是镜像的运行实例**——类比：镜像是类，容器是对象；或镜像是程序安装包，容器是运行中的进程。

镜像采用**分层存储**（UnionFS）：Dockerfile 的每条指令生成一层，多层只读叠加，容器启动时在顶部加一个可写层。分层让镜像可缓存、可复用：相同基础层的镜像只需存一份。

常用命令：`docker build` 构建镜像、`docker run` 启动容器、`docker ps` 查看运行中容器、`docker exec` 进入容器。

## Dockerfile 怎么写更高效？

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

**最佳实践**：
1. **先 COPY 依赖清单再 COPY 代码**：代码变动时命中依赖层缓存，不必重装依赖
2. **用小基础镜像**：`slim` / `alpine` 变体，镜像从 1GB 降到百 MB 级
3. **多阶段构建**：编译阶段用完整工具链镜像，运行阶段只拷贝产物，显著瘦身
4. 合并 RUN 指令减少层数，清理包管理缓存

## Volume 和 Bind Mount 有什么区别？

容器可写层随容器删除而消失，持久化靠挂载：

1. **Volume**（`-v mydata:/data`）：Docker 管理的数据卷，存放在 Docker 目录，可备份、可被多容器共享，生产首选
2. **Bind Mount**（`-v /host/path:/container/path`）：把宿主机目录直接映射进容器，开发时同步代码用
3. **tmpfs**：内存挂载，存放临时敏感数据

## Docker Compose 解决什么问题？

Compose 用一个 `docker-compose.yml` 声明多容器应用（应用 + 数据库 + 缓存），一条 `docker compose up` 拉起全部并自动组网。

**核心配置**：`services`（各容器）、`depends_on`（启动顺序）、`networks`（内部 DNS，服务名即主机名）、`volumes`（持久化声明）。服务间通过服务名互相访问，如应用容器内连 `postgres:5432` 而非 localhost。

## 容器的资源怎么限制？

`docker run` 支持：
1. `--memory=512m --memory-swap=512m`：内存上限，超限触发 OOM Kill
2. `--cpus=1.5`：CPU 配额（1.5 核）
3. Kubernetes 更细粒度：requests（调度依据）/ limits（硬上限）

排查容器问题：`docker logs` 看日志、`docker stats` 看实时资源占用、`docker inspect` 看元数据与退出码（137 = OOM 或手动 kill）。
