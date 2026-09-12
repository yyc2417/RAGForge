# Python 进阶

## 什么是 GIL？如何绕过它？

GIL（全局解释器锁）是 CPython 的机制：同一时刻只有一个线程执行 Python 字节码。这意味着多线程无法利用多核做并行计算。

**绕过方案**：
1. **multiprocessing**：多进程各自持有解释器，真正并行，适合 CPU 密集任务
2. **concurrent.futures.ProcessPoolExecutor**：进程池的高级封装
3. **C 扩展**：NumPy 等库在 C 层释放 GIL，重计算仍可并行
4. **asyncio**：IO 密集任务用协程并发（不是并行），单线程内高效率切换

经验法则：CPU 密集用多进程，IO 密集用 asyncio 或多线程。

## asyncio 的事件循环是怎么工作的？

asyncio 是 Python 的异步 IO 框架，核心是事件循环（Event Loop）调度协程。

**关键概念**：
1. **协程（coroutine）**：`async def` 定义的函数，用 `await` 暂停让出控制权
2. **Task**：对协程的调度封装，`asyncio.create_task()` 启动
3. **Future**：表示"将来才有结果"的对象

```python
import asyncio

async def fetch(i):
    await asyncio.sleep(1)
    return i

async def main():
    results = await asyncio.gather(*[fetch(i) for i in range(10)])
```

上述 10 个协程并发执行，总耗时约 1 秒而非 10 秒。注意：协程中调用阻塞函数（如 `time.sleep`）会卡住整个事件循环，应改用 `asyncio.sleep` 或 `run_in_executor`。

## 上下文管理器的原理？

`with` 语句背后是上下文管理器协议：进入时调用 `__enter__`，退出时调用 `__exit__`（即使抛异常也会执行，适合释放资源）。

**两种实现方式**：
1. 类实现 `__enter__` / `__exit__` 方法
2. `contextlib.contextmanager` 装饰生成器函数，`yield` 前后分别是进入/退出逻辑

典型用途：文件句柄、数据库连接、锁、临时修改环境（pytest 的 monkeypatch 内部就是该模式）。

## 装饰器带参数怎么实现？

带参数的装饰器是「装饰器工厂」：外层函数接收参数，返回真正的装饰器。

```python
import functools

def retry(times):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if i == times - 1:
                        raise
        return wrapper
    return decorator
```

`functools.wraps` 用于保留被装饰函数的 `__name__` 和 `__doc__`，否则调试和文档工具会看到错误的函数元信息。

## 生成器和迭代器的区别？

迭代器实现 `__iter__` 和 `__next__`；生成器是迭代器的简洁写法——含 `yield` 的函数自动成为生成器。

**核心价值**：惰性求值（Lazy Evaluation），逐个产出元素而不必把全部数据载入内存。处理大文件时逐行读取的内存占用是 O(1) 而非 O(n)。

生成器表达式 `(x*x for x in range(10))` 与列表推导式语法相似，但返回生成器不立即计算。`yield` 还能双向通信：`send()` 可向生成器内部传值，这是协程的历史雏形。
