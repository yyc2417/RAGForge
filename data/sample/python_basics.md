# Python 编程基础

## 列表推导式

列表推导式（List Comprehension）是 Python 中创建列表的简洁语法。基本格式为 `[表达式 for 变量 in 可迭代对象]`。

示例：
```python
# 创建平方数列表
squares = [x**2 for x in range(10)]
# 结果: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 带条件过滤
even_squares = [x**2 for x in range(10) if x % 2 == 0]
# 结果: [0, 4, 16, 36, 64]
```

列表推导式比等价的 for 循环更快，因为它在 C 层面优化了迭代。

## 装饰器

装饰器（Decorator）是 Python 中修改函数行为的函数。使用 `@decorator` 语法糖。

```python
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 耗时: {end - start:.4f}秒")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)
    return "完成"
```

装饰器本质上是高阶函数：接收一个函数，返回一个新函数。常见的内置装饰器包括 `@staticmethod`、`@classmethod`、`@property`。

## 生成器

生成器（Generator）是一种惰性求值的迭代器，使用 `yield` 关键字。

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 取前10个斐波那契数
fib = fibonacci()
first_10 = [next(fib) for _ in range(10)]
# 结果: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

生成器的优势是内存效率：不需要一次性把所有元素加载到内存，适合处理大数据流。

## 上下文管理器

上下文管理器使用 `with` 语句，确保资源正确释放。

```python
# 文件操作（自动关闭文件）
with open("data.txt", "r") as f:
    content = f.read()

# 自定义上下文管理器
from contextlib import contextmanager

@contextmanager
def database_connection(url):
    conn = create_connection(url)
    try:
        yield conn
    finally:
        conn.close()
```

上下文管理器的核心是 `__enter__` 和 `__exit__` 方法。`__exit__` 保证即使发生异常也会执行清理操作。

## GIL（全局解释器锁）

GIL（Global Interpreter Lock）是 CPython 中的互斥锁，同一时刻只有一个线程执行 Python 字节码。

这意味着：
- **CPU 密集型任务**：多线程不会提速，应该用多进程（`multiprocessing`）
- **I/O 密集型任务**：多线程仍然有效，因为 I/O 操作会释放 GIL
- **异步编程**：`asyncio` 在单线程内实现并发，适合 I/O 密集场景

Python 3.13 引入了实验性的 free-threaded 模式（无 GIL），未来可能彻底移除 GIL。
