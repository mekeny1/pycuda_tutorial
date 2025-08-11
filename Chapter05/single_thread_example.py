'''
Author: mekeny1
Date: 2025-06-13 01:51:59
LastEditors: mekeny1
LastEditTime: 2025-08-11 12:26:00
FilePath: \pycuda_tutorial_hapril\Chapter05\single_thread_example.py
Description: 使用Python多线程演示基本的线程创建、执行和同步机制，展示线程的生命周期管理和返回值传递，为理解多线程与CUDA结合编程奠定基础
Copyright (c) 2025 by mekeny1, All Rights Reserved. 
'''
import threading

"""
代码总体说明：
本程序演示了Python多线程的基本使用，主要特点包括：

1. 算法思想：
   - 创建自定义线程类，继承自Python的Thread类
   - 通过线程执行特定任务，实现并发处理
   - 演示线程的创建、启动、执行和同步的完整流程

2. 多线程编程策略：
   - 自定义线程类封装特定的业务逻辑
   - 重写run()方法定义线程的执行内容
   - 重写join()方法实现返回值的传递

3. 线程生命周期管理：
   - 线程创建：实例化自定义线程类
   - 线程启动：调用start()方法开始执行
   - 线程同步：通过join()等待线程完成并获取结果

4. 软硬件特性利用：
   - CPU：多线程并发执行，提高CPU利用率
   - 内存：线程间共享内存空间，支持数据交换
   - 操作系统：线程调度器管理线程的执行和切换
   - 并发控制：通过join()实现线程同步和结果收集
"""


class PointlessExampleThread(threading.Thread):
    """
    无意义示例线程类，继承自Python的Thread类

    这个类演示了：
    1. 如何创建自定义线程类
    2. 如何重写run()方法定义线程行为
    3. 如何重写join()方法传递返回值
    4. 线程的基本生命周期管理
    """

    def __init__(self):
        """
        初始化线程对象

        设置线程的基本属性，包括返回值存储变量
        """
        threading.Thread.__init__(self)  # 调用父类构造函数，初始化线程
        self.return_value = None  # 初始化返回值变量，用于存储线程执行结果

    def run(self):
        """
        线程的主要执行方法

        当线程启动后，会自动调用此方法执行线程任务
        这里演示了一个简单的打印操作和返回值设置
        """
        print("Hello from the thread you just spawned!")  # 打印线程执行信息
        self.return_value = 123  # 设置线程的返回值

    def join(self):
        """
        重写join方法，实现返回值的传递

        返回：
        - return_value: 线程执行完成后的返回值

        注意：这里重写了父类的join方法，在等待线程完成后返回自定义值
        """
        threading.Thread.join(self)  # 调用父类join方法，等待线程完成
        return self.return_value  # 返回线程执行的结果


# 创建线程实例
newThread = PointlessExampleThread()  # 实例化自定义线程类

# 启动线程执行
newThread.start()  # 调用start()方法启动线程，开始执行run()方法中的代码

# 等待线程完成并获取返回值
thread_output = newThread.join()  # 调用join()方法等待线程完成，并获取返回值

# 输出线程执行结果
print("The thread completed and returned this value: %s" %
      thread_output)  # 打印线程返回的值
