import multiprocessing
import numpy

import ai_util

def appender(res, current):
    res.append(current)
    return res

def extender(res, current):
    res.extend(current)
    return res

def adder(res, current):
    if len(res) == 0:
        res.append(0)
    res[0] += current
    return res

def example_target(begin_index, end_index, queue):
    print("example_target ({} {})".format(begin_index, end_index))
    res = 0
    for i in range(begin_index, end_index):
        res += 1
    print("example_target ({} {}) res={}".format(begin_index, end_index, res))
    queue.put(res)

def run(numThreads, lines, target, args, combiner=extender):
    run_timer = ai_util.Timer("MultiThread: Run (t={})".format(numThreads))
    run_timer.begin()
    threads = []
    results = []
    step = int(numpy.round(len(lines) / numThreads + 1))
    for i in range(numThreads):
        res = multiprocessing.Queue()  # Queue for thread i
        results.append(res)
        currentArgs = []
        currentArgs.append(i * step)  # start_index
        end_index = (i + 1) * step    # end_index
        end_index = min(end_index, len(lines))  # max end_index is len(lines)
        currentArgs.append(end_index)  # end
        for a in args:
            currentArgs.append(a)
        currentArgs.append(res)
        t1 = multiprocessing.Process(target=target, args=tuple(currentArgs))
        t1.start()
        threads.append(t1)
    result = []
    for i in range(numThreads):
        # print("my_thread {}".format(i))
        r = results[i].get()
        # print("my_thread {} res is {}".format(i, r))
        result = combiner(result, r)
    for i in range(numThreads):
        threads[i].join()
    run_timer.end()
    run_timer.report()
    return result
