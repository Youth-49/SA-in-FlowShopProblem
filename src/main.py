import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import re
import time
from functools import wraps

random.seed(7)

def timefn(fn):
    """[calculate running time of a function]

    Returns:
        [type]: [description]
    """
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s")
        w.write(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s \n")
        return result
    return measure_time


def Initial_State(n, m):
    """[generate initial state randomly for Non-Permutation FSP]

    Returns:
        [2D-np.array]: [initial state]
    """
    state = np.zeros((n, m))
    for i in range(n):
        state[i] = np.arange(m)
        np.random.shuffle(state[i])
    return state

def Initial_State_Nolags(n, m): 
    """[generate initial state randomly for Permutation FSP]

    Returns:
        [2D-np.array]: [initial state]
    """
    state = np.zeros((n, m))
    for i in range(n):
        state[i] = np.arange(m)
    return state

def Random_Forward(state, n, m):
    """[generate neighborhood by Forward method for Non-Permutation FSP]

    Returns:
        [2D-np.array]: [a neiborhood]
    """
    ret = state.copy()
    while(1):
        row, i, j = random.randint(
            0, n-1), random.randint(0, m-1), random.randint(0, m-1)
        if(i != j):
            break
    if i > j:
        i, j = j, i

    tmp_arr = np.zeros((1, j-i))
    tmp_arr = np.copy(state[row][i:j])
    tmp = state[row][j]
    ret[row][i+1:j+1] = tmp_arr
    ret[row][i] = tmp
    return ret


def Random_Forward_Nolags(state, n, m):
    """[generate neighborhood by Forward method (no lags) for Permutation FSP]

    Returns:
        [2D-np.array]: [a neiborhood]
    """
    ret = state.copy()
    while(1):
        i, j = random.randint(0, m-1), random.randint(0, m-1)
        if(i != j):
            break
    if i > j:
        i, j = j, i

    for k in range(n):
        tmp_arr = np.zeros((1, j-i))
        tmp_arr = np.copy(state[k][i:j])
        tmp = state[k][j]
        ret[k][i+1:j+1] = tmp_arr
        ret[k][i] = tmp
    return ret


def Value(state, process_time, n, m, flag):
    """[calculate cost time for processing all workpieces]

    Args:
        flag ([int]): [flag == 1 returns time table, otherwise returns cost time]

    Returns:
        [2D-np.array/int]: [depends on flag value]
    """
    time = np.zeros((n, m))
    time[0][0] = process_time[int(state[0][0])][0]
    for j in range(1, m):
        time[0][j] = time[0][j-1] + process_time[int(state[0][j])][0]
    for i in range(1, n):
        for j in range(m):
            ls = state.tolist()
            idx = ls[i-1].index(state[i][j])
            if j == 0:
                time[i][j] = time[i-1][idx] + process_time[int(state[i][j])][i]
            else:
                time[i][j] = max(time[i-1][idx], time[i][j-1]) + \
                    process_time[int(state[i][j])][i]
    if flag == 1:
        return time
    else:
        return time[n-1][m-1]


def Initial_T_exponential(process_time, n, m, a):
    """[generate temperature declining process]

    Returns:
        [1D-np.array]: [temperature declining array]
    """
    ret = []
    sum = process_time.sum()/(5*n*m)
    ret.append(sum)  # initial temperature
    b = 1 - 2*a/(n*(n-1))  # declining rate
    cur = sum
    while(1):
        if cur < 1e-3:
            break
        cur = cur*b
        ret.append(cur)
    return ret


def Random_choice(possibility):
    return random.random() < possibility


@timefn
def Simulated_Annealing(n, m, schedule, process_time):
    """[Simulated Annealing process]

    Args:
        n ([int]): [number of workpieces]
        m ([int]): [number of machines]
        schedule ([1D-np.array]): [temperature declining array]
        process_time ([2D-np.array]): [process time of each workpiece on each machine]

    Returns:
        [2D-np.array, int, list, list, list]: [solution, cost time, iterations, current_cost history, next_cost history]
    """
    T = 0
    y = []
    x = []
    z = []
    current_cost = 0
    next_cost = 0
    del_E = 0
    step = 0
    i = 0
    reject = 0
    current_state = Initial_State_Nolags(m, n)
    while(1):
        T = schedule[step]
        step = step+1
        if T <= 1e-3: 
            return current_state, current_cost, x, y, z
        for _ in range(n):
            i = i+1
            next_state = Random_Forward_Nolags(current_state, m, n)
            current_cost = Value(current_state, process_time, m, n, 0)
            next_cost = Value(next_state, process_time, m, n, 0)
            del_E = -(next_cost - current_cost)
            z.append(next_cost)
            y.append(current_cost)
            x.append(i)
            reject = reject + 1
            if del_E >= 0:
                current_state = next_state
                reject = 0
                break
            elif Random_choice(np.exp(del_E/T)):
                current_state = next_state

            if reject >= 2000 :
                return current_state, current_cost, x, y, z


def Remove_Empty(s):
    while '' in s:
        s.remove('')


def Read_data(read_filename):
    with open(read_filename, 'r') as f:
        content = f.read()

    tmplist = re.split(' |\n', content)
    Remove_Empty(tmplist)
    n, m = int(tmplist[0]), int(tmplist[1])
    tmparr = np.array(tmplist[2:])
    process_time = tmparr.reshape((n, m*2))
    process_time = np.delete(
        process_time, [i for i in range(0, m*2, 2)], axis=1)
    process_time = process_time.astype(np.int64)
    return process_time, n, m


def Show_Result(write_filename, x, y, z, cost, sol):
    plt.figure()
    plt.title('time-iteration')
    plt.xlabel('round of iteration')
    plt.ylabel('cost time')
    plt.plot(x, y, color='red', linewidth=3, label='current time')
    plt.plot(x, z, color='royalblue', label='neighborhood time')
    plt.legend()
    result.append(cost)
    with open(write_filename, 'w') as g:
        g.write(str(cost))
        g.write('\n')
        g.write(str(sol))
        g.write('\n')
    plt.savefig('./figure/result' + str(i) + '.png')


if __name__ == '__main__':
    with open('result.txt', 'a+') as w:
        result = []
        for i in range(11):
            read_filename = './data' + str(i) + '.txt'
            write_filename = './result' + str(i) + '.txt'

            process_time, n, m = Read_data(read_filename)
            a = 0.1
            schedule = Initial_T_exponential(process_time, n, m, a)
            sol, cost, x, y, z = Simulated_Annealing(
                n, m, schedule, process_time)

            Show_Result(write_filename, x, y, z, cost, sol)
            w.write(str(i)+': ')
            w.write(str(result))
            w.write('\n')