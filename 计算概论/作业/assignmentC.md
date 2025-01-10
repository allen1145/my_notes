# Assignment #C: 五味杂陈 

Updated 1148 GMT+8 Dec 10, 2024

2024 fall, Complied by <mark>李天笑、物理学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 1115. 取石子游戏

dfs, https://www.acwing.com/problem/content/description/1117/

思路：

假设a>b，如果[a // b] >= 2或是a%b == 0，那么先手必胜，否则只有一种取法，找到下一组数后，重复上面操作，判断能否得到 [a // b] >= 2或是a%b == 0。

代码：

```python
from collections import deque

def can_win(a, b):
    if a == b:
        return True
    if a // b >= 2:
        return True
    
    queue = deque([(a, b, 0)])
    #visited = set()

    while queue:
        x, y, step = queue.popleft()
        #if (x, y) in visited:
        #    continue
        #visited.add((x, y))

        if x % y == 0:
            return step % 2 == 0
        if x // y >= 2 and x % y != 0:
            return step % 2 == 0

        queue.append((max(y, x - y), min(y, x - y), step + 1))

while True:
    a, b = map(int, input().split())
    if a == 0 and b == 0:
        break
    if can_win(max(a, b), min(a, b)):
        print('win')
    else:
        print('lose')

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241210131933844](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241210131933844.png)



### 25570: 洋葱

Matrices, http://cs101.openjudge.cn/practice/25570

思路：

逐层检索找到最大即可

代码：

```python
n = int(input())
matrix = [list(map(int, input().split())) for _ in range(n)]
ans = float('-inf')
for i in range((n + 1) // 2):
    cur_ans = 0
    if i != n - 1 - i:
        cur_ans += sum(matrix[i][i: n - i])
        cur_ans += sum(matrix[n - 1 - i][i: n - i])
        for j in range(i + 1, n - i - 1):
            cur_ans += matrix[j][n - i - 1]
            cur_ans += matrix[j][i]
    if i == n - 1 - i:
        cur_ans += sum(matrix[i][i: n - i])
        
        
    ans = max(cur_ans, ans)
print(ans)
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20241210132025958](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241210132025958.png)



### 1526C1. Potions(Easy Version)

greedy, dp, data structures, brute force, *1500, https://codeforces.com/problemset/problem/1526/C1

思路：

gread，如果能喝就喝，不能喝就看是否可以吐出来最小的并喝。可以利用最小堆来实现。

PS.（C1用列表min也能过，但是C2只能用最小堆过了）

代码：

```python
import heapq

n = int(input())
potions = list(map(int, input().split()))
health_value = 0
drunk = []
for potion in potions:
    if health_value + potion >= 0:
        heapq.heappush(drunk, potion)
        health_value += potion
    elif len(drunk) != 0 and drunk[0] < potion:
        health_value -= heapq.heappop(drunk)
        heapq.heappush(drunk, potion)
        health_value += potion
print(len(drunk))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241210133456235](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241210133456235.png)



### 22067: 快速堆猪

辅助栈，http://cs101.openjudge.cn/practice/22067/

思路：

创建两个栈，一个存储猪重量，另一个存储最小的猪重量。

如果输入'pop'，就将两个列表中最后一个元素弹出

如果输入'min'，就输出min_stack[-1]

如果输入'push n'，就将n加入stack中，并且如果n小于min_stack[-1]，将n加入min_stack，否则将min_stack[-1]加入min_stack

代码：

```python
import sys

stack = []
min_stack = []

input = sys.stdin.read
data = input().strip().splitlines()

for s in data:
    if s == 'pop':
        if stack:
            stack.pop()
            min_stack.pop()
    elif s == 'min':
        if stack:
            print(min_stack[-1])
    else:
        command, value = s.split()
        if command == 'push':
            weight = int(value)
            stack.append(weight)
            if not min_stack or weight <= min_stack[-1]:
                min_stack.append(weight)
            else:
                min_stack.append(min_stack[-1])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241210134418920](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241210134418920.png)



### 20106: 走山路

Dijkstra, http://cs101.openjudge.cn/practice/20106/

思路：

利用最小堆实现Dijkstra算法。思路见注释

代码：

```python
import heapq

def dijkstra(grid, start, end):
    m, n = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
	
    #如果起点或终点是'#'，无法到达
    if grid[start[0]][start[1]] == '#' or grid[end[0]][end[1]] == '#':
        return 'NO'
	
    #记录到每一点的最小体力
    dist = [[float('inf')] * n for _ in range(m)]
    dist[start[0]][start[1]] = 0
	
    #初始化
    pq = [(0, start[0], start[1])]

    while pq:
        current_dist, u_row, u_col = heapq.heappop(pq)
		
        #如果到这一点的体力大于最小值，跳过
        if current_dist > dist[u_row][u_col]:
            continue
		
        #如果到终点，输出体力值
        if (u_row, u_col) == end:
            return current_dist
        
        #遍历每种走法
        for dr, dc in directions:
            v_row, v_col = u_row + dr, u_col + dc

            if 0 <= v_row < m and 0 <= v_col < n and grid[v_row][v_col] != '#':
                new_dist = current_dist + abs(int(grid[u_row][u_col]) - int(grid[v_row][v_col]))
				#如果体力值小于已知体力值，更改列表，并添加到堆里
                if new_dist < dist[v_row][v_col]:
                    dist[v_row][v_col] = new_dist
                    heapq.heappush(pq, (new_dist, v_row, v_col))
            
    return 'NO'
    
m, n, p = map(int, input().split())
grid = [list(input().split()) for _ in range(m)]
answers = []
for i in range(p):
    x, y, x_c, y_c = map(int, input().split())
    ans = dijkstra(grid, (x, y), (x_c, y_c))
    answers.append(ans)
for answer in answers:
    print(answer)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241210134956357](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241210134956357.png)



### 04129: 变换的迷宫

bfs, http://cs101.openjudge.cn/practice/04129/

思路：

bfs，visited数组开到三维，多一个记录时间的维数即可

代码：

```python
from collections import deque

dx = [1, -1, 0, 0]
dy = [0, 0, 1, -1]

def in_bound(x, y, R, C):
    return 0 <= x < R and 0 <= y < C

def can_go(x, y, time, matrix, K):
    if (time + 1) % K == 0 or matrix[x][y] != '#':
        return True
    return False

def find_target(matrix, target, R, C):
    for i in range(R):
        for j in range(C):
            if matrix[i][j] == target:
                return i, j 

def bfs(x, y, x_c, y_c, R, C, K, matrix):
    visited = [[[False] * K for _ in range(C)] for i in range(R)]
    queue = deque([(x, y, 0)])
    visited[x][y][0] = True

    while queue:
        cur_x, cur_y, time = queue.popleft()
        for i in range(4):
            next_x = cur_x + dx[i]
            next_y = cur_y + dy[i]
            if in_bound(next_x, next_y, R, C) and not visited[next_x][next_y][(time + 1) % K] and can_go(next_x, next_y, time, matrix, K):
                if next_x == x_c and next_y == y_c:
                    return time + 1
                visited[next_x][next_y][(time + 1) % K] = True
                queue.append((next_x, next_y, time + 1))
    return "Oop!"

T = int(input())
answers = []
for i in range(T):
    R, C, K = map(int, input().split())
    matrix = [list(input()) for _ in range(R)]
    x, y = find_target(matrix, 'S', R, C)
    x_c, y_c = find_target(matrix, 'E', R, C)
    ans = bfs(x, y, x_c, y_c, R, C, K, matrix)
    answers.append(ans)
    #print(x, y, x_c, y_c)
for answer in answers:
    print(answer)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241210135656076](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241210135656076.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

全是每日选做题，直接就交了（

每日选做做到了12.11的

每周一求：希望机考不要太难，希望笔试不要太难



