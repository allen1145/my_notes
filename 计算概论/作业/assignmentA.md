# Assignment #10: dp & bfs

Updated 2 GMT+8 Nov 25, 2024

2024 fall, Complied by <mark>李天笑、物理学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### LuoguP1255 数楼梯

dp, bfs, https://www.luogu.com.cn/problem/P1255

思路：

dp，每一步为前两步方案之和

代码：

```python
n = int(input())
dp = [1] * 5000
dp[1] = 2
for i in range(2, 5000):
    dp[i] = dp[i - 1] + dp[i - 2]
print(dp[n - 1])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241126144704204](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241126144704204.png)



### 27528: 跳台阶

dp, http://cs101.openjudge.cn/practice/27528/

思路：

dfs做的

代码：

```python
n = int(input())
dx = [i for i in range(1, n + 1)]

def in_bound(x, n):
    return 0 <= x < n + 1

def dfs(x, n):
    if x == n:
        return 1
    
    ans = 0
    for i in dx:
        next_x = x + i
        if in_bound(next_x, n):
            ans += dfs(next_x, n)
    return ans

ans = dfs(0, n)
print(ans)
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20241126145749215](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241126145749215.png)



### 474D. Flowers

dp, https://codeforces.com/problemset/problem/474/D

思路：

动态规划，求解每个数量的方式，配合prefix_sum求解区间和，最后输出答案

代码：

```python
MOD = 1000000007

t, k = map(int, input().split())

dp = [0] * (100001)
dp[0] = 1 

for i in range(1, 100001):
    dp[i] = dp[i - 1] 
    if i >= k:
        dp[i] = (dp[i] + dp[i - k]) % MOD 

prefix_sum = [0] * (100002)
for i in range(1, 100002):
    prefix_sum[i] = (prefix_sum[i - 1] + dp[i - 1]) % MOD

results = []
for _ in range(t):
    a, b = map(int, input().split())
    result = (prefix_sum[b + 1] - prefix_sum[a]) % MOD
    results.append(result)

for result in results:
    print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241126233418977](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241126233418977.png)



### LeetCode5.最长回文子串

dp, two pointers, string, https://leetcode.cn/problems/longest-palindromic-substring/

思路：

纯暴力，没想到AC了

代码：

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        ans = ''
        length = 0
        def check(s):
            n = len(s)
            i, j = 0, n - 1
            while i < j:
                if s[i] == s[j]:
                    i += 1
                    j -= 1
                else:
                    return False
            return True
        n = len(s)
        if n == 1:
            return s
        for i in range(n):
            for j in range(i+1, n + 1):
                cur_s = s[i:j]
                if cur_s == cur_s[::-1] and len(cur_s) > length:
                    ans = cur_s
                    length = len(cur_s)
        return ans
    ans = longestPalindrome
    print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241126224753839](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241126224753839.png)





### 12029: 水淹七军

bfs, dfs, http://cs101.openjudge.cn/practice/12029/

思路：

bfs就可以

逆天数据输入😡改了半天发现不是bfs的问题，是读取数据的问题

代码：

```python
import sys
sys.setrecursionlimit(300000)
input = sys.stdin.read

from collections import deque

dx = [1, -1, 0, 0]
dy = [0, 0, 1, -1]

def in_bound(x, y, M, N):
    return 0 <= x < M and 0 <= y < N

def bfs(matrix, M, N, positions, x_c, y_c):
    visited = [[False] * N for _ in range(M)]
    queue = deque([])
    
    for x_i, y_i, heigh in positions:
        if x_i == x_c and y_i == y_c:
            return True
        visited[x_i][y_i] = True
        queue.append((x_i, y_i, heigh))
    
    while queue:
        cur_x, cur_y, heigh = queue.popleft()
        for i in range(4):
            next_x = cur_x + dx[i]
            next_y = cur_y + dy[i]
            
            if in_bound(next_x, next_y, M, N) and not visited[next_x][next_y] and matrix[next_x][next_y] < heigh:
                if next_x == x_c and next_y == y_c:
                    return True
                visited[next_x][next_y] = True
                queue.append((next_x, next_y, heigh))
    
    return visited[x_c][y_c]

# 读取所有输入
data = input().split()
index = 0

ans = []
K = int(data[index])
index += 1

for i in range(K):
    M, N = map(int, data[index:index + 2])
    index += 2
    matrix = []
    for _ in range(M):
        matrix.append(list(map(int, data[index:index + N])))
        index += N    
    x_c, y_c = map(int, data[index:index + 2])
    index += 2
    P = int(data[index])
    index += 1
    positions = []
    for j in range(P):
        x_i, y_i = map(int, data[index: index + 2])
        positions.append((x_i - 1, y_i - 1, matrix[x_i - 1][y_i - 1]))
        index += 2
    
    if bfs(matrix, M, N, positions, x_c - 1, y_c - 1):
        ans.append('Yes')
    else:
        ans.append('No')

for answer in ans:
    print(answer)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241126155612476](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241126155612476.png)



### 02802: 小游戏

bfs, http://cs101.openjudge.cn/practice/02802/

思路：

bfs，优先保证走直线

bfs搜索可行路线，通过pre回溯重现路径，然后检测路径拐点。

为什么自己写的如此长😅

代码：

```python
from collections import deque

dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def in_bound(x, y, h, w):
    return 0 <= x < h + 2 and 0 <= y < w + 2

def bfs(matrix, h, w, x_1, y_1, x_2, y_2):
    if abs(x_1 - x_2) + abs(y_1 - y_2) == 1:
        return 1
    queue = deque([(x_1, y_1)])
    visited = [[False] * (w + 2) for _ in range(h + 2)]
    pre = [[None] * (w + 2)  for _ in range(h + 2)]
    visited[x_1][y_1] = True
    matrix[x_2][y_2] = ' '
    while queue:

        cur_x, cur_y = queue.popleft()

        if cur_x == x_2 and cur_y == y_2:
            break 

        for i in range(4):
            next_x = cur_x + dx[i]
            next_y = cur_y + dy[i]

            if in_bound(next_x, next_y, h, w) and not visited[next_x][next_y] and matrix[next_x][next_y] == ' ':
                visited[next_x][next_y] = True
                pre[next_x][next_y] = (cur_x, cur_y)
                queue.append((next_x, next_y))
    matrix[x_2][y_2] = 'X'
    
    #回溯路径
    path = []
    current = (x_2, y_2)
    while current:
        path.append(current)
        current = pre[current[0]][current[1]]
    
    steps = len(path)

    if steps == 1:
        return -1

    #寻找拐点

    if 1 < steps < 3:
        return 1
    turning_points = 0

    for i in range(1, steps - 1):
        p1 = path[i - 1]
        p2 = path[i]
        p3 = path[i + 1]

        vec1 = (p2[0] - p1[0], p2[1] - p1[1])
        vec2 = (p3[0] - p2[0], p3[1] - p2[1])

        if vec1 != (0, 0) and vec2 != (0, 0):
            if vec1[0] * vec2[0] + vec1[1] * vec2[1] == 0:
                turning_points += 1
    

    return turning_points + 1
p = 1
while True:
    w, h = map(int, input().split())  
    if w == 0 and h == 0:  
        break 

    matrix = [[' '] * (w + 2)]  
    for i in range(h):  
        row = list(input()) 
        matrix.append([' '] + row + [' '])
    matrix.append([' '] * (w + 2))
    #for _ in matrix:
    #    print(_)
    print('Board #' + str(p) + ':')
    j = 1
    while True:
        x_1, y_1, x_2, y_2 = map(int,input().split())
        if x_1 == 0 and y_1 == 0 and x_2 == 0 and y_2 == 0:
            break
        ans = bfs(matrix, h, w, y_1, x_1, y_2, x_2)
        if ans == -1:
            print('Pair ' + str(j) + ': impossible.')
        else:
            print('Pair ' + str(j) + ': '+ str(ans) + ' segments.')
        j += 1
    
    print()
    p += 1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241126204248660](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241126204248660.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

正常跟上每日选做，做了sy dfs、bfs 的题

最近ddl较多😅，争取额外练习一些题，另外，好像该学习笔试了（悲



