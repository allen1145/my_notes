# Assignment #D: 十全十美 

Updated 1254 GMT+8 Dec 17, 2024

2024 fall, Complied by <mark>李天笑、物院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 02692: 假币问题

brute force, http://cs101.openjudge.cn/practice/02692

思路：

很久以前的每日选做了，甚至差点看不懂自己写的代码（）

初始化dp，k1，然后模拟三次称重。

（1）如果两边一样重，记录相应的硬币dp[i] = 10，表示不可能为假币

（2）如果两边不一样重，则对权重不为10的硬币操作，重的一边‘+1’，轻的一边‘-1’。同时k1 +1。

如果有偶数次天平倾斜，则寻找dp中数量不为10， 0且模2为0的硬币，此即为假币

如果有奇数次天平倾斜，则寻找dp中模2为1的硬币，此即为假币

通过判断假币dp中权重的正负来判断假币是轻了还是重了

代码：

```python
n = int(input())
d = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8
     ,'J':9,'K':10,'L':11}
d1 = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I'
     ,9:'J',10:'K',11:'L'}
for i in range(n):
    k1 = 0
    step = []
    dp = [0]*12
    for i in range(3):
        step.append(input().split())
    for i in range(3):
        l, r, s = step[i][0], step[i][1], step[i][2]
        if s == 'even':
            for i in l+r:
                dp[d[i]] = 10
        if s == 'up':
            k1 += 1
            for i in l:
                if dp[d[i]] != 10:
                    dp[d[i]] -= 1
            for i in r:
                if dp[d[i]] != 10:
                    dp[d[i]] += 1
        if s == 'down':
            k1 += 1
            for i in l:
                if dp[d[i]] != 10:
                    dp[d[i]] += 1
            for i in r:
                if dp[d[i]] != 10:
                    dp[d[i]] -= 1
    #if -1 in dp:
    #    p = dp.index(-1)
    #    x = d1[p]
    #    weight = 'heavy.'
    #if -2 in dp:
    #    p = dp.index(-2)
    #    x = d1[p]
    #    weight = 'light.'
    if k1 % 2 == 0 :
        for i in dp:
            if i != 10 and abs(i) % 2 == 0 and i != 0:
                p = dp.index(i)
                x = d1[p]
                if i > 0:
                    weight = 'light.'
                else:
                    weight = 'heavy.'
                    break
    if k1 % 2 == 1:
        for i in dp:
            if i != 10 and abs(i) % 2 == 1 and i != 0:
                p = dp.index(i)
                x = d1[p]
                if i > 0:
                    weight = 'light.'
                else:
                    weight = 'heavy.'
                    break
                
    print(x + ' is the counterfeit coin and it is ' + weight)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241217144348314](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241217144348314.png)



### 01088: 滑雪

dp, dfs similar, http://cs101.openjudge.cn/practice/01088

思路：

dfs，从每个点开始向四周搜索，入果可以到达，max_length更新为max(max_length, 1 + dfs(nx, ny))

要用lru_cache才能过

代码：

```python
from functools import lru_cache

R, C = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(R)]
dx = [1, -1, 0, 0]
dy = [0, 0, 1, -1]

@lru_cache(None)
def dfs(x, y):
    max_length = 1
    
    for i in range(4):
        nx, ny = x + dx[i], y + dy[i]
        
        if 0 <= nx < R and 0 <= ny < C and matrix[x][y] > matrix[nx][ny]:
            max_length = max(max_length, 1 + dfs(nx, ny))
    
    return max_length

answer = 0
for i in range(R):
    for j in range(C):
        answer = max(answer, dfs(i, j))

print(answer)
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20241217160732434](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241217160732434.png)



### 25572: 螃蟹采蘑菇

bfs, dfs, http://cs101.openjudge.cn/practice/25572/

思路：

按照套路打一遍bfs模板即可

代码：

```python
from collections import deque

dx = [1, -1, 0, 0]
dy = [0, 0, 1, -1]

def in_bound(x1, y1, x2, y2, n):
    if 0 <= x1 < n and 0 <= y1 < n and 0 <= x2 < n and 0 <= y2 < n:
        return True
    return False

def have_visitied(x1, y1, x2, y2, visited):
    if visited[x1][y1] and visited[x2][y2]:
        return False
    return True

def find_target(matrix, target, n):
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == target:
                matrix[i][j] = 0
                return i, j

def bfs(x1, y1, x2, y2, matrix, n):

    queue = deque([(x1, y1, x2, y2)])
    visitied = [[False] * n for _ in range(n)]
    visitied[x1][y1] = True
    visitied[x2][y2] = True

    while queue:
        cur_x1, cur_y1, cur_x2, cur_y2 = queue.popleft()
        
        for i in range(4):
            nx1 = cur_x1 + dx[i]
            ny1 = cur_y1 + dy[i]
            nx2 = cur_x2 + dx[i]
            ny2 = cur_y2 + dy[i]

            if in_bound(nx1, ny1, nx2, ny2, n) and have_visitied(nx1, ny1, nx2, ny2, visitied) and (matrix[nx1][ny1] != 1 and matrix[nx2][ny2] != 1): 
                if matrix[nx1][ny1] == 9 or matrix[nx2][ny2] == 9:
                    return True

                queue.append((nx1, ny1, nx2, ny2))
                visitied[nx1][ny1] = True
                visitied[nx2][ny2] = True

    return False

n = int(input())
matrix = [list(map(int, input().split())) for _ in range(n)]
x1, y1 = find_target(matrix, 5, n)
x2, y2 = find_target(matrix, 5, n)
#print(x1, y1, x2, y2)
if bfs(x1, y1, x2, y2, matrix, n):
    print('yes')
else:
    print('no')

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241217152240058](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241217152240058.png)



### 27373: 最大整数

dp, http://cs101.openjudge.cn/practice/27373/

思路：

最大整数算法（cmp_to_key）加上不完全背包问题（0—1背包）

代码：

```python
from functools import cmp_to_key

def compare(a, b):
    if str(a) + str(b) < str(b) + str(a):
        return 1
    if str(a) + str(b) == str(b) + str(a):
        return 0
    if str(a) + str(b) > str(b) + str(a):
        return -1

m = int(input())
n = int(input())
num = list(map(int, input().split()))
num.sort(key = cmp_to_key(compare))
dp = [0] * (m + 1)
for i in range(1, n + 1):
    for j in range(m, 0, -1):
        if len(str(num[i - 1])) <= j:
            dp[j] = max(dp[j], int(str(dp[j - len(str(num[i - 1]))]) + str(num[i - 1])))
print(dp[-1])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241217143133310](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241217143133310.png)



### 02811: 熄灯问题

brute force, http://cs101.openjudge.cn/practice/02811

思路：

如果先对2——5排开关操作，使得前4排灯熄灭，会发现最后一排灯不好操作，且此时还没有对第一排开关操作过。不妨考虑先对第一排开关操作，改变初始条件，之后对后面四排开关操作，使得前四排灯熄灭。可证明，总有一种初始条件下，第五排灯恰好熄灭。我们要做的就是遍历所有可能的初始条件。

这里可以考虑生成（0——63）的二进制数来表示64种可能的初始条件，python内置bin（）函数即可实现。

代码：

```python
matrix = [list(map(int, input().split())) for _ in range(5)]
def pass_botton(x, y, matrix, ans):
    ans[x][y] = 1 - ans[x][y]
    dx = [1, -1, 0, 0, 0]
    dy = [0, 0, 1, -1, 0]
    for i in range(5):
        nx = x + dx[i]
        ny = y + dy[i]
        if 0 <= nx < 5 and 0 <= ny < 6:
            matrix[nx][ny] = 1 - matrix[nx][ny]
          
def solution(matrix):
    for i1 in range(64):
        s = bin(i1)[2:]
        l = [0] * 6
        for i2 in range(1, len(s) + 1):
            l[-i2] = int(s[-i2])
        cur_ans = [[0] * 6 for _ in range(5)]
        cur_matrix = [row[:]for row in matrix]
        for i in range(6):
            if l[i] == 1:
                pass_botton(0, i, cur_matrix, cur_ans)
        for i in range(4):
            for j in range(6):
                if cur_matrix[i][j] == 1:
                    pass_botton(i + 1, j, cur_matrix, cur_ans)
        if cur_matrix[4] == [0] * 6:
            return cur_ans
    
ans = solution(matrix)
for i in ans:
    print(' '.join([str(e) for e in i]))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241217143638956](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241217143638956.png)



### 08210: 河中跳房子

binary search, greedy, http://cs101.openjudge.cn/practice/08210/

思路：

二分查找

先定义一个函数，判断在答案为min_distance下需要移动走的石头数量，如果敲好为M，则找到了正确的min_distance，否则没有找到。

然后进行二分查找搜索答案，定义low = 1, heigh = L。为最小距离的最小和最大的可能答案。对min = (low + heigh) // 2检测。如果恰好可以，就是答案将其输出，否则继续二分查找。

代码：

```python
def can_remove_stones(stones, min_distance, M):
    removed = 0
    last_position = stones[0]
    for i in range(1, len(stones)):
        if stones[i] - last_position < min_distance:
            removed += 1
            if removed > M:
                return False
        else:
            last_position = stones[i]
    return True

L, N, M = map(int, input().split())
stones = [0] + [int(input()) for _ in range(N)] + [L]

low, high = 1, L
result = 0
while low <= high:
    mid = (low + high) // 2
    if can_remove_stones(stones, mid, M):
        result = mid
        low = mid + 1
    else:
        high = mid - 1

print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241217144252796](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241217144252796.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

每日选做做完了，感觉有的题目对于贪心策略要求挺高的，不然容易tle

打算再看一看leetcode的题目，机考加油（



