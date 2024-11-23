# Assignment #9: dfs, bfs, & dp

Updated 2107 GMT+8 Nov 19, 2024

2024 fall, Complied by <mark>李天笑、物理学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 18160: 最大连通域面积

dfs similar, http://cs101.openjudge.cn/practice/18160

思路：dfs，检测'W'的位置，如果检测到了，就对这一位置标记为'.'并对其周围调用dfs函数进行深度搜索，同时记录面积，最后返面积值，存储到列表中。输出面积最大值即可。



代码：

```python
#维护，防止越界
def in_bound(x, y, n, m):
    return 0 <= x < n and 0 <= y < m

#每次向四周移动1格
dx = [1, -1, 0, 0, 1, 1, -1, -1]
dy = [0 ,0, 1, -1, 1, -1, 1, -1]

def dfs(x, y, n ,m , matrix):

    if not in_bound(x, y, n, m) or matrix[x][y] != 'W':
        return 0

    matrix[x][y] = '.'
    value = 1

    for i in range(8):
        next_x = x + dx[i]
        next_y = y + dy[i]
        if in_bound(next_x, next_y, n, m) and matrix[next_x][next_y] == 'W':
            value += dfs(next_x, next_y, n, m, matrix)
            
    return value

#主程序
T = int(input())
ans = []
for i in range(T):
    n, m = map(int,input().split())
    matrix = [list(input()) for _ in range(n)]
    values = []

    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 'W':
                values.append(dfs(i, j, n, m, matrix))
    if values:
        ans.append(max(values))
    else:
        ans.append(0)
for i in ans:
    print(i)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241119230248078](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241119230248078.png)



### 19930: 寻宝

bfs, http://cs101.openjudge.cn/practice/19930

思路：

（没看见bfs，用dfs写的）

代码：

```python
#维护，防止越界
def in_bound(x, y, n, m):
    return 0 <= x < n and 0 <= y < m

#可移动范围
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

#dfs函数
def dfs(x, y, n, m, map, visited, result, min_result):
    #如果遇到2，返回'inf'
    if map[x][y] == 2:
        return float('inf')
	
    #如果遇到1，且result小于当前min_result,更新min_result并且将其返回
    if map[x][y] == 1:
        if result < min_result:
            min_result = result
        return min_result
    
    for i in range(4):
        next_x = x + dx[i]
        next_y = y + dy[i]
		
        #递归调用以及回溯
        if in_bound(next_x, next_y, n, m) and not visited[next_x][next_y] and map[next_x][next_y] != 2:
            result += 1
            visited[next_x][next_y] = True
            min_result = dfs(next_x, next_y, n, m, map, visited, result, min_result)
            visited[next_x][next_y] = False
            result -= 1
    return min_result

#主程序
n, m = map(int,input().split())
map = [list(map(int,input().split())) for _ in range(n)]
visited = [[False] * m for _ in range(n)]
visited[0][0] = True
result = 0
min_result = float('inf')
min_result = dfs(0, 0, n, m, map, visited, result, min_result)
if min_result == float('inf'):
    print('NO')
else:
    print(min_result)
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20241119234130348](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241119234130348.png)



### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123

思路：

dfs，对于给定起点，尝试每一种移动方式能否满足要求

代码：

```python
#可移动方式
dx = [1, -1, -1, 1, 2, 2, -2, -2]
dy = [2, 2, -2, -2, 1, -1, -1, 1]

#维护，防止越界
def in_bound(x, y, n, m):
    return 0 <= x < n and 0 <= y < m

#dfs函数
def dfs(x, y, n, m, visited, count):
    #如果走过了每一区域，返回一种方式
    if count == n * m:
        return 1
    
    #可行方式数目，初值设为0
    ans = 0
    
    #对于每种方式递归调用并回溯
    for i in range(8):
        next_x = x + dx[i]
        next_y = y + dy[i]
        if in_bound(next_x, next_y, n, m) and not visited[next_x][next_y]:
            visited[next_x][next_y] = True
            count += 1
            ans += dfs(next_x, next_y, n, m, visited, count)
            count -=1
            visited[next_x][next_y] = False
    return ans

#主程序
T = int(input())
answers = []
for i in range(T):
    n, m, x, y = map(int,input().split())
    visited = [[False] * m for _ in range(n)]
    visited[x][y] = True
    count = 1
    ans = dfs(x, y, n, m, visited, count)
    answers.append(ans)
for answer in answers:
    print(answer)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241119235929674](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241119235929674.png)



### sy316: 矩阵最大权值路径

dfs, https://sunnywhy.com/sfbj/8/1/316

思路：

dfs，深度搜索每一种路径，同时记录权值，返回最大的即可

代码：

```python
#可移动方式
dx = [1, -1, 0, 0]
dy = [0, 0, 1, -1]

#维护，防止越界
def in_bound(x, y, n, m):
    return 0 <= x < n and 0 <= y < m

#定义dfs函数
def dfs(x, y, n, m, visited, matrix, path, value, last_path, last_value):
    
    #如果走到终点，且value大于last_value，更新last_value以及last_path
    if x == n - 1 and y == m - 1:
        if value > last_value or last_value == 0:
            last_value = value
            last_path = path[:]
        return last_path, last_value
    
    #递归调用与回溯
    for i in range(4):
        next_x = x + dx[i]
        next_y = y + dy[i]
        if in_bound(next_x, next_y, n, m) and not visited[next_x][next_y]:
            value += matrix[next_x][next_y]
            path.append([next_x, next_y])
            visited[next_x][next_y] = True
            last_path, last_value = dfs(next_x, next_y, n, m, visited, matrix, path, value, last_path, last_value)
            value -= matrix[next_x][next_y]
            path.pop()
            visited[next_x][next_y] = False
    return last_path, last_value

#主函数
n, m = map(int,input().split())
visited = [[False] * m for _ in range(n)]
visited[0][0] = True
matrix = []
for i in range(n):
    matrix.append(list(map(int,input().split())))
#初始化
value, last_value = matrix[0][0], float('-inf')
path, last_path = [[0, 0]], []
last_path, last_value = dfs(0, 0, n, m, visited, matrix, path, value, last_path, last_value)
for i in last_path:
    print(' '.join([str(e + 1) for e in i]))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241119231103321](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241119231103321.png)





### LeetCode62.不同路径

dp, https://leetcode.cn/problems/unique-paths/

思路：

动态规划，到达每一个格子的方式是到达其左侧格子的方式加上到达其上方格子的方式和。

代码：

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * n for _ in range(m)]
        for  i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]
if __name__ == '__main__':
    sol = Solution()
    print(sol.uniquePaths)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241119231025540](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241119231025540.png)



### sy358: 受到祝福的平方

dfs, dp, https://sunnywhy.com/sfbj/8/3/539

思路：

dfs，先定义一个集合，存储所有小$10^9$的平方数。然后将数据按照位数生成列表，便于提取数据。定义dfs函数，将数字列表从高位开始分割，对于每次分割，检查前面的数字是否为平方数，以及后面的数字是否可以继续分割为平方数，如果可以，返回True，不可以，返回False。

代码：

```python
squares = set()
i = 1
while i ** 2 < 10 ** 9:
    squares.add(i ** 2)
    i += 1
def dfs(idx):
    if idx == len(digits):
        return True

    num = 0
    for i in range(idx, len(digits)):
        num = num * 10 + digits[i]
        if num in squares:
            if dfs(i + 1):
                return True
    return False

A = int(input())
digits = list(map(int, str(A)))
if dfs(0):
    print('Yes')
else:
    print('No')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241120102900941](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241120102900941.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

感觉自己dfs的模板大概会套用了，bfs掌握情况一般，还需要做题练习

为什么感觉期中考完后事情更多了😡（好几篇论文ddl，以及开放性实验）

目前跟上了每日选做，做了一小点LeetCode题目和sy题目。



