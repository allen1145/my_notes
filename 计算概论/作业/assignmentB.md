# Assignment #B: Dec Mock Exam大雪前一天

Updated 1649 GMT+8 Dec 5, 2024

2024 fall, Complied by <mark>李天笑、物理学院</mark>



**说明：**

1）⽉考： AC<mark>1（破防）</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### E22548: 机智的股民老张

http://cs101.openjudge.cn/practice/22548/

思路：

dp

代码：

```python
l = list(map(int, input().split()))
n = len(l)
dp = [[0] * 2 for _ in range(n)]
dp[0][0] = l[0]
for i in range(1, n):
    dp[i][0] = min(dp[i - 1][0], l[i])
    dp[i][1] = max(dp[i - 1][1], l[i] - dp[i][0])
print(dp[-1][1])
    
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241205183913713](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241205183913713.png)



### M28701: 炸鸡排

greedy, http://cs101.openjudge.cn/practice/28701/

思路：

尽量让时间长的鸡排单独炸，如果时间长的鸡排所用时间大于平均时间，则最后一个鸡排必然不能被炸熟

那么将最后一个鸡排放到从头炸到尾。此时变为同时炸前面n-1给鸡排的问题。如果如果时间长的鸡排所用时间小于平均时间，必然可以合理分配使得所有鸡排炸好。此时最长时间就是平均时间。

代码：

```python
n, k = map(int, input().split())
t = list(map(int, input().split()))
t.sort()
s = sum(t)
while True:
    if t[-1] > s / k:
        s -= t.pop()
        k -= 1
    else:
        print(f'{s / k:.3f}')
        break
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20241205202647631](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241205202647631.png)



### M20744: 土豪购物

dp, http://cs101.openjudge.cn/practice/20744/

思路：

想不到一点，选择求助题解了（好nb的思路）

dp1储存**以第 i 个商品结尾的连续子数组最大和**，不考虑放回商品

`dp2[i]` 表示 **以第 i 个商品结尾的连续子数组最大和**，允许放回其中一个商品。

- 状态转移公式：`dp2[i] = max(dp1[i - 1], dp2[i - 1] + a[i], a[i])`
- 解释：
  - `dp1[i - 1]` 表示选择前面的子数组，但不加当前商品。
  - `dp2[i - 1] + a[i]` 表示当前商品加入到之前可能已经放回一个商品的子数组。
  - `a[i]` 表示单独选择当前商品

代码：

```python
a = list(map(int, input().split(',')))
dp1 = [0] * len(a)
dp2 = [0] * len(a)
dp1[0] = a[0]
dp2[0] = a[0]
for i in range(1, len(a)):
    dp1[i] = max(dp1[i - 1] + a[i], a[i])
    dp2[i] = max(dp1[i - 1], dp2[i - 1] + a[i], a[i])
print(max(dp2))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241205200913496](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241205200913496.png)



### T25561: 2022决战双十一

brute force, dfs, http://cs101.openjudge.cn/practice/25561/

思路：

暴力，dfs，搜索时可以做一个剪枝，提高效率

好麻烦，但感觉比前面dp，greedy简单

代码：

```python

n, m = map(int, input().split())
values = []
discount = []
for i in range(n):
    a = list(input().split())
    for j in range(len(a)):
        num, value = map(int, a[j].split(':'))
        a[j] = (num, value)
    values.append(a)
for i in range(m):
    b = list(input().split())
    for j in range(len(b)):
        money, dis = map(int, b[j].split('-'))
        b[j] = (money, dis)
    discount.append(b)
#print(values)
#print(discount)
sum_spend = [0] * (m + 1)
bought = [False] * n
visited = set()
def pay(sum_spend, discount):
    ans = 0
    for i in range(1, m + 1):
        cur_ans = sum_spend[i]
        for money, dis in discount[i - 1]:
            if sum_spend[i] >= money:
                cur_ans  = min(cur_ans, sum_spend[i] - dis)
        ans += cur_ans
    return ans

def dfs(values, bought, sum_spend, min_ans):
    state = tuple(bought) + tuple(sum_spend)
    if state in visited:
        return
    visited.add(state)

    if all(bought):
        min_ans[0] = min(min_ans[0], pay(sum_spend, discount) - (sum(sum_spend) // 300) * 50)
        return 
    
    for i in range(n):
        if not bought[i]:
            for num, value in values[i]:
                sum_spend[num] += value
                bought[i] = True
                dfs(values, bought, sum_spend, min_ans)
                bought[i] = False
                sum_spend[num] -= value

min_ans = [float('inf')]
dfs(values, bought, sum_spend, min_ans)
print(min_ans[0])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241205232712793](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241205232712793.png)



### T20741: 两座孤岛最短距离

dfs, bfs, http://cs101.openjudge.cn/practice/20741/

思路：

dfs

md考试时状态不对，递归那里把nx， ny写成了x， y还没查出来😡

代码：

```python
n = int(input())
matrix = [list(map(int, input().strip())) for _ in range(n)]
m = len(matrix[0])
dx = [1, -1, 0, 0]
dy = [0, 0, 1, -1]
def dfs(matrix, n, m, x, y, position):
    matrix[x][y] = 0
    position.append((x, y))
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if 0 <= nx < n and 0 <= ny < m and matrix[nx][ny] == 1:
            dfs(matrix, n, m, nx, ny, position)
    return
position=[]

for i in range(n):
    for j in range(m):
        if matrix[i][j] == 1:
            position1 = []
            dfs(matrix, n, m, i, j, position1)
            position.append(position1)
ans = float('inf')
for x1, y1 in position[0]:
    for x2, y2 in position[1]:
        ans = min(ans, abs(x1 - x2) + abs(y1 - y2))
print(ans - 1)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241205232730575](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241205232730575.png)



### T28776: 国王游戏

greedy, http://cs101.openjudge.cn/practice/28776

思路：

奇怪的排序方式

代码：

```python
n=int(input())
king_left, king_right = map(int, input().split())
values = []
for i in range(n):
    x, y = map(int, input().split())
    values.append((x, y))
values.sort(key = lambda x:(x[0] * x[1], x[0], x[1]))
cur = king_left
ans = king_left // values[0][1]
for i in range(1,n):
    cur *= values[i-1][0]
    ans = max(ans, cur // values[i][1])
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241205235459713](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241205235459713.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

感觉贪心题和dp题思路不好想出来，感觉甚至比后面的tough题目还难。希望期末机考可以出简单一点，要不然就裂开了啊！

目前跟着每日选做



