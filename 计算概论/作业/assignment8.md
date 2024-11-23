# Assignment #8: 田忌赛马来了

Updated 1021 GMT+8 Nov 12, 2024

2024 fall, Complied by <mark>李天笑 物理学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 12558: 岛屿周⻓

matices, http://cs101.openjudge.cn/practice/12558/ 

思路：

检测1的数量和相邻1的数量，经过简单计算即可得到岛屿周长。

代码：

```python
n, m = map(int,input().split())
land = []
for i in range(n):
    land.append(list(map(int, input().split())))
counter1 = 0
counter2 = 0
for i in range(n):
    for j in range(m):
        if land[i][j] == 1:
            counter1 += 1
            if i != 0:
                if land[i-1][j] == 1:
                    counter2 += 1
            if j != 0:
                if land[i][j-1] == 1:
                    counter2 += 1
ans = 4 * counter1 - 2 * counter2
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241112150615251](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241112150615251.png)



### LeetCode54.螺旋矩阵

matrice, https://leetcode.cn/problems/spiral-matrix/

与OJ这个题目一样的 18106: 螺旋矩阵，http://cs101.openjudge.cn/practice/18106

思路：

采用递归，将n-1矩阵按照一定规律复制到新矩阵，让后再填充新矩阵第一行和最后一列即可。

代码：

```python
def matrix(n):
    if n == 1:
        return [[1]]
    else:
        current_matrix = matrix(n-1)
        dp = [[0] * n for _ in range(n)]
        for i in range(n-1):
            for j in range(n-1):
                dp[n - 1 - i][n - 2 - j] = current_matrix[i][j] +\
                      (n ** 2 - (n - 1) ** 2)
        for i in range(n):
            dp[0][i] = i + 1
        for j in range(1,n):
            dp[j][n-1] = j + n
        return dp
n = int(input())
ans = matrix(n)
for row in ans:
    print(' '.join([str(e) for e in row]))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20241112152152104](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241112152152104.png)



### 04133:垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/

思路：

矩阵、动态规划

先初始化数组，将每个数据以及周围可以影响到的位置加上垃圾数量。然后遍历查找最大值以及其出现的次数，如果当前格子的值大于已知的最大值，则更新最大值并将计数器重置为 1（因为找到了一个新的最大值）；如果当前格子的值等于已知的最大值，则简单地增加计数器 `res` 的值。最后输出 `res` 和 `max_point`，即拥有最大值的格子数量及这个最大值。

### 总结

代码：

```python
d = int(input())
n = int(input())
dp = [[0]*1025 for _ in range(1025)]
for _ in range(n):
    x, y, k = map(int, input().split())
    for i in range(max(x-d, 0), min(x+d+1, 1025)):
        for j in range(max(y-d, 0), min(y+d+1, 1025)):
            dp[i][j] += k
res = max_point = 0
for i in range(0, 1025):
    for j in range(0, 1025):
        if dp[i][j] > max_point:
            max_point = dp[i][j]
            res = 1
        elif dp[i][j] == max_point:
            res += 1
print(res, max_point)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241112152236142](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241112152236142.png)



### LeetCode376.摆动序列

greedy, dp, https://leetcode.cn/problems/wiggle-subsequence/

与OJ这个题目一样的，26976:摆动序列, http://cs101.openjudge.cn/routine/26976/

思路：

动态规划，创建dp1和dp2,分别存储先升和先降的摆动数列，主次检验，如果上升，则更新dp1[i]的值为dp[i-1]和dp2[i-1]+1的较大值，下降和维持稳定同理，最后输出dp1和dp2中的极大值即可。

代码：

```python
n = int(input())
l = list(map(int, input().split()))
dp1 = [1] * n
dp2 = [1] * n
for i in range(1,n):
    if l[i] - l[i-1] > 0:
        dp1[i] = max(dp1[i-1], dp2[i-1] + 1)
    if l[i] - l[i-1] < 0:
        dp2[i] = max(dp2[i-1], dp1[i-1] + 1)
    if l[i] == l[i-1]:
        dp1[i] = dp1[i-1]
        dp2[i] = dp2[i-1]
#print(dp1)
#print(dp2)
a = max(dp1)
b = max(dp2)
print(max(a, b))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241112154834826](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241112154834826.png)



### CF455A: Boredom

dp, 1500, https://codeforces.com/contest/455/problem/A

思路：

动态规划

先创建字典，存储出现的数字和数字出现的次数。初始化列表dp=[0]*(N+1)，N为最大数字。遍历列表，如果i在字典中，更新dp[i] = max(dp[i-2] + i * d[i], dp[i-1])(及删掉这个数字和不删掉这个数字得分的极大值)；如果不在字典中，更新dp[i] = dp[i-1]。

最后输出dp[-1]即可。

代码：

```python
n = int(input())
sequence = list(map(int, input().split()))
#d={1:2,2:5,3:2}
#创建字典，存储每个数据出现的次数
d = {}
for i in sequence:
    if i not in d:
        d[i] = 1
    else:
        d[i] += 1
#print(d)
N = max(sequence)
dp = [0] * (N + 1)
for i in range(1, N + 1):
    if i in d:
        dp[i] = max(dp[i-2] + i * d[i], dp[i-1])
    if i not in d:
        dp[i] = dp[i-1]
print(dp[-1])
#print(dp)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241112155510304](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241112155510304.png)



### 02287: Tian Ji -- The Horse Racing

greedy, dfs http://cs101.openjudge.cn/practice/02287

思路：

贪心，具体见注释

代码：

```python
def max_value(n, v1, v2):
    score = 0
    v1.sort()
    v2.sort()
    #双指针
    i, l = 0, 0
    j, m = n-1, n-1
    while i <= j:
        #如果田忌速度慢的大于齐王速度慢的，直接比赛
        if v1[i] > v2[l]:
            score += 200
            i += 1
            l += 1
        #如果田忌速度快大于齐王速度快的，直接比赛
        elif v1[j] > v2[m]:
            score += 200
            j -= 1
            m -= 1
        #如果前两种情况都不满足
        else:
            #如果田忌速度慢的小于齐王速度快的，比赛输分
            if v1[i] < v2[m]:
                score -= 200
            i += 1
            m -= 1
    return score
while True:
    n = int(input())
    if n == 0:
        break
    else:
        v1 = list(map(int, input().split()))
        v2 = list(map(int, input().split()))
        ans = max_value(n, v1, v2)
        print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241112172310130](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241112172310130.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

期中考试结束！（虽然今天考试考破防了😅）

感觉这次作业前面的dp、矩阵、递归题目自己掌握的还是不错，基本都是独立写出来的，最后一题贪心对于平局处理不到位，参考了题解。

每日选做目前落了两天进度，这周追平，此外要加大投入计算概论的时间了



