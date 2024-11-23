# Assignment #6: Recursion and DP

Updated 2201 GMT+8 Oct 29, 2024

2024 fall, Complied by <mark>李天笑 物理学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### sy119: 汉诺塔

recursion, https://sunnywhy.com/sfbj/4/3/119  

思路：

递归，讲前n-1从A挪到B，将最后一层从A挪到C，再将前n-1层从B挪到C。

代码：

```python
moves = []
def F(n,A,B,C,moves):
    if n == 1:
        moves.append(A+'->'+C)
    else:
        F(n-1,A,C,B,moves)
        moves.append(A+'->'+C)
        F(n-1,B,A,C,moves)
n = int(input())
F(n,'A','B','C',moves)
m = len(moves)
print(m)
for move in moves:
    print(move)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

<img src="C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241029224432589.png" alt="image-20241029224432589" style="zoom: 50%;" />



### sy132: 全排列I

recursion, https://sunnywhy.com/sfbj/4/3/132

思路：

递归，对于n-1全排列，在每种排列的每个位置加上n即可

代码：

```python
def A(n):
    if n == 1:
        return ['1']
    else:
        s = A(n-1)
        result = []
        for i in range(len(s)):
            for j in range(n):
                o = s[i][:j] + str(n) +s[i][j:]
                result.append(o)
        return result
n = int(input())
result = A(n)
result.sort()
for i in result:
    print(' '.join(i))

        
```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241029231524285.png" alt="image-20241029231524285" style="zoom: 50%;" />



### 02945: 拦截导弹 

dp, http://cs101.openjudge.cn/2024fallroutine/02945

思路：

dp，建立长度为n的列表，进行双层循环，找到当前步骤可以达到最大的数量

代码：

```python
n = int(input())
l = list(map(int,input().split()))
dp = [1]*n
for i in range(1,n):
    for j in range(i):
        if l[i] <= l[j]:
            dp[i] = max(dp[i],dp[j] + 1)
print(max(dp))
```



代码运行截图 

![image-20241029224136853](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241029224136853.png)

### 23421: 小偷背包 

dp, http://cs101.openjudge.cn/practice/23421

思路：

dp，建立长度为（B+1）的列表（B为总重量），进行双层循环。对于每一件物品，如果当前物体的重量小于背包剩余可容纳质量，就更新为容纳此物品前的价值加上此物品的价值的和当前价值的最大值。

代码：

```python
N, B = map(int,input().split())
value = list(map(int,input().split()))
weight = list(map(int,input().split()))
dp = [0] * (B + 1)
for i in range(1, N + 1):
    for j in range(B, 0, -1):
        if weight[i-1] <=  j:
            dp[j] = max(dp[j],dp[j - weight[i-1]] + value[i-1])
print(dp[B])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241029224211341](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241029224211341.png)



### 02754: 八皇后

dfs and similar, http://cs101.openjudge.cn/practice/02754

思路：

递归，从第一行开始，尝试在每一列放置皇后。对于每一列，检查是否与之前放置的皇后冲突，如果没用冲突就继续递归；如果冲突就回到上一步，尝试其它列。如果row = n + 1，则结束，找到一种方式，将方式添加到o列表中。最后按要求输出答案即可。（自己想不出来/(ㄒoㄒ)/~~） 

代码：

```python
def A(row, n, position, check, count, o):
    if row == n + 1:
        count[0] += 1
        o.append(position[1:])
        return
    for col in range(1, n + 1):
        if not check[col]:
            s = True
            for pre_row in range(1, row):
                if abs(row - pre_row) == abs(col - position[pre_row]):
                    s = False
                    break
            if s:
                position[row] = col
                check[col] = True
                A(row + 1, n, position, check, count, o)
                check[col] = False
n = 8
position = [0] * (n + 1)
check = [False] * (n + 1)
count = [0]
o = []
A(1, n, position, check, count, o)
answers = []
for i in o:
    i = [str(e) for e in i]
    answers.append(''.join(i))
    answers.sort()
answer = []
k = int(input())
for i in range(k):
    answer.append(answers[int(input())-1])
for i in answer:
    print(i)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241030123147850](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241030123147850.png)



### 189A. Cut Ribbon 

brute force, dp 1300 https://codeforces.com/problemset/problem/189/A

思路：

一开始没看懂题（

后来看了答案代码

对于每一种长度，循环三种裁剪结果的长度，如果当前长度大于等于所裁剪的，则更新当前数量为裁剪前的数量加上1和当前数量的最大值。

代码：

```python
n, a, b, c = map(int,input().split())
dp = [0] + [float('-inf')] * n
 
for i in range(1,n + 1):
    for j in (a, b, c):
        if i >= j:
            dp[i] = max(dp[i-j] + 1, dp[i])
print(dp[n])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241029224328097](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241029224328097.png)

## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>



熟悉了简单的dp题目思路和递归题目思路，但是一些复杂的递归和dp还是想不出来（悲

最近期中复习任务有点多，只能跟上每日选做和作业了，额外练习较少

