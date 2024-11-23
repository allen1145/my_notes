# Assignment #4: T-primes + 贪心

Updated 0337 GMT+8 Oct 15, 2024

2024 fall, Complied by <mark>李天笑 物理学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 34B. Sale

greedy, sorting, 900, https://codeforces.com/problemset/problem/34/B



思路：遍历列表，如果小于零，就加入到新列表，新列表内为可以赚钱的电视；对于新列表，如果长度小于m，全卖掉，如果列表长度大于m，卖掉前m个。



代码

```python
n,m=map(int,input().split())
data=list(map(int,input().split()))
data_1=sorted(data)
buy=[]
for i in data_1:
    if i<0:
        buy.append(i)
if len(buy)<=m:
    a=-sum(buy)
    print(a)
if len(buy)>m:
    print(-sum(buy[:m]))

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![34B. Sale](D:\物院\大一秋季学期\计算概论\课程作业\第六周作业\34B. Sale.png)



### 160A. Twins

greedy, sortings, 900, https://codeforces.com/problemset/problem/160/A

思路：将硬币存为列表并按照逆序排序，遍历i从0到n，找到i，使得0-i个硬币和大于后面所有硬币和，输出i



代码

```python
n=int(input())
data=list(map(int,input().split()))
data_1=sorted(data,reverse=True)
for i in range(n+1):
    if sum(data_1[:i])>sum(data_1[i:]):
        a=i
        break
print(a)
```



代码运行截图 ==（至少包含有"Accepted"）==

![160A. Twins](D:\物院\大一秋季学期\计算概论\课程作业\第六周作业\160A. Twins.png)



### 1879B. Chips on the Board

constructive algorithms, greedy, 900, https://codeforces.com/problemset/problem/1879/B

思路：数学思路：有一行或者一列一定会占据所有数字，再另外一列或者一行，取最小数字即可



代码

```python
t=int(input())
n=[]
a=[]
b=[]
for i in range(t):
    n.append(int(input()))
    a.append(list(map(int,input().split())))
    b.append(list(map(int,input().split())))
for j in range(t):
    n_j=n[j]
    a_j=a[j]
    b_j=b[j]
    x_1=sum(a_j)+n_j*min(b_j)
    x_2=sum(b_j)+n_j*min(a_j)
    print(min(x_1,x_2))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![1879B. Chips on the Board](D:\物院\大一秋季学期\计算概论\课程作业\第六周作业\1879B. Chips on the Board.png)



### 158B. Taxi

*special problem, greedy, implementation, 1100, https://codeforces.com/problemset/problem/158/B

思路：动态规划：

生成长度为5的列表，索引为i的位置更新为输入中人数为i的组数。

出租车初始数量设置为4人组数量；三人组数量也要乘坐单独的出租车，每辆车剩下的位置可以放置1人组，更新1人组的数量为0或者减去三人组后的数量；二人组两组一车，为需增加车数量，再分析是否会剩下2人组以及剩下1人组的分配问题。

代码

```python
def min_taxis_needed(n, groups):
    count = [0] * 5  
    for size in groups:
        count[size] += 1
    taxis = count[4]
    taxis += count[3]
    count[1] = max(0, count[1] - count[3]) 
    taxis += count[2] // 2 
    if count[2] % 2 == 1:  
        taxis += 1  
        count[1] = max(0, count[1] - 2)  
    taxis += (count[1] + 3) // 4  
    return taxis
n = int(input())
groups = list(map(int, input().split()))
print(min_taxis_needed(n, groups))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![158B. Taxi](D:\物院\大一秋季学期\计算概论\课程作业\第六周作业\158B. Taxi.png)



### *230B. T-primes（选做）

binary search, implementation, math, number theory, 1300, http://codeforces.com/problemset/problem/230/B

思路：利用欧拉筛法确定输入最大范围内所有质数，并返回这些数，转化为集合set。对于输入每个数据，先判断是否为完全平方数或者1，如果不是完全平方数或者是1，直接输出NO；否则判断开方后是否为质数，如果是输出YES，不是输出NO。



代码

```python
import math
 
def sieve_of_eratosthenes(limit):
    primes = [True] * (limit + 1)
    primes[0] = primes[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if primes[i]:
            for j in range(i*i, limit + 1, i):
                primes[j] = False
    return [i for i in range(limit + 1) if primes[i]]
 
n = int(input())
l = list(map(int, input().split()))
 
primes_set = set(sieve_of_eratosthenes(int(math.sqrt(max(l)))))
 
for i in l:
    a = math.sqrt(i)
    if a - int(a) != 0 or i == 1:
        print('NO')
    else:
        if int(a) in primes_set:
            print('YES')
        else:
            print('NO')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![230B. T-primes](D:\物院\大一秋季学期\计算概论\课程作业\第六周作业\230B. T-primes.png)



### *12559: 最大最小整数 （选做）

greedy, strings, sortings, http://cs101.openjudge.cn/practice/12559

思路：定义函数描述对列表中字符串的排序方式，按照此排序方式将列表中元素排序并用join（）函数分别输出即可。



代码

```python
def fonction1(a,b):
    if a + b == b + a:
        return 0
    if a + b > b + a:
        return -1
    if a + b < b + a:
        return 1
def fonction2(a,b):
    if a + b == b + a:
        return 0
    if a + b > b + a:
        return 1
    if a + b < b + a:
        return -1
n = int(input())
l = list(input().split())
from functools import cmp_to_key  
l.sort(key = cmp_to_key(fonction1))
a = ''.join(l) 
l2 = l.sort(key = cmp_to_key(fonction2))
b = ''.join(l) 
print(a + ' ' + b)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![12559 最大最小整数](D:\物院\大一秋季学期\计算概论\课程作业\第六周作业\12559 最大最小整数.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

1.学会了eluer_sieve的方法寻找质数，可以显著减小代码的复杂度。

2.利用from functools import cup_to_key,可以将函数作为排序依据对于列表内数据排序。

3.继续跟进度完成每日选做，截至目前已完成了所有每日选做题目。

4.又抽空完成了一些sy的算法题目。



