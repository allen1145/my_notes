# Assignment #5: Greedy穷举Implementation

Updated 1939 GMT+8 Oct 21, 2024

2024 fall, Complied by <mark>李天笑 物理学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 04148: 生理周期

brute force, http://cs101.openjudge.cn/practice/04148

思路：创建列表data，利用while循环读取数据；定义函数findday，遍历1，21253找到与开始天差值为三个周期整数倍的数字，返回结束天，判断正负并输出；针对每组数据运行函数findday并按要求输出即可。



代码：

```python
data = []
while True:
    a = list(map(int,input().split()))
    if a == [-1,-1,-1,-1]:
        break
    else:
        data.append(a)
def findday(l):
    a = 0
    for i in range(1,21253):
        if (l[0] - i)%23 == 0 and\
              (l[1] - i)%28 == 0 and\
                  (l[2] - i)%33 == 0:
            a = i-l[3]
            if a <= 0:
                a += 21252
            break
    return a
for i in range(len(data)):
    o = findday(data[i])
    print('Case ' + str(i+1)+ ': the next triple peak occurs in ' + str(o) + ' days.')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![04148 生理周期](D:\物院\大一秋季学期\计算概论\课程作业\assignment5\04148 生理周期.png)



### 18211: 军备竞赛

greedy, two pointers, http://cs101.openjudge.cn/practice/18211

思路：贪心思路：如果当前剩余资金可以买最便宜的武器，直接购买，如果不足以买最便宜的武器，则将最贵的武器卖掉；重复操作，并将每一步拥有武器数量存为一个列表，最后输出最大值。



代码：

```python
n = int(input())
l = list(map(int,input().split()))
l.sort()
i, j = 0, len(l) - 1
count = 0
o = [0]
while i <= j and count >= 0:
    if n >= l[i]:
        n -= l[i]
        i += 1
        count += 1
    else:
        n += l[j]
        j -= 1
        count -= 1
    o.append(count)
    
print(max(o))
```



代码运行截图 ==（至少包含有"Accepted"）==

![18211 军备竞赛](D:\物院\大一秋季学期\计算概论\课程作业\assignment5\18211 军备竞赛.png)



### 21554: 排队做实验

greedy, http://cs101.openjudge.cn/practice/21554

思路：显然的贪心思路是：按照实验时长排序即可使得平均等待时长最短。用列表处理即可达到题目要求。



代码：

```python
n = int(input())
l = list(map(int,input().split()))
a = []
for i in range(n):
    a.append([l[i],i+1])
a.sort(key = lambda x:x[0])
o = [a[i][1] for i in range(n)]
o = [str(e) for e in o]
l.sort()
s = 0
for i in range(n):
    s += l[i]*(n-1-i)
sum_time = s/n
print(' '.join(o))
print('%.2f'%sum_time)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![21554 排队做实验](D:\物院\大一秋季学期\计算概论\课程作业\assignment5\21554 排队做实验.png)



### 01008: Maya Calendar

implementation, http://cs101.openjudge.cn/practice/01008/

思路：按照题目要求建立两个日历间的关系即可

注意不要输错单词，看清输出要求（本人由于多打印了1个空格检查了近30min，发现后又由于没有输出n检查了10min）（悲



代码：

```python
n = int(input())
o = []
d1 = {
    'pop':1,'no':2,'zip':3,'zotz':4,'tzec':5,'xul':6,'yoxkin':7,
     'mol':8,'chen':9,'yax':10,'zac':11,'ceh':12,'mac':13,
     'kankin':14,'muan':15,'pax':16,'koyab':17,'cumhu':18,'uayet':19
      }
d2 = {
     1:'imix',2:'ik',3:'akbal',4:'kan',5:'chicchan',6:'cimi',
     7:'manik',8:'lamat',9:'muluk',10:'ok',11:'chuen',12:'eb',
     13:'ben',14:'ix',15:'mem',16:'cib',17:'caban',18:'eznab',
     19:'canac',0:'ahau' 
}
d3 = {1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,0:13}

for _ in range(n):
    a, b, c = input().split()
    a = a[:-1]
    days = int(c)*365+(d1[b]-1)*20+int(a)+1
    if days%260 == 0:
        year = days//260 - 1
    else:
        year = days//260
    days_l = days%260
    num = d3[days_l%13]
    mon = d2[days_l%20]
    output = str(num) + ' ' + mon + ' ' + str(year)
    o.append(output)
print(n)
for i in o:
    print(i)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![01008 Maya Calendar](D:\物院\大一秋季学期\计算概论\课程作业\assignment5\01008 Maya Calendar.png)



### 545C. Woodcutters

dp, greedy, 1500, https://codeforces.com/problemset/problem/545/C

思路：按照树的位置排序，首先第一棵和最后一棵树一定可以砍掉；然后对第二棵树到倒数第二棵树遍历，能向右倒就向右，不能就向左，还不行就跳过，并且每次都要更新可以倒塌的范围。



代码：

```python
n = int(input())
data = []
for i in range(n):
    data.append(list(map(int,input().split())))
if n == 1:
    print(1)
else:
    x = 2    
    data.sort(key = lambda x:x[0])
    i = 1
    down = data[0][0]
    up = 0
    while i < n-1:
        up = data[i+1][0]
        if data[i][0] - data[i][1] > down:
            down = data[i][0]
            x += 1
            i += 1
            continue
        if data[i][0] + data[i][1] < up:
            x += 1
            down = data[i][0] + data[i][1]
            i += 1
            continue
        down = data[i][0]
        i += 1
    print(x)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![545C. Woodcutters](D:\物院\大一秋季学期\计算概论\课程作业\assignment5\545C. Woodcutters.png)



### 01328: Radar Installation

greedy, http://cs101.openjudge.cn/practice/01328/

思路：将岛屿按照$x$坐标从小到达排序，从左到右遍历，从左到右在保证能扫描到左侧岛屿时尽可能使得雷达靠右。如果能扫描到下一个岛屿i+1，ans不变，如果不能，ans + 1，更新雷达所能扫描到右侧的最大坐标。



代码：

```python
import math
o = []
while True:
    n, d = map(int,input().split())
    if n == 0 and d == 0:
        break
    else:
        l = []
        ans = 1
        for i in range(n):
            l.append(list(map(int,input().split())))
        input()
        l.sort(key = lambda x:(x[0],-x[1]))
        if l[0][1] > d:
            ans = -1
            o.append(ans)
            continue
        else:
            up_x = math.sqrt(d**2-l[0][1]**2) + l[0][0] + d
            i = 1
            while i < n:
                x, y = l[i][0], l[i][1]
                if y > d:
                    ans = -1
                    break
                else:
                    if x > up_x:
                        up_x = math.sqrt(d**2-y**2) + x + d
                        ans += 1
                    else:
                        up_y = math.sqrt(d**2-(x-up_x+d)**2)
                        if y > up_y and x > up_x -d:
                            up_x = math.sqrt(d**2-y**2) + x + d
                            ans += 1
                        if y > up_y and x < up_x -d:
                            up_x = math.sqrt(d**2-y**2) + x + d
                i += 1
        o.append(ans)
for i in range(len(o)):
    a = i+1
    print('Case ' + str(a) + ': ' + str(o[i]))

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![01328Radar Installation](D:\物院\大一秋季学期\计算概论\课程作业\assignment5\01328Radar Installation.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

感觉作业难度确实提升了，不过一般情况下还是可以不借助AI和题解把题目AC掉，虽然可能花费时间稍长并且要尝试好几遍（悲 ，感觉对greedy题目有一定的理解和思路了。

由于在复习一些科目的期中考试，导致无太多时间学习py，仅勉强跟上了每日选做（今天的还没写



