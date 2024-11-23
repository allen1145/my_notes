[TOC]



## 1.题目

#### 263A. Beautiful Matrix

##### 思路：

①创建列表，表示矩阵。②查找矩阵中元素1的位置。③输出1与矩阵中心的距离

##### 代码：

```python
matrix = []
for i in range(5):
    matrix.append(list(map(int,input().split())))
a = None
for i in range(5):
    if 1 in matrix[i]:
        a = i
        break
b = matrix[a].index(1)
n = abs(2-a) + abs(2-b)
print(n)
```

##### 截图：

![263A. Beautiful Matrix](D:\物院\大一秋季学期\计算概论\课程作业\第三周作业\263A. Beautiful Matrix.png)

#### 1328A. Divisibility Problem

##### 思路：

①计算$b$对$a$的模$c$。②分类，输出0或者$b-c$

##### 代码：

```python
n=int(input())
l = []
for i in range(n):
    l.append(list(map(int,input().split())))
for i in l:
    a = i[0]
    b = i[1]
    if a % b != 0:
        m = b - (a % b)
        print(m) 
    else:
        print(0)
```



##### 截图：

![1328A. Divisibility Problem](D:\物院\大一秋季学期\计算概论\课程作业\第三周作业\1328A. Divisibility Problem.png)

#### 427A. Police Recruits

##### 思路：

$a$、$b$分别记录警员数和犯罪事件数，直接顺次计算

##### 代码：

```python
n=int(input())
events=list(map(int,input().split()))
a=0
b=0
for i in range(n):
    if events[i]>0:
        a+=events[i]
    if events[i]<0:
        if a>0:
            a-=1
        else:
            b+=1
print(b)
```

##### 截图：

![427A. Police Recruits](D:\物院\大一秋季学期\计算概论\课程作业\第三周作业\427A. Police Recruits.png)

#### 02808: 校门外的树

##### 思路：

①创建列表代表每一棵树。②遍历列表，如果在修建地铁范围内就加到count列表。③计算剩余树的数量

##### 代码：

```python
L,M=map(int,input().split())
areas=[]
trees=list(range(L+1))
count=[]
for i in range(M):
    areas.append(list(map(int,input().split())))
for tree in trees:
    for area in areas:
        if area[0]<=tree<=area[1]:
            count.append(tree)
            break
print(L+1-len(count))
```



##### 截图：

![02808 校门外的树](D:\物院\大一秋季学期\计算概论\课程作业\第三周作业\02808 校门外的树.png)

#### sy60: 水仙花数II

##### 思路：

遍历整个区间，找到每个水仙花数

##### 代码：

```python
a,b = map(int,input().split())
op=[]
for i in range(a,b+1):
    a = int(str(i)[0])
    b = int(str(i)[1])
    c = int(str(i)[2])
    if i == a**3 + b**3 + c**3:
        op.append(str(i))
if len(op) !=0:
    print(' '.join(op))
else:
    print("NO")
```



##### 截图:

##### ![sy60 水仙花数Ⅱ](D:\物院\大一秋季学期\计算概论\课程作业\第三周作业\sy60 水仙花数Ⅱ.png)：

#### 01922: Ride to School

##### 思路：直接计算每个人到达的时刻，选取出发时间为非负的人中最先到达的时间即可

##### 代码：

```python
import math
while True:
    n=int(input())
    if n == 0:
        break
    else:
        p = []
        for i in range(n):
            p.append(list(map(int,input().split())))
        time = []
        for i in p:
            v = i[0]
            t = i[1]
            T = math.ceil((4500 * 36) / (10 * v) + t)
            if t >= 0:
                time.append(T)
        print(min(time))
```



##### 截图：

![01922 Ride to School](D:\物院\大一秋季学期\计算概论\课程作业\第三周作业\01922 Ride to School.png)

## 2. 学习总结和收获

（1）继续跟着进度完成每日选做。

（2）从第三章开始学习《算法笔记》，并刷了三四章书上部分习题。

（3）更加进一步地了解了python语言的语法，能够熟练运用基本语法与内置函数。
