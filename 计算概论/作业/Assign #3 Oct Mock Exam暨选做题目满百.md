# Assign #3: Oct Mock Exam暨选做题目满百

Updated 1537 GMT+8 Oct 10, 2024

2024 fall, Complied by Hongfei Yan==（请改为同学的姓名、院系）==



**说明：**

1）Oct⽉考： AC5。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。

2）请把每个题目解题思路（可选），源码Python, 或者C++/C（已经在Codeforces/Openjudge上AC），截图（包含Accepted, 学号），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、作业评论有md或者doc。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### E28674:《黑神话：悟空》之加密

http://cs101.openjudge.cn/practice/28674/



思路：遍历,将每个字母对应的字母加到新的字符串中，最后输出



代码

```python
n = int(input())
s = input()
l1 = 'abcdefghijklmnopqrstuvwxyz'
l2 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
a = len(s)
o = ''
for i in range(a):
    if s[i] in l1:
        b = l1.find(s[i])
        c = (b-n) % 26
        o += l1[c]
    if s[i] in l2:
        b = l2.find(s[i])
        c = (b-n) % 26
        o += l2[c]
print(o)
```



代码运行截图 ==（至少包含有"Accepted"）==

![89da71f20f758f898766b138d74540d](C:\Users\ltx18\Documents\WeChat Files\wxid_6o3y3jf8tgi122\FileStorage\Temp\89da71f20f758f898766b138d74540d.png)



### E28691: 字符串中的整数求和

http://cs101.openjudge.cn/practice/28691/



思路：提取字符串中数字，转化为int，直接求和



代码

```python
l = list(input().split())
s1 = l[0]
s2 = l[1]
s11 = s1[0:2]
s22 = s2[0:2]
o = int(s11) + int(s22)
print(o)
```



代码运行截图 ==（至少包含有"Accepted"）==

![28691](D:\物院\大一秋季学期\计算概论\课程作业\第五周作业（10月月考）\28691.png)



### M28664: 验证身份证号

http://cs101.openjudge.cn/practice/28664/



思路：按照题目要求写即可



代码

```python
n = int(input())
l = []
def check(x):
    if x == 0:
        return '1'
    if x == 1:
        return '0'
    if x == 2:
        return 'X'
    if x == 3:
        return '9'
    if x == 4:
        return '8'
    if x == 5:
        return '7'
    if x == 6:
        return '6'
    if x == 7:
        return '5'
    if x == 8:
        return '4'
    if x == 9:
        return '3'
    if x == 10:
        return '2'
for i in range(n):
    l.append(input())
s = [7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2]
for _ in l:
    a = 0
    for i in range(17):
        a += int(_[i])*s[i]
    b = a % 11
    c = check(b)
    if c==_[-1]:
        print('YES')
    else:
        print('NO')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![28664](D:\物院\大一秋季学期\计算概论\课程作业\第五周作业（10月月考）\28664.png)



### M28678: 角谷猜想

http://cs101.openjudge.cn/practice/28678/



思路：直接遍历



代码

```python
n = int(input())
while n != 1:
    if n % 2 == 0:
        a = n // 2
        print(str(n) + '/2=' + str(a))
        n = n // 2
    else:
        a = n*3 + 1
        print(str(n) + '*3+1=' + str(a))
        n = n*3 + 1
print('End')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![28678](D:\物院\大一秋季学期\计算概论\课程作业\第五周作业（10月月考）\28678.png)



### M28700: 罗马数字与整数的转换

http://cs101.openjudge.cn/practice/28700/



思路：

①先定义从数字转换为罗马数字的函数，从千位遍历，从字典中查找数字对应的字母相加即可。

②定义从罗马数字转换为数字的函数，遍历字符，按照不同情况加上或者减去字母对应的值即可

（也可以直接用1①打表然后查找，更加暴力，思路更简单）



##### 代码

```python
s = input()
def change1(s):
    d1 = {'0':'','1':'I','2':'II','3':'III','4':'IV','5':'V',
          '6':'VI','7':'VII','8':'VIII','9':'IX'}
    d2 = {'0':'','1':'X','2':'XX','3':'XXX','4':'XL','5':'L',
          '6':'LX','7':'LXX','8':'LXXX','9':'XC'}
    d3 = {'0':'','1':'C','2':'CC','3':'CCC','4':'CD','5':'D',
          '6':'DC','7':'DCC','8':'DCCC','9':'CM'}
    d4 = {'0':'','1':'M','2':'MM','3':'MMM'}
    n = len(s)
    o = ''
    if n == 4:
        o = d4[s[0]]+d3[s[1]]+d2[s[2]]+d1[s[3]]
    if n == 3:
        o = d3[s[0]]+d2[s[1]]+d1[s[2]]
    if n == 2:
        o = d2[s[0]]+d1[s[1]]
    if n == 1:
        o = d1[s[0]]
    return o
def change3(s):
    d5 = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    a = len(s)
    o = 0
    for i in range(a-1):
        if (s[i] == 'I' and s[i+1] == 'V') or (s[i] == 'I' and s[i+1] == 'X'):
            o -= 1
        elif (s[i] == 'X' and s[i+1] == 'L') or (s[i] == 'X' and s[i+1] == 'C'):
            o -= 10
        elif (s[i] == 'C' and s[i+1] == 'D') or (s[i] == 'C' and s[i+1] == 'M'): 
            o -= 100
        else:
            o += int(d5[s[i]])
    o += int(d5[s[-1]])
    return o
        
if s.isdigit():
    print(change1(s))
else:
    print(change3(s))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![28700](D:\物院\大一秋季学期\计算概论\课程作业\第五周作业（10月月考）\28700.png)



### *T25353: 排队 （选做）

http://cs101.openjudge.cn/practice/25353/



思路：称现在可以直接移动到队头的节点为“自由的”，选出所有自由节点，按照升序排序并放到最前面，去除剩下的数字，重复以上操作，剩下无剩下的数字。



代码

```python
N,D=map(int,input().split())
l=[]
for i in range(N):
    l.append(int(input()))

o=[]
while len(l)>0:
    maxv=l[0]
    minv=l[0]
    a=[l[0]]
    b=[]
    for i in range(1,len(l)):
        if l[i]>=maxv-D and l[i]<=minv+D:
            a.append(l[i])
        else:
            b.append(l[i])
        minv=min(minv,l[i])
        maxv=max(maxv,l[i])
    l=b
    a.sort()
    o.append(a)

for i in o:
    for j in i:
        print(j)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![25353](D:\物院\大一秋季学期\计算概论\课程作业\第五周作业（10月月考）\25353.png)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。==

（1）排队好难，不太好想出简单高效的算法（

（2）坚持跟进每日选做，感觉题目难度有点提升不过自己还是能独立做出大部分题目

（3）尝试做了几道百练小组题目，无难度划分，容易卡住

（4）完成部分sy上三四章的题（感觉这上面贪心偏简单？）

![sy1](D:\物院\大一秋季学期\计算概论\课程作业\第五周作业（10月月考）\sy1.png)

![sy1](D:\物院\大一秋季学期\计算概论\课程作业\第五周作业（10月月考）\sy2.png)