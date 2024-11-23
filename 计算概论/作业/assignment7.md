# Assignment #7: Nov Mock Exam立冬

Updated 1646 GMT+8 Nov 7, 2024

2024 fall, Complied by <mark>李天笑、物理学院</mark>



**说明：**

1）⽉考： AC<mark>4</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### E07618: 病人排队

sorttings, http://cs101.openjudge.cn/practice/07618/

思路：

比较简单，分组排序输出就行

代码：

```python
n = int(input())
l1 = []
l2 = []
for i in range(n):
    ID, age = input().split()
    if int(age) >= 60:
        l1.append([ID, age])
    else:
        l2.append(ID)
l1.sort(key = lambda x:-int(x[1]))
for i in l1:
    print(i[0])
for i in l2:
    print(i)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241107184854207](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241107184854207.png)



### E23555: 节省存储的矩阵乘法

implementation, matrices, http://cs101.openjudge.cn/practice/23555/

思路：

在一般矩阵乘法的基础上添加了非零数字的搜索，也较为简单

代码：

```python
n, m1, m2 = map(int,input().split())
A = [[0]*n for _ in range(n)]
B = [[0]*n for _ in range(n)]
for i in range(m1):
    row, col, value = map(int,input().split())
    A[row][col] = value
for i in range(m2):
    row, col, value = map(int,input().split())
    B[row][col] = value
C = [[0]*n for _ in range(n)]
for i in range(n):
    for j in range(n):
        for k in range(n):
            C[i][j] += A[i][k] * B[k][j]
ans = []
for i in range(n):
    for j in range(n):
        if C[i][j] != 0:
            ans.append([i, j, C[i][j]])
for i in ans:
    print(' '.join([str(e) for e in i]))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20241107184939040](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241107184939040.png)



### M18182: 打怪兽 

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/

思路：

模拟，直接用程序模拟一个打怪兽的过程，为了尽可能打死怪兽，在每个时间段，哟溴铵使用伤害高的技能。采用current_time、used_skills、current_damage依次存储当前时间、用过技能、当前伤害。遍历排序后的技能，如果当前时间可以使用，则伤害+xi，used_skills+1直到技能打满或无技能，则要检测剩余血量是否大于0，如果小于等于0，直接返回当前时间，否则将伤害打满，清空伤害和用过技能数量，寻找下一个可以打伤害的时间，重复上述操作。最后将剩余伤害全部打到怪兽身上，再次检测血量是否大于0。

代码：

```python
def fight(n, m, b, tricks):    
    tricks.sort(key=lambda x: (x[0], -x[1]))  
    current_time = 0    
    used_skills = 0  
    current_damage = 0 
    for i in range(n):  
        ti, xi = tricks[i]  
        if ti != current_time:  
            if current_damage >= b:  
                return current_time   
            b -= current_damage  
            current_damage = 0  
            current_time = ti  
            used_skills = 0  
        if used_skills < m:  
            current_damage += xi  
            used_skills += 1  
    b -= current_damage   
    if b <= 0:  
        return current_time  
    else:  
        return 'alive'  
nCases = int(input())  
results = []  
for _ in range(nCases):  
    n, m, b = map(int, input().split())  
    tricks = []  
    for __ in range(n):  
        tricks.append(list(map(int, input().split())))  
    result = fight(n, m, b, tricks)  
    results.append(result)  
for res in results:  
    print(res)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241107201509981](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241107201509981.png)



### M28780: 零钱兑换3

dp, http://cs101.openjudge.cn/practice/28780/

思路：

经典dp题

代码：

```python
n, m = map(int,input().split())
l = list(map(int,input().split()))
dp = [10e9]*(m+1)
dp[0] = 0
for i in range(1,m+1):
    for j in l:
        if i >= j:
            dp[i] = min(dp[i],dp[i-j] + 1)
if dp[-1] == 10e9:
    print(-1)
else:
    print(dp[-1])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241107185041400](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241107185041400.png)



### T12757: 阿尔法星人翻译官

implementation, http://cs101.openjudge.cn/practice/12757

思路：

非常麻烦（恼

先利用字典将英文单词和数字对应起来，然后将单词转换为对应的数字。定义便改良total和current，分别存储总的数字和当前数字。从前往后检索每个数字，如果遇到了大于100的数字，就用current*这个数字赋值为新的current；否则就用current+这个数字为新的current；如果这个数字为1000或者1000000，由于1000和1000000不会作为更大的数字的修饰，故将total+current作为新的total，current更新为0。全部检索完最后再将current剩余值加到total中。

此外，再检索第一个单词，检查输出数据的正负号即可。

代码：

```python
s = input()
check = True
d = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5
     , 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 
     'eleven':11, 'twelve':12, 'thirteen':13, 'fourteen':14, 
     'fifteen':15, 'sixteen':16, 'seventeen':17, 'eighteen':18, 
     'nineteen':19, 'twenty':20, 'thirty':30, 'forty':40, 
     'fifty':50, 'sixty':60, 'seventy':70, 'eighty':80, 'ninety':90,
     'hundred':100, 'thousand':1000, 'million':1000000}
s1 = s.split()
if s1[0] == 'negative':
    check = False
    parts = s1[1:]
else:
    parts = s1
total = 0
current = 0
for part in parts:
    if part == 'and':
        continue
    number = d[part]
    if number >= 100:
        current *= number
    else:
        current += number 
    if (number == 1000 or number == 1000000) and len(parts) > 1:
        total += current
        current = 0
total += current
if check:
    print(total)
else:
    print('-'+str(total))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241107185146356](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241107185146356.png)



### T16528: 充实的寒假生活

greedy/dp, cs10117 Final Exam, http://cs101.openjudge.cn/practice/16528/

思路：

其实是一道比较经典的贪心题目，之前也做过类似的，但是考试时由于前面三、五题耽误的时间太长，导致留给最后一题时间太短，着急之下提交了三次wa，最后改对了但是刚好5：00没来的及提交

思路上就是先把一系列活动按照结束时间从小到大排序，定义变量last_time。对于每个活动，如果开始的时间大于last_time，则可以参加，count+1，然后把last_time更新为新活动结束时间和last_time中的最大值，最后输出count即可。

代码：

```python
n = int(input())
time = []
for i in range(n):
    time.append(list(map(int, input().split())))
time.sort(key = lambda x:(x[1], x[0]))
last_time = float('-inf')
count = 0
for i in range(n):
    start, end = time[i][0], time[i][1]
    if start > last_time:
        count += 1
        last_time = max(last_time, end)
print(count)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241107191026574](C:\Users\ltx18\AppData\Roaming\Typora\typora-user-images\image-20241107191026574.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

认为今天考试发挥不好，只AC4（）感觉最后一题比3、5都简单，但是没时间做完。此外代码wa或Ra后检查代码能力不够，下周考完期中后要尽可能把计算概论追回来。

感觉以后写题可以尝试写一些注释，复杂的题目代码阅读检查方便

目前每日选做做到11.6号。



