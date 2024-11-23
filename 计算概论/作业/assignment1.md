### 1. 题目

### 02733：判断闰年

```python
n=int(input())
if (n%4==0 and n%100!=0) or (n%400==0):
    print("Y")
else:
    print("N")
```

![判断闰年](D:\物院\大一秋季学期\计算概论\课程作业\第一周作业\判断闰年.png)

### 02750：鸡兔同笼

```python
def calculate_animals(a):
    if a%2!=0:
        print(str(0)+" "+str(0))
    if a%4==2:
        max_animals=a // 2
        min_animals=(a+2) // 4
        print(str(min_animals)+" "+str(max_animals))
    if a%4==0:
        max_animals=a // 2
        min_animals=a // 4
        print(str(min_animals)+" "+str(max_animals))
a = int(input().strip())
calculate_animals(a)
```

![鸡兔同笼](D:\物院\大一秋季学期\计算概论\课程作业\第一周作业\鸡兔同笼.png)

### 50A. Domino piling

```python
def square(M,N):
    area=M*N
    if area%2==0:
        number=area//2
        print(number)
    else:
        number=(area-1)//2
        print(number)
M, N = map(int, input().strip().split())
square(M,N)
```

![Domino piling](D:\物院\大一秋季学期\计算概论\课程作业\第一周作业\Domino piling.png)

### 1A. Theatre Square

```python
def number(n,m,a):
    flagstones_length = (n+a-1)//a
    flagstones_width = (m+a-1)//a
    total_flagstones = flagstones_length * flagstones_width
    print(total_flagstones)
n,m,a = map(int, input().strip().split())
number(n,m,a)
```

![Theatre Square](D:\物院\大一秋季学期\计算概论\课程作业\第一周作业\Theatre Square.png)

### 112A. Petya and Strings

```python
letters_1=input()
letters_2=input()
letters_1_lower=letters_1.lower()
letters_2_lower=letters_2.lower()
if letters_1_lower==letters_2_lower:
    print(0)
if letters_1_lower<letters_2_lower:
    print(-1)
if letters_1_lower>letters_2_lower:
    print(1)
```

![Petya and Strings](D:\物院\大一秋季学期\计算概论\课程作业\第一周作业\Petya and Strings.png)

### 231A. Team

```python
n=int(input())
numbers=[]
for i in range(n):
    line=input()
    numbers.append(list(map(int,line.strip().split())))
num_0=[]
for number in numbers:
    num_1=[]
    for num in number:
        if num==1:
            num_1.append(num)
    if len(num_1)>=2:
        num_0.append(num_1)
print(len(num_0))
```

![Team](D:\物院\大一秋季学期\计算概论\课程作业\第一周作业\Team.png)

## 2. 学习总结和收获

1.学习了python基本的语法，能独立完成python较为简单的习题,难度大于1000的题目有时还得向GPT求助。

2.学习了markdown的基本编辑方法。

3.进行了额外练习：每天跟着进度完成每日选做（截至目前已经做完了9.20及以前的全部每日选做），又独立完成了2020年OJ和CF的部分习题。





