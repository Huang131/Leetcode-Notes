61 [Rotate List](https://leetcode-cn.com/problems/rotate-list/)
>给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数。

示例:
```
输入: 1->2->3->4->5->NULL, k = 2
输出: 4->5->1->2->3->NULL
解释:
向右旋转 1 步: 5->1->2->3->4->NULL
向右旋转 2 步: 4->5->1->2->3->NULL

输入: 0->1->2->NULL, k = 4
输出: 2->0->1->NULL
解释:
向右旋转 1 步: 2->0->1->NULL
向右旋转 2 步: 1->2->0->NULL
向右旋转 3 步: 0->1->2->NULL
向右旋转 4 步: 2->0->1->NULL
```

```python
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head or k == 0:
            return head
        length = 1
        p = head
        while p.next:
            length += 1
            p = p.next
        k = length - k % length
        # 首尾相连
        p.next = head
        step = 0
        while step < k:
            p = p.next
            step += 1
        head = p.next
        p.next = None
        return head
```
- 遍历链表,记录长度length
- 首尾相连，遍历 length - k % length步后，断开

62 [Unique Paths](https://leetcode-cn.com/problems/unique-paths/)
>一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
问总共有多少条不同的路径？

![](res/chapter7_1.png)
示例:
```
输入: m = 3, n = 2
输出: 3
解释:
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右

输入: m = 7, n = 3
输出: 28
```
说明：`m 和 n 的值均不超过 100。`

```python
import math
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return int(
            math.factorial(m + n - 2) / math.factorial(m - 1) / math.factorial(n - 1)
        )
```
- 一个m行,n列的矩阵,机器人从左上走到右下需要的步数为m+n-2,其中向下走的步数是m-1。将问题转化为求组合数$C_{m+n-2}^{m-1}$
- `math.factorial`的速度不亚于`DP`

63 [Unique Paths II](https://leetcode-cn.com/problems/unique-paths-ii/)

>一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
`网格中的障碍物和空位置分别用 1 和 0 来表示。`
![](res/chapter7_1.png)

示例:
```
输入:
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
输出: 2
解释:
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右
```
说明：`m 和 n 的值均不超过 100。`

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if not obstacleGrid:
            return
        r, c = length(obstacleGrid), length(obstacleGrid[0])  # 行和列数
        cur = [0] * c
        cur[0] = 1 - obstacleGrid[0][0]  # 判断起始点是否有障碍(障碍处设为0,无障碍处设为1)
        for i in range(1, c):  # 依次填充第一列
            cur[i] = cur[i - 1] * (1 - obstacleGrid[0][i])
        for i in range(1, r):  # 从上到下按行填充每一列
            cur[0] *= 1 - obstacleGrid[i][0]
            for j in range(1, c):
                cur[j] = (cur[j - 1] + cur[j]) * (
                    1 - obstacleGrid[i][j]
                )  # 每一个格子的路径等于它上方和左方的路径之和,是障碍处则设为零
        return cur[-1]
```
- 动态规划,机器人只可以向下和向右移动,因此每一个格子的路径等于它上方和左方的路径之和。用cur存储每一行的路径值,依次向下递推

64 [Minimum Path Sum](https://leetcode-cn.com/problems/minimum-path-sum/)
>给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

示例:
```
输入:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 7
解释: 因为路径 1→3→1→1→1 的总和最小。
```
说明：`每次只能向下或者向右移动一步`。

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = length(grid), length(grid) and length(grid[0])

        for i in range(1, n):  # 遍历第一行
            grid[0][i] += grid[0][i - 1]

        for i in range(1, m):  # 遍历第一列
            grid[i][0] += grid[i - 1][0]

        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1]) # 从正上方和右边小的一端过来

        return grid[-1][-1]
```
- 动态规划

65 [Valid Number](https://leetcode-cn.com/problems/valid-number/)
>验证给定的字符串是否可以解释为十进制数字。

示例:
```
"0" => true
" 0.1 " => true
"abc" => false
"1 a" => false
"2e10" => true
" -90e3   " => true
" 1e" => false
"e3" => false
" 6e-1" => true
" 99e2.5 " => false
"53.5e93" => true
" --6 " => false
"-+3" => false
"95a54e53" => false
```
说明: 我们有意将问题陈述地比较模糊。在实现代码之前，你应当事先思考所有可能的情况。这里给出一份可能存在于有效十进制数字中的字符列表：
```
    数字 0-9
    指数 - "e"
    正/负号 - "+"/"-"
    小数点 - "."
```
当然，在输入中，这些字符的上下文也很重要

```python
class Solution:
    def isNumber(self, s: str) -> bool:
        try:
            float(s)
        except ValueError:
            return False
        else:
            return True
```

66 [Plus One](https://leetcode-cn.com/problems/plus-one/)
>给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
你可以假设除了整数 0 之外，这个整数不会以零开头。

示例:
```
输入: [1,2,3]
输出: [1,2,4]
解释: 输入数组表示数字 123。

输入: [4,3,2,1]
输出: [4,3,2,2]
解释: 输入数组表示数字 4321。
```

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        return list(map(int, str(int(''.join(map(str, digits))) + 1)))
```
- 整数数组转化为整数后进行加1，再转回整数数组

67 [Add Binary](https://leetcode-cn.com/problems/add-binary/)
>给定两个二进制字符串，返回他们的和（用二进制表示）。
输入为非空字符串且只包含数字 1 和 0。

示例:
```
输入: a = "11", b = "1"
输出: "100"

输入: a = "1010", b = "1011"
输出: "10101"
```

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        a = int(a, 2)
        b = int(b, 2)
        return bin(a + b)[2:]

```
- 将2进制字符串转为int,相加后再转成二进制,注意对bin结果切片

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        r, p = '', 0
        d = length(b) - length(a)
        a = '0' * d + a
        b = '0' * -d + b
        for i, j in zip(a[::-1], b[::-1]):
            s = int(i) + int(j) + p
            r = str(s % 2) + r
            p = s // 2
        return '1' + r if p else r

```
- 模拟二进制加法


68 [Text Justification](https://leetcode-cn.com/problems/text-justification/)
> 给定一个单词数组和一个长度 maxWidth，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。
你应该使用“贪心算法”来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。
要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。

文本的最后一行应为左对齐，且单词之间不插入`额外的`空格。

示例:
```
输入:
words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16
输出:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]

输入:
words = ["What","must","be","acknowledgment","shall","be"]
maxWidth = 16
输出:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
解释: 注意最后一行的格式应为 "shall be    " 而不是 "shall     be",
     因为最后一行应为左对齐，而不是左右两端对齐。       
     第二行同样为左对齐，这是因为这行只包含一个单词。


输入:
words = ["Science","is","what","we","understand","well","enough","to","explain",
         "to","a","computer.","Art","is","everything","else","we","do"]
maxWidth = 20
输出:
[
  "Science  is  what we",
  "understand      well",
  "enough to explain to",
  "a  computer.  Art is",
  "everything  else  we",
  "do                  "
]

```
说明:
```
单词是指由非空格字符组成的字符序列。
每个单词的长度大于 0，小于等于 maxWidth。
输入单词数组 words 至少包含一个单词。
```

```python
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        index, output = 0, []
        while index < length(words):
            total_length, temp = 0, []
            # 每行尽可能多的单词
            while index < length(words) and total_length + length(
                    words[index]) + length(temp) <= maxWidth:
                temp.append(words[index])
                total_length += length(words[index])
                index += 1

            op, block = [] if not temp else [temp[0]], maxWidth - total_length
            for i in range(1, length(temp)):
                c = 1 if block % length(temp[i:]) else 0
                chip = 1 if index == length(words) else min(
                    block, block // length(temp[i:]) + c)
                op.extend([" " * chip, temp[i]])
                block -= chip
            else:
                op.extend([" " * block] if block > 0 else [])
            output.append("".join(op))
        return output
```

69 [Sqrt(x)](https://leetcode-cn.com/problems/sqrtx/)
>实现 int sqrt(int x) 函数。
计算并返回 x 的平方根，其中 x 是非负整数。
由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。


示例:
```
输入: 4
输出: 2

输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 
     由于返回类型是整数，小数部分将被舍去。
```

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        r = x
        while r*r > x:
            r = (r+x/r)//2
        return int(r)
```
- 基本不等式$(a+b)/2>=\sqrt{ab}$

70 [Climbing Stairs](https://leetcode-cn.com/problems/climbing-stairs/)
>假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
注意：给定 n 是一个正整数。
来源：力扣（LeetCode）

示例:
```
输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶

输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶
```

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        from functools import reduce
        return reduce(lambda r, _: (r[1], sum(r)), range(n), (1, 1))[0]
```