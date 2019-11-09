61. [Rotate List](https://leetcode-cn.com/problems/rotate-list/)
```python
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head or k == 0:
            return head
        len = 1
        p = head
        while p.next:
            len += 1
            p = p.next
        k = len - k % len
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

62. [Unique Paths](https://leetcode-cn.com/problems/unique-paths/)
```python
import math
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return int(
            math.factorial(m + n - 2) / math.factorial(m - 1) / math.factorial(n - 1)
        )
```
- 一个m行,n列的矩阵,机器人从左上走到右下需要的步数为m+n-2,其中向下走的步数是m-1。将问题转化为求组合数$C_{m+n-2}^{m-1}$


63. [Unique Paths II](https://leetcode-cn.com/problems/unique-paths-ii/)
```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if not obstacleGrid:
            return
        r, c = len(obstacleGrid), len(obstacleGrid[0])  # 行和列数
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



64. [Minimum Path Sum](https://leetcode-cn.com/problems/minimum-path-sum/)
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid) and len(grid[0])

        for i in range(1, n):  # 遍历第一行
            grid[0][i] += grid[0][i - 1]

        for i in range(1, m):  # 遍历第一列
            grid[i][0] += grid[i - 1][0]

        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])

        return grid[-1][-1]
```


65. [Valid Number](https://leetcode-cn.com/problems/valid-number/)
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
66. [Plus One](https://leetcode-cn.com/problems/plus-one/)
```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        return list(map(int, str(int(''.join(map(str, digits))) + 1)))
```

67. [Add Binary](https://leetcode-cn.com/problems/add-binary/)
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
        d = len(b) - len(a)
        a = '0' * d + a
        b = '0' * -d + b
        for i, j in zip(a[::-1], b[::-1]):
            s = int(i) + int(j) + p
            r = str(s % 2) + r
            p = s // 2
        return '1' + r if p else r

```
- 模拟二进制加法


68. [Text Justification](https://leetcode-cn.com/problems/text-justification/solution/)
```python
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        index, output = 0, []
        while index < len(words):
            total_len, temp = 0, []
            # 每行尽可能多的单词
            while index < len(words) and total_len + len(
                    words[index]) + len(temp) <= maxWidth:
                temp.append(words[index])
                total_len += len(words[index])
                index += 1

            op, block = [] if not temp else [temp[0]], maxWidth - total_len
            for i in range(1, len(temp)):
                c = 1 if block % len(temp[i:]) else 0
                chip = 1 if index == len(words) else min(
                    block, block // len(temp[i:]) + c)
                op.extend([" " * chip, temp[i]])
                block -= chip
            else:
                op.extend([" " * block] if block > 0 else [])
            output.append("".join(op))
        return output
```

69. [Sqrt(x)](https://leetcode-cn.com/problems/sqrtx/)
```python
class Solution:
    def mySqrt(self, x: int) -> int:
        r = x
        while r*r > x:
            r = (r+x/r)//2
        return int(r)
```
- 基本不等式$(a+b)/2>=\sqrt{ab}$

70. [Climbing Stairs](https://leetcode-cn.com/problems/climbing-stairs/)
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        from functools import reduce
        return reduce(lambda r, _: (r[1], sum(r)), range(n), (1, 1))[0]
```