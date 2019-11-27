51 [N-Queens](https://leetcode-cn.com/problems/n-queens/)
>n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
![](res/chapter6-1.png)  
给定一个整数 n，返回所有不同的 n 皇后问题的解决方案。
每一种解法包含一个明确的 n 皇后问题的棋子放置方案，该方案中`'Q'`和`'.'`分别代表了皇后和空位。

示例:
```
输入: 4
输出: [
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]
解释: 4 皇后问题存在两个不同的解法。
```

```python
class Solution:
    def solveNQueens(self, n):
        def DFS(queens, xy_dif, xy_sum):
            p = len(queens)
            if p == n:
                result.append(queens)
                return None
            for q in range(n):  # q是每列坐标
                if (
                    q not in queens and p - q not in xy_dif and p + q not in xy_sum
                ):  # 斜边条件检查(斜边坐标相减为定值,反斜边坐标相加为定值)
                    DFS(queens + [q], xy_dif + [p - q], xy_sum + [p + q])

        result = []
        DFS([], [], [])
        return [["." * i + "Q" + "." * (n - i - 1) for i in sol] for sol in result]
```

52 [N-Queens II](https://leetcode-cn.com/problems/n-queens-ii/)
>n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
![](res/chapter6-1.png)  
给定一个整数 n，返回 n 皇后不同的解决方案的数量。

示例:
```
输入: 4
输出: 2
解释: 4 皇后问题存在如下两个不同的解法。
[
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]
```

```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        def DFS(queens, xy_dif, xy_sum):
            p = len(queens)
            if p == n:
                result.append(queens)
                return None
            for q in range(n):  # q是每列坐标
                if (
                    q not in queens and p - q not in xy_dif and p + q not in xy_sum
                ):  # 斜边条件检查(斜边坐标相减为定值,反斜边坐标相加为定值)
                    DFS(queens + [q], xy_dif + [p - q], xy_sum + [p + q])

        result = []
        DFS([], [], [])
        return len(result)
```

53 [Maximum Subarray](https://leetcode-cn.com/problems/maximum-subarray/)
>给定一个整数数组`nums`，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

示例:
```
输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```
进阶：`如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。`

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            nums[i] = max(nums[i], nums[i] + nums[i - 1])
        return max(nums)
```
54 [Spiral Matrix](https://leetcode-cn.com/problems/spiral-matrix/)
>给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。

示例:
```
输入:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
输出: [1,2,3,6,9,8,7,4,5]

输入:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
输出: [1,2,3,4,8,12,11,10,9,5,6,7]
```

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        return matrix and [*matrix.pop(0)] + self.spiralOrder([*zip(*matrix)][::-1])
```
- 注意是`[*matrix.pop(0)] `

55 [Jump Game](https://leetcode-cn.com/problems/jump-game/)
>给定一个非负整数数组，你最初位于数组的第一个位置。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
判断你是否能够到达最后一个位置。

示例:
```
输入: [2,3,1,1,4]
输出: true
解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。

输入: [3,2,1,0,4]
输出: false
解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。
```

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        m = 0
        for i, n in enumerate(nums):
            if i > m: # 位置i超过了所能跳到的最大位置
                return False
            m = max(m, i + n)
        return True
```
- m保存能跳到的最远距离
- 遍历数组，依次更新m，当i>m时，返回False

56 [合并区间](https://leetcode-cn.com/problems/merge-intervals/)
>给出一个区间的集合，请合并所有重叠的区间。

示例:
```
输入: [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

输入: [[1,4],[4,5]]
输出: [[1,5]]
解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。
```

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        out = []
        for i in sorted(intervals, key=lambda i: i[0]):
            if out and i[0] <= out[-1][-1]:
                out[-1][-1] = max(out[-1][-1], i[-1])
            else:
                out += [i]
        return out
```
- 按照坐标起点排序,判断边界合并

57 [Insert Interval](https://leetcode-cn.com/problems/insert-interval/)
>给出一个无重叠的 ，按照区间起始端点排序的区间列表。
在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

示例:
```
输入: intervals = [[1,3],[6,9]], newInterval = [2,5]
输出: [[1,5],[6,9]]

输入: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
输出: [[1,2],[3,10],[12,16]]
解释: 这是因为新的区间 [4,8] 与 [3,5],[6,7],[8,10] 重叠。
```

```python
class Solution:
    def insert(self, intervals: List[List[int]],
               newInterval: List[int]) -> List[List[int]]:
        start, end = newInterval[0], newInterval[-1]
        left, right = [], []
        for i in intervals:
            if i[-1] < start:
                left.append(i)
            elif i[0] > end:
                right.append(i)
            else:
                start = min(start, i[0])
                end = max(end, i[-1])

        return left + [[start, end]] + right
```
- 借鉴56题的思路


58 [Length of Last Word](https://leetcode-cn.com/problems/length-of-last-word/)
>给定一个仅包含大小写字母和空格` ' ' `的字符串，返回其最后一个单词的长度。
如果不存在最后一个单词，请返回 0 。
说明：一个单词是指由字母组成，但不包含任何空格的字符串。

示例:
```
输入: "Hello World"
输出: 5
```
说明：`你的算法的时间复杂度应为O(n)，并且只能使用常数级别的空间。`

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        return len(s.strip(' ').split(' ')[-1])
```

59 [Spiral Matrix II](https://leetcode-cn.com/problems/spiral-matrix-ii/)
>给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。

示例:
```
输入: 3
输出:
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]
```

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        r, n = [[n**2]], n**2
        while n > 1:
            n, r = n - len(r), [[*range(n - len(r), n)]] + [*zip(*r[::-1])]
        return r
```
- 流程图
```
||  =>  |9|  =>  |8|      |6 7|      |4 5|      |1 2 3|
		         |9|  =>  |9 8|  =>  |9 6|  =>  |8 9 4|
				                     |8 7|      |7 6 5|
```

60 [Permutation Sequence](https://leetcode-cn.com/problems/permutation-sequence/)
>给出集合 [1,2,3,…,n]，其所有元素共有 n! 种排列。
按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：
```
    "123"
    "132"
    "213"
    "231"
    "312"
    "321"
```
给定 n 和 k，返回第 k 个排列。
说明:给定 n 的范围是 [1, 9]。给定 k 的范围是[1,  n!]。

示例:
```
输入: n = 3, k = 3
输出: "213"

输入: n = 4, k = 9
输出: "2314"
```

```python
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        # 0-9的阶乘
        self.fac = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
        self.nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # 康托编码
        res = ''
        k -= 1
        for i in reversed(range(n)):
            cur = self.nums[k // self.fac[i]]
            res += str(cur)
            self.nums.remove(cur)
            if i != 0:
                k %= self.fac[i]
                self.fac[i] //= i
        return res
```
- 康托编码