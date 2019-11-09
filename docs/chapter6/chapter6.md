51. [N-Queens](https://leetcode-cn.com/problems/n-queens/solution/hui-su-fa-by-jason-2/)
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

52. [N-Queens II](https://leetcode-cn.com/problems/n-queens-ii/)
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

53. [Maximum Subarray](https://leetcode-cn.com/problems/maximum-subarray/)
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            nums[i] = max(nums[i], nums[i] + nums[i - 1])
        return max(nums)
```
54. [Spiral Matrix](https://leetcode-cn.com/problems/spiral-matrix/)
```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        return matrix and [*matrix.pop(0)] + self.spiralOrder([*zip(*matrix)][::-1])
```


55. [Jump Game](https://leetcode-cn.com/problems/jump-game/)
```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        m = 0
        for i, n in enumerate(nums):
            if i > m:
                return False
            m = max(m, i + n)
        return True
```


56. [合并区间](https://leetcode-cn.com/problems/merge-intervals/)
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

57. [Insert Interval](https://leetcode-cn.com/problems/insert-interval/)
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


58. [Length of Last Word](https://leetcode-cn.com/problems/length-of-last-word/)
```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        return len(s.strip(' ').split(' ')[-1])
```

59. [Spiral Matrix II](https://leetcode-cn.com/problems/spiral-matrix-ii/)
```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        r, n = [[n**2]], n**2
        while n > 1:
            n, r = n - len(r), [[*range(n - len(r), n)]] + [*zip(*r[::-1])]
        return r
```


60. [Permutation Sequence](https://leetcode-cn.com/problems/permutation-sequence/)
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