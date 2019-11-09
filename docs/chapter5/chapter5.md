41. [First Missing Positive](https://leetcode-cn.com/problems/first-missing-positive/solution/que-shi-de-di-yi-ge-zheng-shu-by-leetcode/)
```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            while 0 <= nums[i] - 1 < len(nums) and nums[nums[i]-1] != nums[i]:
                tmp = nums[i] - 1
                nums[i], nums[tmp] = nums[tmp], nums[i]
        for i in range(len(nums)):
            if nums[i] != i + 1:
                return i + 1
        return len(nums) + 1
```
- 桶排序 负数、零和大于N的正数不用考虑。将1~n之内的数映射到0~n-1,最后遍历找出缺失数



42. [Trapping Rain Water](https://leetcode-cn.com/problems/trapping-rain-water/)
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        # 找到最高的柱子,处理左边一半,处理右边一半
        max = 0
        for i in range(len(height)):
            if height[i] > height[max]:
                max = i
        water, peak, top = 0, 0, 0
        for i in range(max):
            if height[i] > peak:
                peak = height[i]
            else:
                water += peak - height[i]
        for i in range(len(height) - 1, max, -1):
            if height[i] > top:
                top = height[i]
            else:
                water += top - height[i]
        return water
```
43. [Multiply Strings](https://leetcode-cn.com/problems/multiply-strings/)
```python
import re
import math
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        d = {}
        for i, n1 in enumerate(num1[::-1]):
            for j, n2 in enumerate(num2[::-1]):
                d[i + j] = d.get(i + j, 0) + (ord(n1) - 48) * (ord(n2) - 48)
        for k in [*d]:
            d[k+1], d[k] = d.get(k + 1, 0) + math.floor(d[k] * 0.1), d[k] % 10
        return re.sub('^0*', '', ''.join(map(str, d.values()))[::-1]) or '0'
```
- 大数乘法


44. [Wildcard Matching](https://leetcode-cn.com/problems/wildcard-matching/)
```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        length = len(s)
        if len(p) - p.count('*') > length:
            return False
        dp = [True] + [False] * length
        for i in p:
            if i != '*':
                for n in reversed(range(length)):
                    dp[n + 1] = dp[n] and (i == s[n] or i == '?')
            else:
                for n in range(1, length + 1):
                    dp[n] = dp[n - 1] or dp[n]
            dp[0] = dp[0] and i == '*'
        return dp[-1]
```
- '?' 可以匹配任何单个字符。'*' 可以匹配任意字符串（包括空字符串）。

45. [Jump Game II](https://leetcode-cn.com/problems/jump-game-ii/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-10/)
```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n, start, end, step = len(nums), 0, 0, 0
        while end < n - 1:
            step += 1
            maxend = end + 1
            for i in range(start, end + 1):
                if i + nums[i] >= n - 1:
                    return step
                maxend = max(maxend, i + nums[i])
            start, end = end + 1, maxend
        return step
```
46. []

47. [Permutations](https://leetcode-cn.com/problems/permutations/)
```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        return [[n] + sub for i, n in enumerate(nums)
                for sub in self.permute(nums[:i] + nums[i + 1:])] or [nums]
```

- 每次固定第一个数字递归地排列数组剩余部分

```python
 class Solution:
     def permute(self, nums: List[int]) -> List[List[int]]:
        from itertools import permutations
        return list(permutations(nums))
```
- 用内置函数直接实现


47. [Permutations II](https://leetcode-cn.com/problems/permutations-ii/solution/hui-su-suan-fa-python-dai-ma-java-dai-ma-by-liwe-2/)
```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        from itertools import permutations
        return list(set(permutations(nums)))
```

48. [Rotate Image](https://leetcode-cn.com/problems/rotate-image/)
```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        for i in range(m):  # 沿着副对角线翻折
            for j in range(m - i):
                matrix[i][j], matrix[m - 1 - j][m - 1 -
                                                i] = matrix[m - 1 - j][m - 1 - i], matrix[i][j]
        for i in range(m//2):  # 沿着水平中线翻折
            for j in range(m):
                matrix[i][j], matrix[m-1-i][j] = matrix[m-1-i][j], matrix[i][j]
```

50. [Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)
```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if not n:
            return 1
        elif n < 0:
            return 1 / self.myPow(x, -n)
        elif n % 2:
            return x * self.myPow(x, n-1)
        return self.myPow(x*x, n/2)
```