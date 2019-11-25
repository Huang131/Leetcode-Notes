41 [First Missing Positive](https://leetcode-cn.com/problems/first-missing-positive/)

>给定一个未排序的整数数组，找出其中没有出现的最小的正整数。

示例:
```
输入: [1,2,0]
输出: 3

输入: [3,4,-1,1]
输出: 2

输入: [7,8,9,11,12]
输出: 1
```
说明：`你的算法的时间复杂度应为O(n)，并且只能使用常数级别的空间。`

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


42 [Trapping Rain Water](https://leetcode-cn.com/problems/trapping-rain-water/)
>给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

![](res/chapter5-1.png)

示例:
```
输入: [0,1,0,2,1,0,1,3,2,1,2,1]
输出: 6
```

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
- 找到最高的柱子后，左右两边分别处理。`peak`和`top`取局部最大值,对`height[i]`做减法求得`water`

43 [Multiply Strings](https://leetcode-cn.com/problems/multiply-strings/)
>给定两个以字符串形式表示的非负整数`num1`和`num2`，返回`num1`和`num2`的乘积，它们的乘积也表示为字符串形式。

示例:
```
输入: num1 = "2", num2 = "3"
输出: "6"

输入: num1 = "123", num2 = "456"
输出: "56088"

```
说明:
```
1. num1 和 num2 的长度小于110。
2. num1 和 num2 只包含数字 0-9。
3. num1 和 num2 均不以零开头，除非是数字 0 本身。
4. 不能使用任何标准库的大数类型（比如 BigInteger）或直接将输入转换为整数来处理。
```

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
- 大数乘法，本题难点在于计算整数的时候不能超过32bits，因此使用竖式计算
- 注意遍历字典时用的`[*d]`而不是`d.keys()`，因为字典的大小在循环时会发生变化

44 [Wildcard Matching](https://leetcode-cn.com/problems/wildcard-matching/)
>给定一个字符串`(s)`和一个字符模式`(p)`，实现一个支持`'?'`和`'*'`的通配符匹配。

```
'?' 可以匹配任何单个字符。
'*' 可以匹配任意字符串（包括空字符串）。
```
两个字符串完全匹配才算匹配成功。
说明:
```
s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。
```

示例：
```
输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。

输入:
s = "aa"
p = "*"
输出: true
解释: '*' 可以匹配任意字符串。

输入:
s = "cb"
p = "?a"
输出: false
解释: '?' 可以匹配 'c', 但第二个 'a' 无法匹配 'b'。

输入:
s = "adceb"
p = "*a*b"
输出: true
解释: 第一个 '*' 可以匹配空字符串, 第二个 '*' 可以匹配字符串 "dce".

输入:
s = "acdcb"
p = "a*c?b"
输入: false
```

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

45 [Jump Game II](https://leetcode-cn.com/problems/jump-game-ii/)
>给定一个非负整数数组，你最初位于数组的第一个位置。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
你的目标是使用最少的跳跃次数到达数组的最后一个位置。

示例:
```
输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
```
说明:`假设你总是可以到达数组的最后一个位置。`


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
- 贪心算法


46 [Permutations](https://leetcode-cn.com/problems/permutations/)
>给定一个`没有重复数字`的序列，返回其所有可能的全排列。

示例:
```
输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]

```

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


47 [Permutations II](https://leetcode-cn.com/problems/permutations-ii/)
>给定一个可包含`重复数字`的序列，返回所有不重复的全排列。

示例：
```
输入: [1,1,2]
输出:
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]
```

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        from itertools import permutations
        return list(set(permutations(nums)))
```
- set去重

48 [Rotate Image](https://leetcode-cn.com/problems/rotate-image/)

>给定一个 n × n 的二维矩阵表示一个图像。
将图像顺时针旋转 90 度。
说明：
你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。

示例:
```
给定 matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

原地旋转输入矩阵，使其变为:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]

给定 matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

原地旋转输入矩阵，使其变为:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]
```

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
- 注意边界条件

49 [Group Anagrams](https://leetcode-cn.com/problems/group-anagrams/)
>给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

示例:
```
输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
```
说明:
```
所有输入均为小写字母。
不考虑答案输出的顺序。
```

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        d = {}
        for w in sorted(strs):
            key = tuple(sorted(w))
            d[key] = d.get(key, []) + [w]
        return d.values()
```

```python
class Solution:
    def groupAnagrams(self, strs):
        return [[*x] for _, x in itertools.groupby(sorted(strs, key=sorted), sorted)]
```
- 使用 groupby 函数依据 sorted 结果分组

50 [Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)
>实现 pow(x, n) ，即计算 x 的 n 次幂函数。

示例:
```
输入: 2.00000, 10
输出: 1024.00000

输入: 2.10000, 3
输出: 9.26100

输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25
```
说明:
```
-100.0 < x < 100.0
n 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。

```

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
- 递归