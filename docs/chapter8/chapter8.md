71. [Simplify Path](https://leetcode-cn.com/problems/simplify-path/)
```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []
        for token in path.split('/'):
            if token in ('', '.'):
                pass
            elif token == '..':
                if stack:
                    stack.pop()
            else:
                stack.append(token)
        return '/' + '/'.join(stack)
```

72. [Edit Distance](https://leetcode-cn.com/problems/edit-distance/)
```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        l1, l2 = len(word1) + 1, len(word2) + 1
        pre = [0 for _ in range(l2)]
        for j in range(l2):
            pre[j] = j
        for i in range(1, l1):
            cur = [i] * l2
            for j in range(1, l2):
                cur[j] = min(cur[j - 1] + 1, pre[j] + 1,
                             pre[j - 1] + (word1[i - 1] != word2[j - 1]))
            pre = cur[:]
        return pre[-1]
```

73. [Set Matrix Zeroes](https://leetcode-cn.com/problems/set-matrix-zeroes/)
```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix) and len(matrix[0])
        row_has_zero = False
        col_has_zero = False

        for i in range(n):
            if matrix[0][i] == 0:
                row_has_zero = True
                break
        for i in range(m):
            if matrix[i][0] == 0:
                col_has_zero = True
                break
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[0][j] = 0
                    matrix[i][0] = 0
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if row_has_zero:
            for i in range(n):
                matrix[0][i] = 0
        if col_has_zero:
            for i in range(m):
                matrix[i][0] = 0
```

74.  [Search a 2D Matrix](https://leetcode-cn.com/problems/search-a-2d-matrix/)
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or target is None:
            return False
        rows, cols = len(matrix), len(matrix[0])
        i, j = 0, cols - 1
        while (i < rows and j >= 0):
            if matrix[i][j] > target:
                j -= 1
            elif matrix[i][j] < target:
                i += 1
            else:
                return True
        return False
```

75. [Sort Colors](https://leetcode-cn.com/problems/sort-colors/)
```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        red, white, blue = 0, 0, len(nums) - 1
        while white <= blue:
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red]
                white += 1
                red += 1
            elif nums[white] == 1:
                white += 1
            else:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1
```
- 双指针，两边向中间走


76. [Minimum Window Substring](https://leetcode-cn.com/problems/minimum-window-substring/)
```python
import collections
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need, missing = collections.Counter(t), len(t)
        i = I = J = 0
        for j, c in enumerate(s, 1):
            missing -= need[c] > 0
            need[c] -= 1
            if not missing:
                while i < j and need[s[i]] < 0:
                    need[s[i]] += 1
                    i += 1
                if not J or j - i <= J - I:
                    I, J = i, j
        return s[I:J]
```
- 滑动窗口,Counter记录所需字符数 while 循环移动左指针,寻找更短子串


77. [Combinations](https://leetcode-cn.com/problems/combinations/)
```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        from itertools import combinations
        return list(combinations(range(1, n + 1), k))
```


78. [Subsets](https://leetcode-cn.com/problems/subsets/)
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        from itertools import combinations
        return sum([list(combinations(nums, i)) for i in range(len(nums) + 1)],
                   [])
```

79. [Word Search](https://leetcode-cn.com/problems/word-search/)
```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board:
            return False
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.dfs(board, i, j, word):
                    return True
        return False

    def dfs(self, board, i, j, word):
        if len(word) == 0:
            return True
        if i < 0 or i >= len(board) or j < 0 or j >= len(
                board[0]) or word[0] != board[i][j]:
            return False
        tmp = board[i][j]
        board[i][j] = "#"
        res = self.dfs(board, i+1, j, word[1:]) or self.dfs(board, i-1, j, word[1:]) or self.dfs(board, i, j+1, word[1:]) or self.dfs(board, i, j-1, word[1:])
        board[i][j] = tmp
        return res
```


80. [Remove Duplicates from Sorted Array II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii)
```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return len(nums)
        index = 2
        for i in range(2, len(nums)):
            if nums[index - 2] != nums[i]:
                nums[index] = nums[i]
                index += 1
        return index

``` 
- 加入一个变量记录元素出现次数
