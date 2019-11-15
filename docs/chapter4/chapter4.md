31 [Next Permutation](https://leetcode-cn.com/problems/next-permutation/)

>实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。
必须<font color='red'>原地</font>修改，只允许使用<font color='red'>额外常数空间</font>。
以下是一些例子，输入位于左侧列，其相应输出位于右侧列
```
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1
```

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        i = n - 1
        # 从右到左遍历,找到交换点索引
        while i > 0 and nums[i - 1] >= nums[i]:
            i -= 1
        if i == 0:
            return nums.reverse()  # 如果完全递减,将数字排成最小的序列
        else:
            nums[i:] = sorted(nums[i:])  # 交换点后的数字进行升序排列
            for j in range(i, n):
                if nums[j] > nums[i - 1]:
                    nums[i - 1], nums[j] = nums[j], nums[i - 1]  # 交换
                    break
```

32 [Longest Valid Parentheses](https://leetcode-cn.com/problems/longest-valid-parentheses/)

>给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度。
<br>
示例:

```
输入: "(()"
输出: 2
解释: 最长有效括号子串为 "()"

输入: ")()())"
输出: 4
解释: 最长有效括号子串为 "()()"
```

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        maxLength = 0
        stack = [-1]

        for k, ch in enumerate(s):
            if ch == '(':
                stack.append(k)
            else:
                stack.pop()
                if stack:
                    maxLength = max(maxLength, k - stack[-1])
                else:
                    stack.append(k)
        return maxLength
```
- satck中保存`'('`在字符串`s`中的下标，每次遇到`')'`后，更新maxLength

33 [Search in Rotated Sorted Array](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

>假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
你可以假设数组中不存在重复的元素。
你的算法时间复杂度必须是 O(log n) 级别。 
<br>
示例:

```
输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4

输入: nums = [4,5,6,7,0,1,2], target = 3
输出: -1
```

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        first, last = 0, len(nums)
        while first != last:
            mid = first + (last - first) // 2
            if nums[mid] == target:
                return mid
            elif nums[first] <= nums[mid]:
                if nums[first] <= target and target < nums[mid]:
                    last = mid
                else:
                    first = mid + 1
            else:
                if nums[mid] < target and target <= nums[last - 1]:
                    first = mid + 1
                else:
                    last = mid
        return -1
```
- 二分查找，注意判断target是落在哪个区间，再次使用二分查找。

34 [Find First and Last Position of Element in Sorted Array](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

>给定一个按照升序排列的整数数组`nums`，和一个目标值`target`。找出给定目标值在数组中的开始位置和结束位置。
你的算法时间复杂度必须是 O(log n) 级别。
如果数组中不存在目标值，返回 [-1, -1]。
<br>
示例:

```
输入: nums = [5,7,7,8,8,10], target = 8
输出: [3,4]

输入: nums = [5,7,7,8,8,10], target = 6
输出: [-1,-1]
```

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # 注意条件x >= A[mid],会导致right从右边逼近x
        def binarySearchLeft(A, x):
            left, right = 0, len(A) - 1
            while left <= right:
                mid = (left + right) // 2
                if x > A[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
            return left

        # 注意条件x >= A[mid],会导致right从右边逼近x
        def binarySearchRight(A, x):
            left, right = 0, len(A) - 1
            while left <= right:
                mid = (left + right) // 2
                if x >= A[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
            return right

        left, right = binarySearchLeft(nums, target), binarySearchRight(
            nums, target)
        return (left, right) if left <= right else [-1, -1]
```

35 [Search Insert Position](https://leetcode-cn.com/problems/search-insert-position/)

>给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
你可以假设数组中无重复元素。 
<br>
示例:

```
输入: [1,3,5,6], 5
输出: 2

输入: [1,3,5,6], 2
输出: 1

输入: [1,3,5,6], 7
输出: 4

输入: [1,3,5,6], 0
输出: 0
```
- 调用库函数
```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        return bisect.bisect_left(nums, target, 0, len(nums))
```
- 二分查找
```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left
```

36 [Valid Sudoku](https://leetcode-cn.com/problems/valid-sudoku/)

>判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是否有效即可。
- 数字 1-9 在每一行只能出现一次。
- 数字 1-9 在每一列只能出现一次。
- 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
<br>
示例:

```
输入:
[
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
输出: true
```

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row = [[x for x in y if x != '.'] for y in board]
        col = [[x for x in y if x != '.'] for y in zip(*board)]
        pal = [[
            board[i + m][j + n] for m in range(3) for n in range(3)
            if board[i + m][j + n] != '.'
        ] for i in (0, 3, 6) for j in (0, 3, 6)]
        return all(len(set(x)) == len(x) for x in (*row, *col, *pal))
```
- 利用 set 检查每个区块中是否有重复数字
- pal 取区块的遍历方式是利用 i，j 遍历每个宫格左上角位置，然后取 3*3 区块

37 [Sudoku Solver](https://leetcode-cn.com/problems/sudoku-solver/)

>编写一个程序，通过已填充的空格来解决数独问题。
一个数独的解法需遵循如下规则：
- 数字 1-9 在每一行只能出现一次。
- 数字 1-9 在每一列只能出现一次。
- 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
空白格用 '.' 表示。

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row = [set(range(1, 10)) for _ in range(9)]  # 行剩余可用数字
        col = [set(range(1, 10)) for _ in range(9)]  # 列剩余可用数字
        block = [set(range(1, 10)) for _ in range(9)]  # 块剩余可用数字

        empty = []  # 收集需填数位置

        for i in range(9):
            for j in range(9):
                if board[i][j] != ".":  # 更新可用数字
                    val = int(board[i][j])
                    row[i].remove(val)
                    col[j].remove(val)
                    block[(i // 3) * 3 + j // 3].remove(val)
                else:
                    empty.append((i, j))

        def backtrack(iter=0):
            if iter == len(empty):  # 处理完empty代表找到了答案
                return True
            i, j = empty[iter]
            b = (i // 3) * 3 + j // 3
            for val in row[i] & col[j] & block[b]:  # row、col、block中均有val可用
                row[i].remove(val)
                col[j].remove(val)
                block[b].remove(val)
                board[i][j] = str(val)
                if backtrack(iter + 1):
                    return True
                row[i].add(val)  # 回溯
                col[j].add(val)
                block[b].add(val)
            return False

        backtrack()
```

38 [Count and Say](https://leetcode-cn.com/problems/count-and-say/)

>报数序列是一个整数序列，按照其中的整数的顺序进行报数，得到下一个数。其前五项如下：
```
1.     1
2.     11
3.     21
4.     1211
5.     111221
``` 
`1` 被读作  `"one 1"`  (`"一个一"`) , 即 11。  
`11` 被读作 `"two 1s"` (`"两个一"`）, 即 21。  
`21` 被读作 `"one 2"`, `"one 1"` （`"一个二"` ,  `"一个一"`) , 即 1211。  
给定一个正整数 n（1 ≤ n ≤ 30），输出报数序列的第 n 项。  
注意：整数顺序将表示为一个字符串。
<br>
示例:

```
输入: 1
输出: "1"

输入: 4
输出: "1211"
```

```python
class Solution:
    def countAndSay(self, n: int) -> str:
        return '1' * (n is 1) or re.sub(r'(.)\1*', lambda m: str(len(m.group())) + m.group(1),
                                                                        self.countAndSay(n - 1))
```
- re.sub(正则,替换字符串或函数,被替换字符串,是否区分大小写)
- '.'可匹配任意一个除了'\n'的字符。(.) 匹配任意一个除了\n的字符并把这个匹配结果放进第一组。(.)\1 匹配一个任意字符的二次重复并把那个字符放入数组。(.)\1* 匹配一个任意字符的多次重复并把那个字符放入数组
- group(default=0)可以取匹配文本。group(1)取第一个括号内的文本

39 [Combination Sum](https://leetcode-cn.com/problems/combination-sum/)

>给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
candidates 中的数字可以无限制重复被选取。
说明：
- 所有数字（包括 target）都是正整数。
- 解集不能包含重复的组合。

<br>
示例:

```
输入: candidates = [2,3,6,7], target = 7,
所求解集为:
[
  [7],
  [2,2,3]
]

输入: candidates = [2,3,5], target = 8,
所求解集为:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, [], res)
        return res

    def dfs(self, nums, target, index, path, res):
        if target < 0:
            return
        if target == 0:
            res.append(path)
            return
        for i in range(index, len(nums)):
            self.dfs(nums, target - nums[i], i, path + [nums[i]], res)
```
- dfs

40 [Combination Sum II](https://leetcode-cn.com/problems/combination-sum-ii/)

>给定一个数组`candidates` 和一个目标数`target` ，找出`candidates`中所有可以使数字和为`target`的组合。
`candidates`中的<font color='red'>每个数字在每个组合中只能使用一次</font>。
说明：
- 所有数字（包括目标数）都是正整数。
- 解集不能包含重复的组合。 
<br>

示例:
```
输入: candidates = [10,1,2,7,6,1,5], target = 8,
所求解集为:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]

输入: candidates = [2,5,2,1,2], target = 5,
所求解集为:
[
  [1,2,2],
  [5]
]
```

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        size = len(candidates)
        if size == 0:
            return []
        candidates.sort()
        res = []

        self.__dfs(candidates, size, 0, [], target, res)
        return res

    def __dfs(self, candidates, size, start, path, residue, res):
        if residue == 0:
            res.append(path[:])
            return
        for index in range(start, size):
            if candidates[index] > residue:
                break
            # 剪枝的前提是数组升序排序
            if index > start and candidates[index - 1] == candidates[index]:
                continue

            path.append(candidates[index])
            # 传入index+1,当前元素不能被重复使用
            self.__dfs(
                candidates, size, index + 1, path, residue - candidates[index], res
            )
            path.pop()
```
