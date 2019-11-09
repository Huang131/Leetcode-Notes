31. [Next Permutation](https://leetcode-cn.com/problems/next-permutation/)
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

32. [Longest Valid Parentheses](https://leetcode-cn.com/problems/longest-valid-parentheses/)
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
33. [Search in Rotated Sorted Array](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)
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

35. [Search Insert Position](https://leetcode-cn.com/problems/search-insert-position/)
```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        return bisect.bisect_left(nums, target, 0, len(nums))
```
- 调用库函数
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

36. [Valid Sudoku](https://leetcode-cn.com/problems/valid-sudoku/)
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

37. [Sudoku Solver](https://leetcode-cn.com/problems/sudoku-solver/)
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

38. [Count and Say](https://leetcode-cn.com/problems/count-and-say/)
```python
class Solution:
    def countAndSay(self, n: int) -> str:
        return '1' * (n is 1) or re.sub(
            r'(.)\1*', lambda m: str(len(m.group())) + m.group(1),
            self.countAndSay(n - 1))
```
- re.sub(正则,替换字符串或函数,被替换字符串,是否区分大小写)
- '.'可匹配任意一个除了'\n'的字符。(.) 匹配任意一个除了\n的字符并把这个匹配结果放进第一组。(.)\1 匹配一个任意字符的二次重复并把那个字符放入数组。(.)\1* 匹配一个任意字符的多次重复并把那个字符放入数组
- group(default=0)可以取匹配文本。group(1)取第一个括号内的文本

39. []()
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

40. [Combination Sum II](https://leetcode-cn.com/problems/combination-sum-ii/)
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
