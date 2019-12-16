81 [Search in Rotated Sorted Array II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)
>假设按照升序排序的数组在预先未知的某个点上进行了旋转。(例如，数组 [0,0,1,2,2,5,6] 可能变为 [2,5,6,0,0,1,2])。
编写一个函数来判断给定的目标值是否存在于数组中。若存在返回 true，否则返回 false。

示例:
```
输入: nums = [2,5,6,0,0,1,2], target = 0
输出: true

输入: nums = [2,5,6,0,0,1,2], target = 3
输出: false
```
进阶:
- 这是[搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/description/)的延伸题目，本题中的 nums 可能包含重复元素。
- 这会影响到程序的时间复杂度吗？会有怎样的影响，为什么？

```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        first, last = 0, len(nums)
        while first != last:
            mid = first + (last - first) // 2
            if nums[mid] == target:
                return True
            elif nums[first] < nums[mid]:
                if nums[first] <= target and target < nums[mid]:
                    last = mid
                else:
                    first = mid + 1
            elif nums[first] > nums[mid]:
                if nums[mid] < target and target <= nums[last - 1]:
                    first = mid + 1
                else:
                    last = mid
            else:
                first += 1
        return False
```

82 [Remove Duplicates from Sorted List II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)
>给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中 没有重复出现 的数字。

示例:
```
输入: 1->2->3->3->4->4->5
输出: 1->2->5

输入: 1->1->1->2->3
输出: 2->3
```

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        prev = dummy = ListNode(-1)
        dummy.next = head
        while head and head.next:
            if head.val == head.next.val:
                while head and head.next and head.val == head.next.val:
                    head = head.next
                head = head.next
                prev.next = head
            else:
                prev = prev.next
                head = head.next
        return dummy.next
```

83 [Remove Duplicates from Sorted List](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

>给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。


示例:
```
输入: 1->1->2
输出: 1->2

输入: 1->1->2->3->3
输出: 1->2->3
```

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        prev = head
        while prev:
            while prev.next and prev.val == prev.next.val:
                prev.next = prev.next.next
            prev = prev.next
        return head
```

84 [Largest Rectangle in Histogram](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

>给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
求在该柱状图中，能够勾勒出来的矩形的最大面积。

示例:
```
输入: [2,1,5,6,2,3]
输出: 10
```

```python
class Solution:
    def largestRectangleArea(self, heights):
        heights.append(0)
        stack = [-1]
        ans = 0
        for k, height in enumerate(heights):
            while height < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = k - stack[-1] - 1
                ans = max(ans, h * w)
            stack.append(k)
        heights.pop()
        return ans
```

85 [Maximal Rectangle](https://leetcode-cn.com/problems/maximal-rectangle/)

>给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。

示例:
```
输入:
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
输出: 6
```

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        # 将每一行看做一个二进制数,然后转化为一个整数
        nums = [int(''.join(row), base=2) for row in matrix]
        ans, N = 0, len(nums)
        # 遍历所有行
        for i in range(N):
            j, num = i, nums[i]
            # 将第i行和接下来所有行做与运算,保留二进制中所有行均有'1'的位置
            while j < N:
                # 经过与运算后，num转化为二进制中的1，表示从i到j行，可以组成一个矩形的那几列
                num = num & nums[j]
                if not num:
                    break
                l, curnum = 0, num  # l表示有效宽度
                while curnum:
                    l += 1
                    curnum = curnum & (curnum << 1)
                ans = max(ans, l * (j - i + 1))
                j += 1
        return ans
```

86 [Partition List](https://leetcode-cn.com/problems/partition-list/)

>给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于 x 的节点都在大于或等于 x 的节点之前。
你应当保留两个分区中每个节点的初始相对位置。

示例:
```
输入: head = 1->4->3->2->5->2, x = 3
输出: 1->2->2->4->3->5
```

```python
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        left_dummy = ListNode(-1)
        right_dummy = ListNode(-1)

        left_cur = left_dummy
        right_cur = right_dummy
        cur = head
        while cur:
            if cur.val < x:
                left_cur.next = cur
                left_cur = cur
            else:
                right_cur.next = cur
                right_cur = cur
            cur = cur.next
        left_cur.next = right_dummy.next
        right_cur.next = None
        return left_dummy.next
```

87 [Scramble String](https://leetcode-cn.com/problems/scramble-string)

>给定一个字符串 s1，我们可以把它递归地分割成两个非空子字符串，从而将其表示为二叉树。

示例:
```
输入: s1 = "great", s2 = "rgeat"
输出: true

输入: s1 = "abcde", s2 = "caebd"
输出: false
```

```python
import functools
class Solution:
    @functools.lru_cache(None)
    def isScramble(self, s1: str, s2: str) -> bool:
        if s1 == s2:
            return True
        if sorted(s1) != sorted(s2):
            return False
        for i in range(1, len(s1)):
            if self.isScramble(s1[:i], s2[:i]) and self.isScramble(
                    s1[i:], s2[i:]):
                return True
            if self.isScramble(s1[:i], s2[-i:]) and self.isScramble(
                    s1[i:], s2[:-i]):
                return True
        return False
```

88 [Merge Sorted Array](https://leetcode-cn.com/problems/merge-sorted-array/)

>给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。
说明:
- 初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
- 你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。

示例:
```
输入:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

输出: [1,2,2,3,5,6]
```

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int],
              n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        while n > 0:
            if m and nums1[m - 1] > nums2[n - 1]:
                nums1[n + m - 1], m = nums1[m - 1], m - 1
            else:
                nums1[n + m - 1], n = nums2[n - 1], n - 1
```


89 [Gray Code](https://leetcode-cn.com/problems/gray-code/)

>格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数的差异。
给定一个代表编码总位数的非负整数 n，打印其格雷编码序列。格雷编码序列必须以 0 开头。

示例:
```
输入: 2
输出: [0,1,3,2]
解释:
00 - 0
01 - 1
11 - 3
10 - 2

对于给定的 n，其格雷编码序列并不唯一。
例如，[0,2,3,1] 也是一个有效的格雷编码序列。

00 - 0
10 - 2
11 - 3
01 - 1



输入: 0
输出: [0]
解释: 我们定义格雷编码序列必须以 0 开头。
     给定编码总位数为 n 的格雷编码序列，其长度为 2n。当 n = 0 时，长度为 20 = 1。
     因此，当 n = 0 时，其格雷编码序列为 [0]。
```

```python
class Solution:
    def grayCode(self, n: int) -> List[int]:
        return [i ^ i >> 1 for i in range(1 << n)]
```

90 [Subsets II](https://leetcode-cn.com/problems/subsets-ii)

>给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
说明：解集不能包含重复的子集。


示例:
```
输入: [1,2,2]
输出:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
```

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        from collections import Counter
        c = Counter(nums)
        res = [[]]
        for i, v in c.items():
            temp = res.copy()
            for j in res:
                temp.extend(j + [i] * (k + 1) for k in range(v))
            res = temp
        return res
```