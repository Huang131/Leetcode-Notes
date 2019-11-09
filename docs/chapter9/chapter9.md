81. [Search in Rotated Sorted Array II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)
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

82. [Remove Duplicates from Sorted List II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)
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

83. [Remove Duplicates from Sorted List](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)
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
84. [Largest Rectangle in Histogram](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)
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

85. [Maximal Rectangle](https://leetcode-cn.com/problems/maximal-rectangle/)
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



86. [Partition List](https://leetcode-cn.com/problems/partition-list/)
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

87. [Scramble String](https://leetcode-cn.com/problems/scramble-string/solution/)
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

88. [Merge Sorted Array](https://leetcode-cn.com/problems/merge-sorted-array/)
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


89. [Gray Code](https://leetcode-cn.com/problems/gray-code/)
```python
class Solution:
    def grayCode(self, n: int) -> List[int]:
        return [i ^ i >> 1 for i in range(1 << n)]
```

90. [Subsets II](https://leetcode-cn.com/problems/subsets-ii/solution/tong-ji-pin-ci-zai-zu-he-fa-by-mai-mai-mai-mai-zi/)
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