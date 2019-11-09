21. [Merge Two Sorted Lists](https://leetcode-cn.com/problems/merge-two-sorted-lists/)
```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = cur = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 or l2
        return dummy.next
```

22. [Generate Parentheses](https://leetcode-cn.com/problems/generate-parentheses/solution/gua-hao-sheng-cheng-by-leetcode/)
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans = []

        def generate(S="", left=0, right=0):
            if len(S) == 2 * n:
                ans.append(S)
                return
            if left < n:
                generate(S + "(", left + 1, right)
            if right < left:
                generate(S + ")", left, right + 1)

        generate()
        return ans
```



23. [Merge k Sorted Lists](https://leetcode-cn.com/problems/merge-k-sorted-lists/solution/he-bing-kge-pai-xu-lian-biao-by-leetcode/)
```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        self.nodes = []
        head = p = ListNode(0)
        for k in lists:
            while k:
                self.nodes.append(k.val)
                k = k.next
        for x in sorted(self.nodes):
            p.next = ListNode(x)
            p = p.next
        return head.next
```


24. [Swap Nodes in Pairs](https://leetcode-cn.com/problems/swap-nodes-in-pairs)
```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        pre, pre.next = self, head
        while pre.next and pre.next.next:
            a = pre.next
            b = pre.next.next
            pre.next, b.next, a.next = b, a, b.next
            pre = a
        return self.next
```
25. [Reverse Nodes in k-Group](https://leetcode-cn.com/problems/reverse-nodes-in-k-group)
```python
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        dummy = jump = ListNode(-1)
        dummy.next = l = r = head
        while True:
            count = 0
            while r and count < k:
                r = r.next
                count += 1
            if count == k:  # 翻转
                pre, cur = r, l
                for _ in range(k):
                    cur.next, cur, pre = pre, cur.next, cur
                jump.next, jump, l = pre, l, r
            else:
                return dummy.next
```
- 用count变量控制节点翻转范围,注意各种边界条件

26. [Remove Duplicates from Sorted Array](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)
```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        index = 0
        for i in range(1, len(nums)):
            if nums[index] != nums[i]:
                index += 1
                nums[index] = nums[i]
        return index + 1
```
27. [Remove Element](https://leetcode-cn.com/problems/remove-element/)
```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        index = 0
        for i in range(len(nums)):
            if val != nums[i]:
                nums[index] = nums[i]
                index += 1
        return index
```

28. [Implement strStr](https://leetcode-cn.com/problems/implement-strstr)
```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i + len(needle)] == needle:
                return i
        return -1
```
- 暴力破解,更高效的算法有KMP,Boyer-Mooer算法和Rabin-Karp算法

30. [Substring with Concatenation of All Words](https://leetcode-cn.com/problems/substring-with-concatenation-of-all-words/)
```python
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        from collections import Counter
        if not s or not words:
            return []
        one_word = len(words[0])
        all_len = len(words) * one_word
        n = len(s)
        words = Counter(words)
        res = []
        for i in range(0, n - all_len + 1):
            tmp = s[i:i + all_len]
            c_tmp = []
            for j in range(0, all_len, one_word):
                c_tmp.append(tmp[j:j + one_word])
            if Counter(c_tmp) == words:
                res.append(i)
        return res
```

