21 [Merge Two Sorted Lists](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

>将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 
<br>
示例:
```
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

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

- 1 or 2 结果为1,False or 2 结果为2
- None and 2 结果为None,True and Number 结果为Number

22 [Generate Parentheses](https://leetcode-cn.com/problems/generate-parentheses/)

>给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
例如，给出 n = 3，生成结果为：
```
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```

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
- 生成结果中,左括号和右括号数目均为n
- 如果left < n,我们就可以放左括号;如果left < right,我们就可以放右括号
- 时间复杂度：$O(\dfrac{4^n}{\sqrt{n}})$,在回溯过程中，每个有效序列最多需要 n 步。
- 空间复杂度：$O(\dfrac{4^n}{\sqrt{n}})$,如上所述，并使用 O(n) 的空间来存储序列。

23 [Merge k Sorted Lists](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

>合并 k 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。 
<br>
示例:
```
输入:
[
  1->4->5,
  1->3->4,
  2->6
]
输出: 1->1->2->3->4->4->5->6
```

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        self.nodes = [] # 使用 [],对node.val进行排序
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

- 遍历所有节点,将所有节点的值放到一个数组。将该数组进行排序,最后遍历数组创建一个新的链表
- 时间复杂度 $O(N\log{N})$
    - 遍历所有值需花费O(N)的时间
    - 一个稳定的排序算法花费$O(N\log{N})$的时间
    - 遍历同时创建新的有序链表花费O(N)的时间
- 空间复杂度 O(N)
    - 排序花费O(N)空间（这取决于你选择的算法）
    - 创建一个新的链表花费O(N)的空间


24 [Swap Nodes in Pairs](https://leetcode-cn.com/problems/swap-nodes-in-pairs)

>给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。 
<br>
示例:
```
给定 1->2->3->4, 你应该返回 2->1->4->3.
```

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

- 先添加一个空头,在进行交换


25 [Reverse Nodes in k-Group](https://leetcode-cn.com/problems/reverse-nodes-in-k-group)

>给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
k 是一个正整数，它的值小于或等于链表的长度。
如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
<br>
示例:
```
给定这个链表：1->2->3->4->5
当 k = 2 时，应当返回: 2->1->4->3->5
当 k = 3 时，应当返回: 3->2->1->4->5
说明:
你的算法只能使用常数的额外空间。
你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
```

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

26 [Remove Duplicates from Sorted Array](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

>给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
<br>
示例:
```
给定 nums = [0,0,1,1,1,2,2,3,3,4],
函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。
你不需要考虑数组中超出新长度后面的元素。
```

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

- index,i分别是快慢指针,只要nums[index] == nums[i],我们就通过增加i跳过重复项。
- 当nums[index] != nums[i],把nums[i]向前复制到`index+1`位置


27 [Remove Element](https://leetcode-cn.com/problems/remove-element/)

>给定一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，返回移除后数组的新长度。
不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
<br>
示例:
```
给定 nums = [3,2,2,3], val = 3,
函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。
你不需要考虑数组中超出新长度后面的元素。
```

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
- 双指针,和上题解法十分相似。
- 时间复杂度:O(n) 假设数组总共有 n 个元素，i 和 j 至少遍历 2n 步。
- 空间复杂度：O(1)

28 [Implement strStr](https://leetcode-cn.com/problems/implement-strstr)

>实现 strStr() 函数。
给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。
<br>
示例:
```
输入: haystack = "hello", needle = "ll"
输出: 2
```

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i + len(needle)] == needle:
                return i
        return -1
```
- 暴力破解,更高效的算法有KMP,Boyer-Mooer算法和Rabin-Karp算法

29 [Divide Two Integers](https://leetcode-cn.com/problems/divide-two-integers/)

>给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。
返回被除数 dividend 除以除数 divisor 得到的商。
<br>
示例:

```
输入: dividend = 10, divisor = 3
输出: 3

输入: dividend = 7, divisor = -3
输出: -2

```

```python
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        a, b, r, t = abs(dividend), abs(divisor), 0, 1
        while a >= b or t > 1:
            if a >= b:
                r += t
                a -= b
                t += t
                b += b
            else:
                t >>= 1
                b >>= 1
        return min((-r, r)[dividend ^ divisor >= 0], (1 << 31) - 1)
```

- 让被除数不断减去除数，直到不够减。每次减完后除数翻倍，并记录当前为初始除数的几倍（用 t 表示倍数 time），若发现不够减且 t 不为 1 则让除数变为原来的一半， t 也减半
- `a << b` 相当于 `a * 2**b，a >> b 相当于 a // 2**b`
- 异或操作 ^ 可以判断俩数字是否异号
- `positive = (dividend < 0) is (divisor < 0)` 也可以判断正负号


30 [Substring with Concatenation of All Words](https://leetcode-cn.com/problems/substring-with-concatenation-of-all-words/)

>给定一个字符串 s 和一些长度相同的单词 words。找出 s 中恰好可以由 words 中所有单词串联形成的子串的起始位置。
注意子串要与 words 中的单词完全匹配，中间不能有其他字符，但不需要考虑 words 中单词串联的顺序。
<br>
示例:

```
输入：
  s = "barfoothefoobarman",
  words = ["foo","bar"]
输出：[0,9]
解释：
从索引 0 和 9 开始的子串分别是 "barfoo" 和 "foobar" 。
输出的顺序不重要, [9,0] 也是有效答案。
```

```python
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        from collections import Counter
        if not s or not words:
            return []
        one_word = len(words[0])
        all_len = len(words) * one_word # words中所有单词组成长度
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

- 因为单词长度固定的，我们可以计算出截取字符串的单词个数是否和 words 里相等
