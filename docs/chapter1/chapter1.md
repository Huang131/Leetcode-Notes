1. [Two Sum](https://leetcode-cn.com/problems/two-sum)
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, n in enumerate(nums):
            if n in d: return [d[n], i]
            d[target - n] = i
```
- 用字典记录 { 需要的值:当前索引 } 时间复杂度:O(n)


2. [Add Two Number](https://leetcode-cn.com/problems/add-two-numbers)
```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode, carry=0) -> ListNode:
        if not (l1 or l2):
            return ListNode(1) if carry else None
        l1, l2 = l1 or ListNode(0), l2 or ListNode(0)
        val = l1.val + l2.val + carry
        l1.val, l1.next = val % 10, self.addTwoNumbers(l1.next,l2.next,al > 9)
        return l1
```
- 用carry记录进位


3. [Longest Substring Without Repeating Characters](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters)
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start = maxLength = 0
        useChar = {}  # 保存非重复字符串
        for i, ch in enumerate(s):
            if ch in useChar and start <= useChar[ch]:
                start = useChar[ch] + 1  # start 移动到字典重复字符所在的后一位
            else:
                maxLength = max(maxLength, i - start + 1)
            useChar[ch] = i  # 当前遍历字符所在索引存入字典

        return maxLength
```
- 滑动窗口问题

