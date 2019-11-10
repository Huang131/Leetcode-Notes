1 [Two Sum](https://leetcode-cn.com/problems/two-sum)

>给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那两个整数，并返回他们的数组下标。
你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。
<br>
示例:
```
给定 nums = [2, 7, 11, 15], target = 9
因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
```

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, n in enumerate(nums):
            if n in d: return [d[n], i]
            d[target - n] = i
```
- 用字典记录 { 需要的值:当前索引 } 时间复杂度:O(n)


2 [Add Two Number](https://leetcode-cn.com/problems/add-two-numbers)

>给出两个<strong>非空</strong>的链表用来表示两个非负的整数。其中，它们各自的位数是按照<strong>逆序</strong>的方式存储的，并且它们的每个节点只能存储 <strong>一位 </strong>数字。
如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
您可以假设除了数字 0 之外，这两个数都不会以 0 开头。
<br>
示例:
```
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
```

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode, carry=0) -> ListNode:
        if not (l1 or l2):
            return ListNode(1) if carry else None
        l1, l2 = l1 or ListNode(0), l2 or ListNode(0)
        val = l1.val + l2.val + carry
        l1.val, l1.next = val % 10, self.addTwoNumbers(l1.next,l2.next,val > 9)
        return l1
```
- False or Number => Number
- 用carry记录进位


3 [Longest Substring Without Repeating Characters](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters)

>给定一个字符串，请你找出其中不含有重复字符的<strong>最长子串</strong>的长度。
<br>
示例:
```
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

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


4 [MedianofTwoSortedArrays](https://leetcode-cn.com/problems/median-of-two-sorted-arrays)

>给定两个大小为`m `和 `n` 的有序数组 `nums1 `和 `nums2`。
请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
你可以假设 `nums1` 和 `nums2` 不会同时为空。
<br>
示例:
```
nums1 = [1, 2]
nums2 = [3, 4]
则中位数是 (2 + 3)/2 = 2.5
```

```python
class Solution:
    def findKth(self, nums1: List[int], nums2: List[int], k: int):

        # 边界条件
        if len(nums1) > len(nums2):
            return self.findKth(nums2, nums1, k)
        elif len(nums1) == 0:
            return nums2[k - 1]
        elif k == 1:
            return min(nums1[0], nums2[0])

        first = min(k // 2, len(nums1))
        second = k - first

        if nums1[first - 1] < nums2[second - 1]:
            return self.findKth(nums1[first:], nums2, k - first)
        elif nums1[first - 1] > nums2[second - 1]:
            return self.findKth(nums1, nums2[second:], k - second)
        else:
            return nums1[first - 1]

    def findMedianSortedArrays(self, nums1: List[int],nums2: List[int]) -> float:

        length = len(nums1) + len(nums2)

        if (length & 0x1):
            return self.findKth(nums1, nums2, length // 2 + 1)
        else:
            return ((self.findKth(nums1, nums2, length // 2) +
                     self.findKth(nums1, nums2, length // 2 + 1)) / 2)
```
- 从K入手,利用有序这个条件,每次排除一半的元素。假设A和B的元素个数都大于K/2,因为K的奇偶性不影响结论,假设K为偶数。将A[k/2-1]与B[k/2-1]进行比较：
    * A[k/2-1] == B[k/2-1]  直接返回A[k/2-1]或者B[k/2-1]
    * A[k/2-1]  < B[k/2-1]  删除A中的k/2个元素
    * A[k/2-1]  > B[k/2-1]  删除B中的k/2个元素
- 函数终止条件：
    * 当A或B是空时:直接返回B[k-1]或A[k-1]
    * 当k=1时,返回min(A[0],B[0])
    * 当A[k/2-1] == B[k/2-1]时,返回A[k/2-1]或B[k/2-1]



5 [Longest Palindromic Substring](https://leetcode-cn.com/problems/longest-palindromic-substring/)

>给定一个字符串 `s`，找到 `s` 中最长的回文子串。你可以假设 `s` 的最大长度为 1000。
<br>
示例:
```
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。
```

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        r = ''
        for i, j in [(i, j) for i in range(len(s)) for j in (0, 1)]:
            while i > -1 and i + j < len(s) and s[i] == s[i + j]:
                i, j = i - 1, j + 2
            r = max(r, s[i + 1:i + j], key=len)
        return '' if not s else r
```
- 字符串的中心可能是一个字符也可能是两个字符,例如字符串abcbd,回文子串bcb,中心字符为c。字符串abccbd,回文子串bccb,中心字符为cc。所以j的取值为0或1。
- i 遍历字符串中的每一个字符,通过j的取值判断回文子串的中心字符取值情况。j为0时,子串假设为一个中心字符。j为1时，子串假设为两个中心字符。
- r保存每次确定中心字符情况后的最大子串


6 [ZigZag Conversion](https://leetcode-cn.com/problems/zigzag-conversion/)

>将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。
比如输入字符串为 `"LEETCODEISHIRING"` 行数为 3 时，排列如下：
```
L   C   I   R
E T O E S I I G
E   D   H   N
```
之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如：`"LCIRETOESIIGEDHN"`。
请你实现这个将字符串进行指定行数变换的函数：
`string convert(string s, int numRows);`
<br>
示例:
```
输入: s = "LEETCODEISHIRING", numRows = 4
输出: "LDREOEIIECIHNTSG"
解释:
L     D     R
E   O E   I I
E C   I H   N
T     S     G
```

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or numRows >= len(s):
            return s
        L = [''] * numRows
        index, step = 0, 1
        for x in s:
            L[index] += x
            if index == 0:
                step = 1  # 正向
            elif index == numRows - 1:
                step = -1  # 反向
            index += step
        return ''.join(L)
```
- step控制前进方向,最后把numRowsa行字符串合并


7 [Reverse Integer](https://leetcode-cn.com/problems/reverse-integer/)

>给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。
<br>
示例:
```
输入: -123
输出: -321
```

```python
class Solution:
    def reverse(self, x: int) -> int:
        # x // max(1, abs(x)) 相当于cmp函数
        r = x // max(1, abs(x)) * int(str(abs(x))[::-1])
        return r if r.bit_length() < 32 or r == -2**31 else 0
```

- x // max(1, abs(x))意味着 0：x为0， 1：x为正， -1：x为负，相当于被废弃的函数cmp
- [::-1]代表序列反转
- 2^31 和 -2^31 的比特数为32，其中正负号占用了一位
- 32位整数范围 [−2^31, 2^31 − 1] 中正数范围小一个是因为0的存在


8 [String to Integer(atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi)

>你来实现一个 atoi 函数，使其能将字符串转换成整数。
首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。
当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。
该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。
注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。
在任何情况下，若函数不能进行有效的转换时，请返回 0。
说明：
假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。
<br>
示例:
```
输入: "4193 with words"
输出: 4193
解释: 转换截止于数字 '3' ，因为它的下一个字符不为数字。
```

```python
class Solution:
    def myAtoi(self, str: str) -> int:
        return max(
            min(int(*re.findall('^[\+\-]?\d+', str.lstrip())), 2**31 - 1),
            -2**31)
```
- 使用正则表达式 ^：匹配字符串开头，[\+\-]：代表一个+字符或-字符，?：前面一个字符可有可无，\d：一个数字，+：前面一个字符的一个或多个，\D：一个非数字字符，*：前面一个字符的0个或多个
- max(min(数字, 2 ** 31 - 1), -2 ** 31) 用来防止结果越界

9 [Palindrome Number](https://leetcode-cn.com/problems/palindrome-number/)

>判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
<br>
示例:
```
输入: -121
输出: false
解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
```

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        return str(x) == str(x)[::-1]
```

10 [Regular Expression Matching](https://leetcode-cn.com/problems/regular-expression-matching/)

>给你一个字符串`s`和一个字符规律`p`，请你来实现一个支持 `'.'` 和 `'*'` 的正则表达式匹配。
```
'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素
```
所谓匹配，是要涵盖<strong>整个</strong>字符串`s`的，而不是部分字符串。  
说明:
```
    `s`可能为空，且只包含从`a-z`的小写字母。
    `p`可能为空，且只包含从`a-z`的小写字母，以及字符`.` 和`*`。
```
<br>
示例:

```
输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。
```

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        if s == p:
            return True
        if len(p) > 1 and p[1] == '*':  # 下一个字符为*
            if s and (s[0] == p[0] or p[0] == '.'):
                return self.isMatch(s, p[2:]) or self.isMatch(s[1:], p)
            else:
                return self.isMatch(s, p[2:])
        elif s and p and (s[0] == p[0] or p[0] == '.'):
            return self.isMatch(s[1:], p[1:])
        return False
```
- '.' 匹配任意单个字符, '*' 匹配零个或多个前面的那一个元素
- 当模式中第二个字符是'*'时：
    -  如果字符串第一个字符跟模式第一个字符不匹配，则模式后移2个字符，继续匹配。如果字符串第一个字符跟模式第一个字符匹配，可以有3种匹配方式：
        1. 模式后移2位字符,即模式前两位被忽略（*匹配0个字符）
        2. 字符串后移1字符,模式不变,*可以匹配多位
        3. 字符后移1字符,模式后移2字符
    发现情况3可以被情况1和情况2包含,即执行一次情况2,再执行一次情况1。所以情况3不用判断
- 当模式中第二个字符不是'*'时:
    - 如果字符串第一个字符和模式中的第一个字符相匹配,字符串和模式都后移一个字符。
    - 如果字符串第一个字符和模式中的第一个字符不匹配,直接返回False
