"""Build data/question_bank.json: expected outputs are computed by RUNNING each reference solution.

The questions (statements, tests inputs, hints, reference solutions) are original, written for this project - nothing is
copied from LeetCode/HackerRank/GfG. Re-run to regenerate:  python scripts/build_question_bank.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services.code_runner import LocalSandbox  # noqa: E402

SB = LocalSandbox(timeout=10)
Q = []

TREE_BUILD = '''
from collections import deque
class Node:
    def __init__(self, v):
        self.val, self.left, self.right = v, None, None
def build(tokens):
    if not tokens or tokens[0] == "null":
        return None
    root = Node(int(tokens[0])); q = deque([root]); i = 1
    while q and i < len(tokens):
        node = q.popleft()
        if i < len(tokens) and tokens[i] != "null":
            node.left = Node(int(tokens[i])); q.append(node.left)
        i += 1
        if i < len(tokens) and tokens[i] != "null":
            node.right = Node(int(tokens[i])); q.append(node.right)
        i += 1
    return root
'''

LIST_BUILD = '''
class ListNode:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt
def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head
def to_list(head):
    out = []
    while head:
        out.append(head.val); head = head.next
    return out
'''


def add(id, title, topic, difficulty, statement, constraints, input_format, output_format, inputs, sol,
        time_c, space_c, hints, styles=("service", "product", "startup"), kind="code"):
    Q.append(dict(id=id, title=title, kind=kind, topic=topic, difficulty=difficulty, company_styles=list(styles),
                  statement=statement, constraints=constraints, input_format=input_format,
                  output_format=output_format, inputs=inputs, reference_solution=sol.strip() + "\n",
                  expected_time_complexity=time_c, expected_space_complexity=space_c, hints=hints,
                  source="original - PrepWise curated (inspired by common interview patterns)"))


# ------------------------------------------------------------------ arrays
add("arr-001", "Best Streak of Daily Scores", "arrays", "medium",
    "A student logs a score change every day (positive or negative). Find the largest total change over any "
    "run of consecutive days (at least one day).",
    ["1 <= n <= 10^5", "-10^4 <= a[i] <= 10^4"], "Line 1: n. Line 2: n integers.", "One integer: the maximum sum.",
    ["9\n-2 1 -3 4 -1 2 1 -5 4", "1\n5", "3\n-3 -1 -2", "5\n1 2 3 4 5", "6\n0 0 0 0 0 0",
     "8\n3 -10 4 5 -1 2 -20 7", "10\n100 -1 -1 -1 -1 -1 -1 -1 -1 100"],
    '''
import sys
d = sys.stdin.read().split()
n = int(d[0]); a = list(map(int, d[1:1 + n]))
best = cur = a[0]
for x in a[1:]:
    cur = max(x, cur + x)
    best = max(best, cur)
print(best)
''', "O(n)", "O(1)",
    ["What is the best sum of a run that must END at day i?", "This is Kadane's algorithm (a 1-D DP).",
     "Keep cur = max(a[i], cur + a[i]) and track the maximum cur seen."])

add("arr-002", "Budget Pair", "arrays", "easy",
    "Given prices of n items and a budget, find two different items whose prices add up exactly to the budget. "
    "Print their 1-based positions i < j. If several pairs exist, choose the one with the smallest j, and then the "
    "smallest i. Print -1 if no pair exists.",
    ["1 <= n <= 10^5", "-10^9 <= price, budget <= 10^9"], "Line 1: n budget. Line 2: n integers.",
    "Two integers i j, or -1.",
    ["4 9\n2 7 11 15", "3 6\n3 2 4", "2 6\n3 3", "3 10\n1 2 3", "1 5\n5", "6 0\n-3 4 3 90 -4 0", "5 8\n4 4 4 4 4"],
    '''
import sys
d = sys.stdin.read().split()
n, t = int(d[0]), int(d[1]); a = list(map(int, d[2:2 + n]))
seen = {}
for j, x in enumerate(a):
    if t - x in seen:
        print(seen[t - x] + 1, j + 1)
        break
    if x not in seen:
        seen[x] = j
else:
    print(-1)
''', "O(n)", "O(n)",
    ["For each item, which price would complete the budget?", "Use a hash map from price to first index.",
     "Scan left to right; before inserting a[j], check whether budget - a[j] is already in the map."])

add("arr-003", "Push Zeros Right", "arrays", "easy",
    "An attendance array uses 0 for absent days. Move every 0 to the end while keeping the relative order of the "
    "non-zero values. Print the resulting array.",
    ["1 <= n <= 10^5"], "Line 1: n. Line 2: n integers.", "The rearranged array, space separated.",
    ["5\n0 1 0 3 12", "1\n0", "3\n1 2 3", "4\n0 0 0 0", "6\n-1 0 0 -2 5 0", "2\n0 7"],
    '''
import sys
d = sys.stdin.read().split()
n = int(d[0]); a = list(map(int, d[1:1 + n]))
w = 0
for x in a:
    if x != 0:
        a[w] = x; w += 1
for i in range(w, n):
    a[i] = 0
print(*a)
''', "O(n)", "O(1)",
    ["Can you do it in one pass without extra arrays?", "Use a write pointer (two pointers).",
     "Copy each non-zero to position w and increment w; fill the rest with zeros."])

add("arr-004", "Team Contribution Products", "arrays", "medium",
    "For each team member i, print the product of everyone else's contribution scores. Division is not allowed.",
    ["2 <= n <= 10^5", "-30 <= a[i] <= 30", "Every answer fits in a 64-bit integer"],
    "Line 1: n. Line 2: n integers.", "n integers: the products.",
    ["4\n1 2 3 4", "2\n5 7", "5\n-1 1 0 -3 3", "3\n0 0 2", "4\n2 2 2 2", "6\n1 -1 1 -1 1 -1"],
    '''
import sys
d = sys.stdin.read().split()
n = int(d[0]); a = list(map(int, d[1:1 + n]))
res = [1] * n
p = 1
for i in range(n):
    res[i] = p; p *= a[i]
p = 1
for i in range(n - 1, -1, -1):
    res[i] *= p; p *= a[i]
print(*res)
''', "O(n)", "O(1) extra (excluding output)",
    ["What is the answer in terms of the elements to the left and to the right?",
     "Use prefix and suffix products.",
     "First pass stores the left product at each index; second pass multiplies by a running right product."])

add("arr-005", "Meeting Room Blocks", "arrays", "medium",
    "Given n meeting time intervals [start, end], merge all overlapping or touching intervals and print the merged "
    "blocks in increasing order of start time.",
    ["1 <= n <= 10^4", "0 <= start <= end <= 10^6"], "Line 1: n. Next n lines: start end.",
    "One merged interval per line: start end.",
    ["4\n1 3\n2 6\n8 10\n15 18", "2\n1 4\n4 5", "1\n5 7", "3\n1 10\n2 3\n4 5", "3\n6 8\n1 2\n3 4",
     "5\n1 2\n2 3\n3 4\n4 5\n10 11"],
    '''
import sys
d = list(map(int, sys.stdin.read().split()))
n = d[0]
iv = sorted((d[1 + 2 * i], d[2 + 2 * i]) for i in range(n))
merged = []
for s, e in iv:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])
for s, e in merged:
    print(s, e)
''', "O(n log n)", "O(n)",
    ["Would sorting the intervals help?", "Sort by start time, then sweep.",
     "Extend the last merged block while the next start <= its end; otherwise start a new block."])

add("arr-006", "Best k-Day Window", "arrays", "easy",
    "Given daily step counts for n days, find the maximum total steps over any k consecutive days.",
    ["1 <= k <= n <= 10^5"], "Line 1: n k. Line 2: n integers.", "One integer.",
    ["5 2\n1 3 -1 2 5", "1 1\n-5", "6 3\n2 1 5 1 3 2", "4 4\n1 1 1 1", "5 1\n-1 -2 -3 -4 -5", "7 2\n0 0 0 0 0 0 0"],
    '''
import sys
d = list(map(int, sys.stdin.read().split()))
n, k = d[0], d[1]; a = d[2:2 + n]
cur = sum(a[:k]); best = cur
for i in range(k, n):
    cur += a[i] - a[i - k]
    best = max(best, cur)
print(best)
''', "O(n)", "O(1)",
    ["Recomputing each window sum is O(n*k). Can you reuse work?", "Use a fixed-size sliding window.",
     "Add the entering element and subtract the leaving one as the window moves."])

# ----------------------------------------------------------------- strings
add("str-001", "Bracket Validator", "strings", "easy",
    "A code formatter must check whether a string of brackets ()[]{} is balanced: every opening bracket is closed "
    "by the same type in the correct order. An empty string is balanced.",
    ["0 <= length <= 10^5"], "One line containing the bracket string (may be empty).", "YES or NO.",
    ["()[]{}", "(]", "([{}])", "(", "", "{[()()]}[", ")("],
    '''
import sys
s = sys.stdin.read().strip()
pairs = {")": "(", "]": "[", "}": "{"}
st = []
ok = True
for ch in s:
    if ch in "([{":
        st.append(ch)
    elif not st or st.pop() != pairs[ch]:
        ok = False
        break
print("YES" if ok and not st else "NO")
''', "O(n)", "O(n)",
    ["Which bracket must be closed first?", "Use a stack.",
     "Push openers; on a closer, pop and check the type; the stack must be empty at the end."])

add("str-002", "Unique Character Run", "strings", "medium",
    "Find the length of the longest substring of s that contains no repeated characters.",
    ["1 <= |s| <= 10^5", "s contains printable ASCII characters without spaces"], "One line: s.", "One integer.",
    ["abcabcbb", "bbbbb", "pwwkew", "a", "abcdef", "abba", "dvdf"],
    '''
import sys
s = sys.stdin.read().strip()
last = {}
best = start = 0
for i, ch in enumerate(s):
    if ch in last and last[ch] >= start:
        start = last[ch] + 1
    last[ch] = i
    best = max(best, i - start + 1)
print(best)
''', "O(n)", "O(min(n, alphabet))",
    ["Think of a window that only grows while characters are unique.", "Sliding window with a hash map of last seen positions.",
     "When s[i] was seen inside the window, move the window start to last[s[i]] + 1."])

add("str-003", "Anagram Buckets", "strings", "medium",
    "Group the given words so that words in the same bucket are anagrams of each other. Print the number of buckets.",
    ["1 <= n <= 10^4", "1 <= word length <= 100, lowercase letters"], "Line 1: n. Line 2: n words.", "One integer.",
    ["6\neat tea tan ate nat bat", "1\na", "3\nabc bca cab", "4\nab ba abc cba", "3\nx y z",
     "5\nlisten silent enlist google gogole"],
    '''
import sys
d = sys.stdin.read().split()
n = int(d[0])
print(len({"".join(sorted(w)) for w in d[1:1 + n]}))
''', "O(n * k log k)", "O(n * k)",
    ["What do two anagrams have in common?", "Use a canonical key such as the sorted word or a letter count.",
     "Insert every key into a set and print its size."])

add("str-004", "Almost Palindrome", "strings", "medium",
    "Decide whether s can become a palindrome after deleting at most one character.",
    ["1 <= |s| <= 10^5", "lowercase letters"], "One line: s.", "YES or NO.",
    ["aba", "abca", "abc", "a", "deeee", "racecar", "abcdef", "cbbcc"],
    '''
import sys
s = sys.stdin.read().strip()
def pal(i, j):
    while i < j:
        if s[i] != s[j]:
            return False
        i += 1; j -= 1
    return True
i, j = 0, len(s) - 1
ans = True
while i < j:
    if s[i] != s[j]:
        ans = pal(i + 1, j) or pal(i, j - 1)
        break
    i += 1; j -= 1
print("YES" if ans else "NO")
''', "O(n)", "O(1)",
    ["Compare characters from both ends.", "Two pointers; you get one 'skip'.",
     "At the first mismatch, try skipping the left or the right character and check the rest."])

add("str-005", "Run-Length Encoder", "strings", "easy",
    "Compress a lowercase string by replacing each run of identical characters with the character followed by the "
    "run length. Example: aaabbc becomes a3b2c1.",
    ["1 <= |s| <= 10^5"], "One line: s.", "The encoded string.",
    ["aaabbc", "a", "abc", "zzzzzzzzzzzz", "aabbaa", "ab"],
    '''
import sys
s = sys.stdin.read().strip()
out = []
i = 0
while i < len(s):
    j = i
    while j < len(s) and s[j] == s[i]:
        j += 1
    out.append(s[i] + str(j - i))
    i = j
print("".join(out))
''', "O(n)", "O(n)",
    ["Process the string run by run.", "Two pointers mark the start and end of a run.",
     "Append char + count for each run; use a list and join to avoid quadratic string building."])

# ------------------------------------------------------------ linked lists
add("ll-001", "Reverse the Playlist", "linked_lists", "easy",
    "A playlist is stored as a singly linked list. Reverse it in place and print the values. Print EMPTY for an "
    "empty list.",
    ["0 <= n <= 10^5"], "Line 1: n. Line 2: n integers (may be empty).", "Reversed values, or EMPTY.",
    ["5\n1 2 3 4 5", "1\n7", "0\n", "2\n1 2", "4\n4 4 1 4", "3\n-1 0 1"],
    LIST_BUILD + '''
import sys
d = sys.stdin.read().split()
n = int(d[0]); head = build(list(map(int, d[1:1 + n])))
prev = None
while head:
    head.next, prev, head = prev, head, head.next
vals = to_list(prev)
print(*vals if vals else ["EMPTY"])
''', "O(n)", "O(1)",
    ["You need to flip every next pointer.", "Iterate with prev/current pointers.",
     "Save next, point current.next to prev, then advance both pointers."])

add("ll-002", "Merge Sorted Queues", "linked_lists", "easy",
    "Two sorted linked lists hold ticket numbers. Merge them into one sorted list and print it. Print EMPTY if both "
    "lists are empty.",
    ["0 <= n, m <= 10^5", "both lists are sorted in non-decreasing order"],
    "Line 1: n m. Line 2: n integers. Line 3: m integers.", "Merged values, or EMPTY.",
    ["3 3\n1 2 4\n1 3 4", "0 0\n\n", "0 1\n\n0", "2 3\n5 6\n1 2 3", "1 1\n2\n2", "4 2\n-5 -1 0 9\n-3 10"],
    LIST_BUILD + '''
import sys
d = sys.stdin.read().split()
n, m = int(d[0]), int(d[1])
a = build(list(map(int, d[2:2 + n]))); b = build(list(map(int, d[2 + n:2 + n + m])))
dummy = tail = ListNode(0)
while a and b:
    if a.val <= b.val:
        tail.next, a = a, a.next
    else:
        tail.next, b = b, b.next
    tail = tail.next
tail.next = a or b
vals = to_list(dummy.next)
print(*vals if vals else ["EMPTY"])
''', "O(n + m)", "O(1)",
    ["Compare the heads of both lists.", "Use a dummy head node and a tail pointer.",
     "Repeatedly attach the smaller head; attach the leftover list at the end."])

add("ll-003", "Drop the k-th From the End", "linked_lists", "medium",
    "Remove the k-th node from the end of a singly linked list in one pass and print the remaining values. Print "
    "EMPTY if the list becomes empty.",
    ["1 <= k <= n <= 10^5"], "Line 1: n k. Line 2: n integers.", "Remaining values, or EMPTY.",
    ["5 2\n1 2 3 4 5", "1 1\n1", "2 1\n1 2", "2 2\n1 2", "4 4\n9 8 7 6", "6 3\n1 1 2 2 3 3"],
    LIST_BUILD + '''
import sys
d = sys.stdin.read().split()
n, k = int(d[0]), int(d[1])
dummy = ListNode(0, build(list(map(int, d[2:2 + n]))))
fast = slow = dummy
for _ in range(k):
    fast = fast.next
while fast.next:
    fast, slow = fast.next, slow.next
slow.next = slow.next.next
vals = to_list(dummy.next)
print(*vals if vals else ["EMPTY"])
''', "O(n)", "O(1)",
    ["How can you know where 'k from the end' is without counting first?", "Two pointers k nodes apart.",
     "Advance fast k steps from a dummy node, then move both until fast reaches the last node."])

# ------------------------------------------------------------------- trees
TREE_FMT = "One line: the tree in level order, using null for missing children (null alone = empty tree)."
add("tree-001", "Org Chart Depth", "trees", "easy",
    "An org chart is stored as a binary tree. Print its maximum depth (number of nodes on the longest root-to-leaf "
    "path). An empty tree has depth 0.",
    ["0 <= nodes <= 10^4"], TREE_FMT, "One integer.",
    ["3 9 20 null null 15 7", "null", "1", "1 2 null 3 null 4", "1 2 3 4 5 6 7", "1 null 2 null 3"],
    TREE_BUILD + '''
import sys
root = build(sys.stdin.read().split())
depth = 0
level = [root] if root else []
while level:
    depth += 1
    level = [c for n in level for c in (n.left, n.right) if c]
print(depth)
''', "O(n)", "O(w) where w is the maximum width",
    ["Depth = 1 + max(depth of children).", "Either recursion (DFS) or level-by-level BFS works.",
     "Count how many BFS levels you process; avoid deep recursion on skewed trees."])

add("tree-002", "Floor-by-Floor Listing", "trees", "medium",
    "Print the values of a binary tree level by level, one line per level from left to right. Print EMPTY for an "
    "empty tree.",
    ["0 <= nodes <= 10^4"], TREE_FMT, "One line per level, or EMPTY.",
    ["3 9 20 null null 15 7", "null", "1", "1 2 3 4 null null 5", "5 4 null 3 null 2", "1 2 3 4 5 6 7"],
    TREE_BUILD + '''
import sys
root = build(sys.stdin.read().split())
if not root:
    print("EMPTY")
level = [root] if root else []
while level:
    print(*[n.val for n in level])
    level = [c for n in level for c in (n.left, n.right) if c]
''', "O(n)", "O(w)",
    ["Process nodes in the order they are discovered.", "Breadth-first search with a queue.",
     "Process the queue one level at a time by remembering its size before each level."])

add("tree-003", "Is It a Search Tree?", "trees", "medium",
    "Check whether a binary tree is a valid binary search tree: every node is strictly greater than all values in "
    "its left subtree and strictly smaller than all values in its right subtree. An empty tree is valid.",
    ["0 <= nodes <= 10^4", "-10^9 <= value <= 10^9"], TREE_FMT, "YES or NO.",
    ["2 1 3", "5 1 4 null null 3 6", "null", "1", "10 5 15 null null 6 20", "8 4 12 2 6 10 14", "2 2 2"],
    TREE_BUILD + '''
import sys
root = build(sys.stdin.read().split())
ok = True
stack = [(root, float("-inf"), float("inf"))] if root else []
while stack:
    n, lo, hi = stack.pop()
    if not (lo < n.val < hi):
        ok = False
        break
    if n.left:
        stack.append((n.left, lo, n.val))
    if n.right:
        stack.append((n.right, n.val, hi))
print("YES" if ok else "NO")
''', "O(n)", "O(h)",
    ["Checking only a node and its direct children is not enough - why?", "Pass down a valid (low, high) range.",
     "Each node must lie strictly inside its range; children inherit a narrowed range."])

add("tree-004", "Common Manager in a BST", "trees", "medium",
    "Employees are stored in a binary search tree by ID. Given two IDs p and q that exist in the tree, print the ID "
    "of their lowest common ancestor.",
    ["2 <= nodes <= 10^4", "all IDs are distinct"], "Line 1: the BST in level order (null for missing). Line 2: p q.",
    "One integer.",
    ["6 2 8 0 4 7 9 null null 3 5\n2 8", "6 2 8 0 4 7 9 null null 3 5\n2 4", "2 1\n2 1",
     "6 2 8 0 4 7 9 null null 3 5\n3 5", "5 3 8 2 4 7 9\n7 9", "5 3 8 2 4 7 9\n2 9"],
    TREE_BUILD + '''
import sys
lines = sys.stdin.read().strip().split("\\n")
root = build(lines[0].split())
p, q = map(int, lines[1].split())
node = root
while node:
    if p < node.val and q < node.val:
        node = node.left
    elif p > node.val and q > node.val:
        node = node.right
    else:
        break
print(node.val)
''', "O(h)", "O(1)",
    ["Use the BST ordering property.", "Walk down from the root.",
     "Go left if both are smaller, right if both are larger; otherwise the current node is the answer."])

# ------------------------------------------------------------------ graphs
GRAPH_FMT = "Line 1: n m. Next m lines: u v (1-indexed)."
add("graph-001", "Friend Circles", "graphs", "medium",
    "n students and m undirected friendships are given. Print the number of friend circles (connected components).",
    ["1 <= n <= 10^5", "0 <= m <= 2*10^5"], GRAPH_FMT, "One integer.",
    ["5 3\n1 2\n2 3\n4 5", "1 0", "4 0", "3 3\n1 2\n2 3\n3 1", "6 2\n1 6\n2 5", "5 4\n1 2\n1 3\n1 4\n1 5"],
    '''
import sys
d = list(map(int, sys.stdin.read().split()))
n, m = d[0], d[1]
parent = list(range(n + 1))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
comps = n
for i in range(m):
    a, b = find(d[2 + 2 * i]), find(d[3 + 2 * i])
    if a != b:
        parent[a] = b; comps -= 1
print(comps)
''', "O((n + m) * alpha(n))", "O(n)",
    ["Each friendship can merge two groups.", "Use BFS/DFS or Union-Find (disjoint set union).",
     "Start with n components and decrement each time a union joins two different roots."])

add("graph-002", "Fewest Bus Hops", "graphs", "medium",
    "Bus stops 1..n are connected by m undirected routes. Print the minimum number of routes needed to go from stop 1 "
    "to stop n, or -1 if it is impossible.",
    ["1 <= n <= 10^5", "0 <= m <= 2*10^5"], GRAPH_FMT, "One integer.",
    ["4 4\n1 2\n2 3\n3 4\n1 3", "2 0", "1 0", "5 4\n1 2\n2 3\n3 4\n4 5", "5 5\n1 2\n1 3\n2 4\n3 4\n4 5", "3 1\n2 3"],
    '''
import sys
from collections import deque
d = list(map(int, sys.stdin.read().split()))
n, m = d[0], d[1]
g = [[] for _ in range(n + 1)]
for i in range(m):
    u, v = d[2 + 2 * i], d[3 + 2 * i]
    g[u].append(v); g[v].append(u)
dist = [-1] * (n + 1); dist[1] = 0
q = deque([1])
while q:
    u = q.popleft()
    for v in g[u]:
        if dist[v] == -1:
            dist[v] = dist[u] + 1; q.append(v)
print(dist[n])
''', "O(n + m)", "O(n + m)",
    ["All routes have the same cost.", "Breadth-first search finds shortest paths in unweighted graphs.",
     "BFS from stop 1, storing the distance when each stop is first discovered."])

add("graph-003", "Course Prerequisites", "graphs", "medium",
    "There are n courses. Each of m lines 'a b' means course b must be completed before course a. Print YES if it is "
    "possible to finish all courses, otherwise NO.",
    ["1 <= n <= 10^5", "0 <= m <= 2*10^5"], "Line 1: n m. Next m lines: a b.", "YES or NO.",
    ["2 1\n2 1", "2 2\n1 2\n2 1", "3 0", "4 3\n2 1\n3 2\n4 3", "3 3\n1 2\n2 3\n3 1", "1 1\n1 1"],
    '''
import sys
from collections import deque
d = list(map(int, sys.stdin.read().split()))
n, m = d[0], d[1]
g = [[] for _ in range(n + 1)]; indeg = [0] * (n + 1)
for i in range(m):
    a, b = d[2 + 2 * i], d[3 + 2 * i]
    g[b].append(a); indeg[a] += 1
q = deque(i for i in range(1, n + 1) if indeg[i] == 0)
done = 0
while q:
    u = q.popleft(); done += 1
    for v in g[u]:
        indeg[v] -= 1
        if indeg[v] == 0:
            q.append(v)
print("YES" if done == n else "NO")
''', "O(n + m)", "O(n + m)",
    ["When is it impossible to finish everything?", "Detect a cycle in a directed graph (topological sort).",
     "Kahn's algorithm: repeatedly take courses with in-degree 0; all n must be taken."])

add("graph-004", "Island Counter", "graphs", "medium",
    "A satellite grid marks land with 1 and water with 0. Count the islands (groups of land cells connected "
    "horizontally or vertically).",
    ["1 <= r, c <= 300"], "Line 1: r c. Next r lines: a string of c characters (0/1).", "One integer.",
    ["4 5\n11110\n11010\n11000\n00000", "4 5\n11000\n11000\n00100\n00011", "1 1\n0", "1 1\n1", "3 3\n101\n010\n101",
     "2 4\n1111\n1111"],
    '''
import sys
d = sys.stdin.read().split()
r, c = int(d[0]), int(d[1])
grid = [list(row) for row in d[2:2 + r]]
count = 0
for i in range(r):
    for j in range(c):
        if grid[i][j] == "1":
            count += 1
            stack = [(i, j)]; grid[i][j] = "0"
            while stack:
                x, y = stack.pop()
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < r and 0 <= ny < c and grid[nx][ny] == "1":
                        grid[nx][ny] = "0"; stack.append((nx, ny))
print(count)
''', "O(r * c)", "O(r * c)",
    ["Each island should be counted when you first touch it.", "Flood fill with DFS or BFS.",
     "For every unvisited land cell, increment the count and sink its whole island."])

add("graph-005", "Cheapest Delivery Route", "graphs", "hard",
    "A delivery network has n hubs and m one-way roads with non-negative costs. Print the minimum cost from hub 1 to "
    "hub n, or -1 if hub n is unreachable.",
    ["1 <= n <= 10^5", "0 <= m <= 2*10^5", "0 <= w <= 10^6"], "Line 1: n m. Next m lines: u v w.", "One integer.",
    ["4 5\n1 2 4\n1 3 1\n3 2 2\n2 4 1\n3 4 5", "2 0", "1 0", "3 3\n1 2 5\n2 3 5\n1 3 20", "4 4\n1 2 1\n2 3 1\n3 4 1\n1 4 2",
     "5 4\n1 2 100\n2 5 100\n1 3 1\n3 4 1"],
    '''
import sys, heapq
d = list(map(int, sys.stdin.read().split()))
n, m = d[0], d[1]
g = [[] for _ in range(n + 1)]
for i in range(m):
    u, v, w = d[2 + 3 * i], d[3 + 3 * i], d[4 + 3 * i]
    g[u].append((v, w))
INF = float("inf"); dist = [INF] * (n + 1); dist[1] = 0
pq = [(0, 1)]
while pq:
    du, u = heapq.heappop(pq)
    if du > dist[u]:
        continue
    for v, w in g[u]:
        if du + w < dist[v]:
            dist[v] = du + w; heapq.heappush(pq, (dist[v], v))
print(dist[n] if dist[n] < INF else -1)
''', "O((n + m) log n)", "O(n + m)",
    ["BFS is not enough because roads have different costs.", "Dijkstra's algorithm with a min-heap.",
     "Pop the closest hub, relax its outgoing roads, and skip stale heap entries."], styles=("product", "startup"))

# ---------------------------------------------------------------------- DP
add("dp-001", "Staircase Ways", "dp", "easy",
    "You can climb 1 or 2 steps at a time. Print the number of distinct ways to reach step n, modulo 1000000007.",
    ["1 <= n <= 10^5"], "One integer n.", "One integer.",
    ["1", "2", "3", "5", "10", "45", "1000"],
    '''
import sys
n = int(sys.stdin.read())
MOD = 10**9 + 7
a, b = 1, 1
for _ in range(n - 1):
    a, b = b, (a + b) % MOD
print(b)
''', "O(n)", "O(1)",
    ["How can you arrive at step n?", "ways(n) = ways(n-1) + ways(n-2).",
     "Iterate with two variables instead of recursion to avoid exponential time."])

add("dp-002", "Fewest Coins for Change", "dp", "medium",
    "Given coin denominations (unlimited supply) and an amount, print the minimum number of coins that make the "
    "amount, or -1 if it cannot be made.",
    ["1 <= n <= 12", "0 <= amount <= 10^4"], "Line 1: n amount. Line 2: n coin values.", "One integer.",
    ["3 11\n1 2 5", "1 3\n2", "1 0\n1", "2 6\n1 3", "3 7\n2 4 6", "4 63\n1 5 10 21", "1 10000\n1"],
    '''
import sys
d = list(map(int, sys.stdin.read().split()))
n, amt = d[0], d[1]; coins = d[2:2 + n]
INF = float("inf"); dp = [0] + [INF] * amt
for x in range(1, amt + 1):
    for c in coins:
        if c <= x and dp[x - c] + 1 < dp[x]:
            dp[x] = dp[x - c] + 1
print(dp[amt] if dp[amt] < INF else -1)
''', "O(amount * n)", "O(amount)",
    ["Greedy fails for some coin systems (try 1, 3, 4 with amount 6).", "Bottom-up DP over amounts.",
     "dp[x] = 1 + min(dp[x - c]) over coins c <= x; dp[0] = 0."])

add("dp-003", "Longest Rising Streak", "dp", "medium",
    "Print the length of the longest strictly increasing subsequence of the array (not necessarily contiguous).",
    ["1 <= n <= 10^5"], "Line 1: n. Line 2: n integers.", "One integer.",
    ["8\n10 9 2 5 3 7 101 18", "1\n5", "6\n0 1 0 3 2 3", "7\n7 7 7 7 7 7 7", "5\n5 4 3 2 1", "6\n1 2 3 4 5 6"],
    '''
import sys
from bisect import bisect_left
d = list(map(int, sys.stdin.read().split()))
n = d[0]; tails = []
for x in d[1:1 + n]:
    i = bisect_left(tails, x)
    if i == len(tails):
        tails.append(x)
    else:
        tails[i] = x
print(len(tails))
''', "O(n log n)", "O(n)",
    ["An O(n^2) DP works; can you do better?", "Patience sorting: keep the smallest tail for each length.",
     "For each x, binary-search the first tail >= x and replace it (or append)."])

add("dp-004", "Shared Skill Sequence", "dp", "medium",
    "Two strings describe skill sequences. Print the length of their longest common subsequence.",
    ["1 <= |a|, |b| <= 2000"], "Line 1: a. Line 2: b.", "One integer.",
    ["abcde\nace", "abc\nabc", "abc\ndef", "a\na", "AGGTAB\nGXTXAYB", "aaaa\naa"],
    '''
import sys
a, b = sys.stdin.read().split()[:2]
prev = [0] * (len(b) + 1)
for ca in a:
    cur = [0] * (len(b) + 1)
    for j, cb in enumerate(b, 1):
        cur[j] = prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1])
    prev = cur
print(prev[-1])
''', "O(|a| * |b|)", "O(|b|)",
    ["Compare the last characters of both strings.", "2-D DP on prefixes; it can be reduced to one row.",
     "If chars match, take diagonal + 1; otherwise the max of left and top."])

add("dp-005", "Hackathon Backpack", "dp", "hard",
    "You can carry at most W kg of gadgets to a hackathon. Each gadget has a weight and a usefulness value and can be "
    "taken at most once. Print the maximum total value.",
    ["1 <= n <= 100", "0 <= W <= 10^4"], "Line 1: n W. Next n lines: weight value.", "One integer.",
    ["3 50\n10 60\n20 100\n30 120", "1 5\n10 100", "4 7\n1 1\n3 4\n4 5\n5 7", "2 0\n1 5\n2 3", "3 6\n2 3\n2 3\n2 3",
     "5 10\n5 10\n4 40\n6 30\n3 50\n2 1"],
    '''
import sys
d = list(map(int, sys.stdin.read().split()))
n, W = d[0], d[1]
dp = [0] * (W + 1)
for i in range(n):
    w, v = d[2 + 2 * i], d[3 + 2 * i]
    for cap in range(W, w - 1, -1):
        dp[cap] = max(dp[cap], dp[cap - w] + v)
print(dp[W])
''', "O(n * W)", "O(W)",
    ["Greedy by value/weight ratio does not always work for 0/1 items.", "Classic 0/1 knapsack DP.",
     "Iterate capacities from W down to w so each item is used at most once."], styles=("product",))

add("dp-006", "Minimum Edits", "dp", "hard",
    "Print the minimum number of single-character insertions, deletions or replacements needed to turn string a into "
    "string b. Either string may be empty.",
    ["0 <= |a|, |b| <= 2000"], "Line 1: a (may be empty). Line 2: b (may be empty).", "One integer.",
    ["horse\nros", "intention\nexecution", "a\na", "abc\n", "\nabc", "kitten\nsitting", "abc\nyabd"],
    '''
import sys
lines = sys.stdin.read().split("\\n")
a = lines[0].strip(); b = lines[1].strip() if len(lines) > 1 else ""
prev = list(range(len(b) + 1))
for i, ca in enumerate(a, 1):
    cur = [i] + [0] * len(b)
    for j, cb in enumerate(b, 1):
        cur[j] = prev[j - 1] if ca == cb else 1 + min(prev[j - 1], prev[j], cur[j - 1])
    prev = cur
print(prev[-1])
''', "O(|a| * |b|)", "O(|b|)",
    ["Think about the last character of each string.", "2-D DP: dp[i][j] = edits for a[:i] -> b[:j].",
     "Match costs dp[i-1][j-1]; otherwise 1 + min(replace, delete, insert)."], styles=("product", "startup"))

# --------------------------------------------------------------------- SQL
EMP = "CREATE TABLE employees(id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary INTEGER);\n"
def emp(rows):
    vals = ",".join(f"({i},'{n}','{d}',{s})" for i, n, d, s in rows)
    return EMP + (f"INSERT INTO employees VALUES {vals};\n" if rows else "")

E1 = [(1, "Asha", "Eng", 72000), (2, "Ravi", "Eng", 48000), (3, "Meena", "HR", 52000), (4, "John", "Sales", 50000),
      (5, "Kiran", "Eng", 72000), (6, "Divya", "HR", 39000)]
E2 = [(1, "Arun", "Ops", 30000), (2, "Bala", "Ops", 41000)]
E3 = [(1, "Zara", "Eng", 90000)]
E4 = [(1, "A", "X", 60000), (2, "B", "X", 60000), (3, "C", "Y", 60000)]
E5 = [(1, "Nila", "Eng", 50001), (2, "Omar", "Eng", 50000), (3, "Priya", "Data", 120000), (4, "Sam", "Data", 75000)]
E6 = []
SQL_INPUTS = [emp(E1), emp(E2), emp(E3), emp(E4), emp(E5), emp(E6)]
SQL_FMT = "A SQLite setup script creating and filling the tables (passed on stdin)."
SQL_OUT = "The query result: one row per line, columns separated by |, NULL for nulls, no header."

add("sql-001", "High Earners", "sql", "easy",
    "Table employees(id, name, dept, salary). Write a query returning name and salary of employees earning more than "
    "50000, ordered by salary descending and then name ascending.",
    ["Use ORDER BY exactly as described"], SQL_FMT, SQL_OUT, SQL_INPUTS,
    "SELECT name, salary FROM employees WHERE salary > 50000 ORDER BY salary DESC, name ASC;",
    "O(n log n)", "O(n)",
    ["Filter rows first.", "Use WHERE with a strict comparison.", "Then add ORDER BY salary DESC, name ASC."],
    kind="sql")

add("sql-002", "Department Pay Averages", "sql", "medium",
    "Table employees(id, name, dept, salary). For each department with at least 2 employees, return dept and the "
    "average salary rounded to 2 decimals (column avg_salary), ordered by avg_salary descending, then dept.",
    ["Use GROUP BY and HAVING"], SQL_FMT, SQL_OUT, SQL_INPUTS,
    "SELECT dept, ROUND(AVG(salary), 2) AS avg_salary FROM employees GROUP BY dept HAVING COUNT(*) >= 2 "
    "ORDER BY avg_salary DESC, dept;",
    "O(n log n)", "O(n)",
    ["You need one row per department.", "GROUP BY dept; filter groups with HAVING (not WHERE).",
     "HAVING COUNT(*) >= 2, select ROUND(AVG(salary), 2), then ORDER BY."], kind="sql")

add("sql-003", "Runner-Up Salary", "sql", "medium",
    "Table employees(id, name, dept, salary). Return a single column second_highest containing the second highest "
    "DISTINCT salary, or NULL if it does not exist.",
    ["Exactly one row must be returned"], SQL_FMT, SQL_OUT, SQL_INPUTS,
    "SELECT (SELECT DISTINCT salary FROM employees ORDER BY salary DESC LIMIT 1 OFFSET 1) AS second_highest;",
    "O(n log n)", "O(n)",
    ["Duplicates of the top salary must not count twice.", "Use DISTINCT with ORDER BY ... LIMIT/OFFSET.",
     "Wrap it in a scalar subquery so an empty result becomes NULL."], kind="sql")

CUST = ("CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT);\n"
        "CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER);\n")
def cust(cs, os_):
    s = CUST
    if cs:
        s += "INSERT INTO customers VALUES " + ",".join(f"({i},'{n}')" for i, n in cs) + ";\n"
    if os_:
        s += "INSERT INTO orders VALUES " + ",".join(f"({i},{c},{a})" for i, c, a in os_) + ";\n"
    return s

add("sql-004", "Customers Without Orders", "sql", "easy",
    "Tables customers(id, name) and orders(id, customer_id, amount). Return the names of customers who never placed "
    "an order, ordered by name.",
    ["Use a LEFT JOIN or NOT EXISTS"], SQL_FMT, SQL_OUT,
    [cust([(1, "Anu"), (2, "Bharath"), (3, "Chitra")], [(1, 1, 500), (2, 1, 200)]),
     cust([(1, "Dev")], []),
     cust([(1, "Esha"), (2, "Farhan")], [(1, 1, 10), (2, 2, 20)]),
     cust([], []),
     cust([(3, "Zoya"), (1, "Mani"), (2, "Leela")], [(1, 2, 99)]),
     cust([(1, "Gopi"), (2, "Hari")], [(1, 5, 70)])],
    "SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id WHERE o.id IS NULL ORDER BY c.name;",
    "O(n + m)", "O(n)",
    ["Which customers have no matching row in orders?", "LEFT JOIN keeps customers without orders (NULL columns).",
     "Filter WHERE o.id IS NULL, then ORDER BY name."], kind="sql")


# ------------------------------------------------------------------- build
out = []
for q in Q:
    lang = "sql" if q["kind"] == "sql" else "python"
    tests = []
    for inp in q.pop("inputs"):
        r = SB.run_once(lang, q["reference_solution"], inp)
        assert r.exit_code == 0 and not r.timed_out, (q["id"], inp, r.stderr)
        tests.append({"input": inp, "output": r.stdout.strip()})
    assert len(tests) >= 6, q["id"]
    for i, t in enumerate(tests):
        t["hidden"] = i >= 2
    q["examples"] = [{"input": t["input"], "output": t["output"], "explanation": ""} for t in tests[:2]]
    q["test_cases"] = tests
    out.append(q)

path = ROOT / "data" / "question_bank.json"
import os  # noqa: E402

os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(out, f, indent=2)
from collections import Counter  # noqa: E402

print(len(out), Counter(q["topic"] for q in out), Counter(q["difficulty"] for q in out))
