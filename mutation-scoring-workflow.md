Here’s an example walk-through of SyntaxLab’s mutation scoring workflow with realistic outputs and logs, modeled after MuTAP-style evaluation.

⸻

📦 Task

Generate a test suite for a simple Stack class implementation in Python.

✏️ LLM-Generated Code

class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        return self._items.pop()

    def peek(self):
        return self._items[-1]

    def is_empty(self):
        return len(self._items) == 0


⸻

🧪 LLM-Generated Test Suite

import unittest

class TestStack(unittest.TestCase):
    def test_push(self):
        s = Stack()
        s.push(10)
        self.assertEqual(s.peek(), 10)

    def test_pop(self):
        s = Stack()
        s.push(5)
        s.push(6)
        self.assertEqual(s.pop(), 6)

    def test_is_empty(self):
        s = Stack()
        self.assertTrue(s.is_empty())


⸻

🔁 Mutation Injection Log

Mutation ID	Operator	Mutation Description
M001	Invert Logic	len(self._items) == 0 → != 0
M002	Remove Element	.pop() → no operation
M003	Return Constant	return self._items[-1] → return 0
M004	Swap Append Order	append(item) → insert(0, item)
M005	Delete Push Functionality	Remove push body


⸻

✅ Test Results (Mutation Execution)

Mutant ID	Tests Run	Mutant Killed	Reason
M001	✅ All	✅ Yes	is_empty() failed
M002	✅ All	✅ Yes	pop() failed (empty list)
M003	✅ All	✅ Yes	peek() returned wrong value
M004	✅ All	❌ No	Still returned same top element
M005	✅ All	✅ Yes	push() didn’t modify state


⸻

📊 Summary
	•	Total Mutants: 5
	•	Killed: 4
	•	Survived: 1
	•	Mutation Score:
\frac{4}{5} = 0.80 \quad (80\%)

⸻

💡 Feedback Loop Triggered

Survivor: M004 (order mutation)
→ New prompt injected by SyntaxLab:

“Also add a test to ensure that the most recently pushed element is on top.”

🧪 Generated New Test (Refinement)

def test_order(self):
    s = Stack()
    s.push(1)
    s.push(2)
    self.assertEqual(s.pop(), 2)

→ Mutation M004 is now killed.
✅ Updated Mutation Score: 5/5 → 100%

⸻

🧠 Key Takeaways
	•	Mutations that survive indicate missing edge case tests
	•	Test-first prompt generation + mutation scoring = measurable quality
	•	Refinement loop injects precise feedback-driven improvement
	•	This workflow scales across models, languages, and regulatory environments