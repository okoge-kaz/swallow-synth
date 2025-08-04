PRE_TRAIN_MATH_TEXT = """
You are given text that contains mathematical content, potentially mixed with non-mathematical information such as exam names, dates, marks, problem titles, or other metadata. This content may also include knowledge from other fields like finance, physics, computer science, and others intertwined with mathematics. Your task is to:

1. Extract only the mathematics-related content, including problem statements, mathematical functions, equations, conditions, and multiple-choice options (e.g., A, B, C, D) or numerical answers and sequences, while removing any non-mathematical information such as problem titles, time limits, memory limits, or footer/header. If the text includes knowledge from fields like finance, physics, computer science, etc., that is intertwined with mathematics, ensure that such information is not omitted and is preserved in the extracted content without loss.
2. Present the extracted content in a clean, LaTeX-formatted structure, preserving the mathematical expressions, numerical results, and answer sequences exactly as they appear in the original text.
3. If the provided text contains little to no mathematical content, generate new mathematics-related text based on the content of the input. The generated text should be educational, designed to help students study mathematics or assess their understanding, and include clear problem statements, equations, or concepts relevant to learning mathematics.
4. Output the extracted or generated mathematics-related content immediately following the marker `<|MATH_TEXT|>`, ensuring the output is formatted in LaTeX for clarity and precision.

### Example input 1:
```
1\nJEE Main 2022 (Online) 26th July Morning Shift\n+4\n-1\n\nLet f : R $$\\to$$ R be a continuous function such that $$f(3x) - f(x) = x$$. If $$f(8) = 7$$, then $$f(14)$$ is equal to :\n\nA\n4\nB\n10\nC\n11\nD\n16\n...
```

### Example output 1:
```
<|MATH_TEXT|>
1. Let \\( f : \\mathbb{R} \\to \\mathbb{R} \\) be a continuous function such that \\( f(3x) - f(x) = x \\). If \\( f(8) = 7 \\), then \\( f(14) \\) is equal to:
   A. 4
   B. 10
   C. 11
   D. 16
```

### Example input 2:
```
Algorithm\n\nB. Shooting\ntime limit per test\n1 second\nmemory limit per test\n256 megabytes\ninput\nstandard input\noutput\nstandard output\n\nRecently Vasya decided to improve his pistol shooting skills. Today his coach offered him the following exercise. He placed n cans in a row on a table. Cans are numbered from left to right from 1 to n. Vasya has to knock down each can exactly once to finish the exercise. He is allowed to choose the order in which he will knock the cans down.\n\nVasya knows that the durability of the i-th can is ai. It means that if Vasya has already knocked x cans down and is now about to start shooting the i-th one, he will need (ai⋅x+1) shots to knock it down. You can assume that if Vasya starts shooting the i-th can, he will be shooting it until he knocks it down.\n\nYour task is to choose such an order of shooting so that the number of shots required to knock each of the n given cans down exactly once is minimum possible.\n\nInput\n\nThe first line of the input contains one integer n (2≤n≤1000) — the number of cans.\n\nThe second line of the input contains the sequence a1,a2,…,an (1≤ai≤1000), where ai is the durability of the i-th can.\n\nOutput\n\nIn the first line print the minimum number of shots required to knock each of the n given cans down exactly once.\n\nIn the second line print the sequence consisting of n distinct integers from 1 to n — the order of indices of cans that minimizes the number of shots required.\n\nExamples\ninput\nCopy\n3\n20 10 20\noutput\nCopy\n43\n1 3 2\ninput\nCopy\n4\n10 10 10 10\noutput\nCopy\n64\n2 1 4 3\n...
```

### Example output 2:
```
<|MATH_TEXT|>
Given \\( n \\) cans in a row, numbered from 1 to \\( n \\), each with durability \\( a_i \\) (where \\( 1 \\leq a_i \\leq 1000 \\)), the number of shots required to knock down the \\( i \\)-th can after \\( x \\) cans have been knocked down is \\( a_i \\cdot x + 1 \\). Find the order of shooting that minimizes the total number of shots to knock down all \\( n \\) cans exactly once, where \\( 2 \\leq n \\leq 1000 \\).

Examples:
1. For \\( n = 3 \\), \\( a_1 = 20 \\), \\( a_2 = 10 \\), \\( a_3 = 20 \\):
   - Minimum number of shots: 43
   - Order: 1, 3, 2

2. For \\( n = 4 \\), \\( a_1 = 10 \\), \\( a_2 = 10 \\), \\( a_3 = 10 \\), \\( a_4 = 10 \\):
   - Minimum number of shots: 64
   - Order: 2, 1, 4, 3
```

Apply this process to the provided text. Extract the mathematics-related content if present, or generate educational mathematics-related content if the input lacks sufficient mathematical material. Ensure the output is formatted in LaTeX and placed after `<|MATH_TEXT|>`.
"""
