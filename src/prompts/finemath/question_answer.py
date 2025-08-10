QUESTION_ANSWER_PROMPT = """
You are an expert in mathematics and related fields, including physics, economics, and applied mathematics. Your task is to convert the provided text into a Question-Answer format suitable for educational purposes, ensuring the output begins with '<|MATH_TEXT|>'.

## Instructions:
1. **Output Structure**:
   - Begin the output with '<|MATH_TEXT|>' on a new line, followed by the Question-Answer pairs.
   - Format questions as **Question 1**, **Question 2**, etc., and answers as **Answer 1**, **Answer 2**, etc.
2. **Question Extraction**:
   - If the text contains explicit problems, rephrase them clearly as questions.
   - If the text lacks explicit problems, generate **two original problems** based on the text's context, ensuring answers involve mathematical equations.
   - If the text lacks sufficient context, supplement it with relevant background information to ensure the problem is well-defined.
3. **Answer Requirements**:
   - Provide concise, accurate answers with equations formatted in LaTeX using $$ for display equations (e.g., $$x^2 + y^2 = r^2$$) and $ for inline equations (e.g., $x^2$).
   - For complex problems (e.g., advanced calculus, physics, or economics), include a step-by-step solution that is clear but not overly verbose.
4. **Code Implementation**:
   - If the text includes code or a programming solution enhances understanding, provide a separate code block labeled **Code Implementation N** after each answer.
   - Use Python (unless otherwise specified) with clear comments explaining each step and linking to the mathematical equations used.
5. **Verification**:
   - For problems in physics, economics, or other applied fields, verify the problem's assumptions and context before deriving the answer to ensure accuracy.

## Example Input and Example Output:
The following are the Example Input and Example Output to illustrate the expected format and content.

### Example Input:
```
1. **Dot Product Function**:  \n   Given two lists of numbers $ \\text{lst}_1 $ and $ \\text{lst}_2 $, compute their dot product:  \n   $$\n   \\text{dot-product}(\\text{lst}_1, \\text{lst}_2) =
\\sum_{i=1}^{n} a_i \\cdot b_i\n   $$  \n   where $ a_i $ and $ b_i $ are elements of $ \\text{lst}_1 $ and $ \\text{lst}_2 $, respectively.  \n   Example:  \n   $$\n   \\text{dot-product}([1, 3, 4], [5, 7,
 8]) = (1 \\cdot 5) + (3 \\cdot 7) + (4 \\cdot 8) = 5 + 21 + 32 = 58\n   $$\n\n2. **Zip Function**:  \n   Given two lists $ \\text{lst}_1 $ and $ \\text{lst}_2 $, pair their elements into a list of pairs:
\n   $$\n   \\text{zip}(\\text{lst}_1, \\text{lst}_2) = [(a_1, b_1), (a_2, b_2), \\dots, (a_n, b_n)]\n   $$  \n   Example:  \n   $$\n   \\text{zip}([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) = [(1, 6), (2, 7), (3,
8), (4, 9), (5, 10)]\n   $$\n\n3. **Map-Combine Generalization**:  \n   A higher-order function $ \\text{map-combine} $ takes:  \n   - A base value $ \\text{base} $,  \n   - An element-wise function $ \\tex
t{func-elem} $,  \n   - A list-combining function $ \\text{func-lst} $,  \n   - Two lists $ \\text{lst}_1 $ and $ \\text{lst}_2 $,  \n   and recursively applies $ \\text{func-elem} $ to corresponding elemen
ts, combining results with $ \\text{func-lst} $.  \n   Example:  \n   $$\n   \\text{map-combine}(0, *, +, [1, 3, 4], [5, 7, 8]) = (1 \\cdot 5) + (3 \\cdot 7) + (4 \\cdot 8) = 58\n   $$\n\n4. **Binary Search
 Tree (BST) Recursion**:  \n   Define a BST with nodes containing a key (number), value (string), left, and right subtrees.  \n   - **Search**: Recursively traverse the tree based on key comparisons.  \n
- **Node Removal**: Maintain BST properties after deletion.  \n\n5. **Unbounded Tree (UT) Mutual Recursion**:  \n   Define a tree with a value and a list of child trees.  \n   - **Traversal**: Recursively p
rocess the root and all subtrees.  \n\n6. **Foldr Examples**:  \n   - **Sum**: $ \\text{foldr}(+, 0, [5, 2, 7, 4, 1, 3]) = 5 + 2 + 7 + 4 + 1 + 3 = 22 $  \n   - **Count Elements**: $ \\text{foldr}(\\text{add
1}, 0, [5, 2, 7, 4, 1, 3]) = 6 $  \n   - **List Reconstruction**: $ \\text{foldr}(\\text{cons}, \\text{empty}, [5, 2, 7, 4, 1, 3]) = [5, 2, 7, 4, 1, 3] $  \n\n7. **Lambda and Abstraction**:  \n   - **Anonym
ous Functions**: Use $ \\lambda $ for simple operations (e.g., $ \\lambda(x)(x^2) $).  \n   - **Filter Example**: $ \\text{filter}(\\lambda(x)(x > 4), [5, 2, 7, 4, 1, 3]) = [5, 7] $.  \n\n8. **Quicksort wit
h Custom Comparator**:  \n   Sort a list using a custom comparison function $ f $.  \n   Example:  \n   $$\n   \\text{quicksort}([5, 2, 7, 4, 1, 3], <) = [1, 2, 3, 4, 5, 7]\n   $$  \n   $$\n   \\text{quicks
ort}([5, 2, 7, 4, 1, 3], \\lambda(a, b)(\\text{even?}(a))) = [4, 2, 5, 7, 1, 3]\n   $$
```

### Example Output:
<|MATH_TEXT|>
**Question 1**:
What is the dot product of two lists, `[2, 4, 6]` and `[1, 3, 5]`, and how is it computed?

**Answer 1**:
The dot product of two lists, $$\text{lst}_1 = [2, 4, 6]$$ and $$\text(lst}_2 = [1, 3, 5]$$, is computed as:
$$
\text{dot-product}(\text{lst}_1, \text{lst}_2) = \sum_{i=1}^{n} a_i \cdot b_i
$$
For $$\text{lst}_1 = [2, 4, 6]$$ and $$\text(lst}_2 = [1, 3, 5]$$:
$$
\text{dot-product} = (2 \cdot 1) + (4 \cdot 3) + (6 \cdot 5) = 2 + 12 + 30 = 44
$$

**Code Implementation 1**:
```python
def dot_product(lst1, lst2):
    # Compute dot product by multiplying corresponding elements and summing
    return sum(a * b for a, b in zip(lst1, lst2))

# Example usage
lst1 = [2, 4, 6]
lst2 = [1, 3, 5]
result = dot_product(lst1, lst2)  # Corresponds to (2*1) + (4*3) + (6*5) = 44
print(result)  # Output: 44
```

**Question 2**:
Using the `map-combine` function with a base value of 0, element-wise multiplication, and list-combining addition, compute the result for the lists `[1, 3, 4]` and `[5, 7, 8]`.

**Answer 2**:
The `map-combine` function applies an element-wise function (multiplication, $$*$$) to corresponding elements of two lists and combines the results using a list-combining function (addition, $$+$$) with a base value of 0. For $$\text(lst}_1 = [1, 3, 4]$$ and $$\text{lst}_2 = [5, 7, 8]$$:
$$
\text{map-combine}(0, *, +, [1, 3, 4], [5, 7, 8]) = (1 \cdot 5) + (3 \cdot 7) + (4 \cdot 8) = 5 + 21 + 32 = 58
$$

**Code Implementation 2**:
```python
def map_combine(base, func_elem, func_lst, lst1, lst2):
    # Pair elements, apply func_elem, and combine with func_lst starting from base
    pairs = zip(lst1, lst2)
    result = base
    for a, b in pairs:
        result = func_lst(result, func_elem(a, b))  # Apply func_elem, then func_lst
    return result

# Example usage
lst1 = [1, 3, 4]
lst2 = [5, 7, 8]
result = map_combine(0, lambda x, y: x * y, lambda x, y: x + y, lst1, lst2)
# Corresponds to (1*5) + (3*7) + (4*8) = 5 + 21 + 32 = 58
print(result)  # Output: 58
```

## Final Instruction:
Always begin the output with '<|MATH_TEXT|>' on a new line, followed by the Question-Answer pairs as shown in the Example Output.
"""
