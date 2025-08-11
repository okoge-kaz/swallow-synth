MULTIPLE_SOLUTION_PROMPT = """
You are an expert mathematical author. Your task is to take a mathematical concept or problem and present it by first showing multiple distinct solutions, and then providing a comparative analysis of those methods. The output must begin with '<|MATH_TEXT|>'.

## Instructions:
1.  **Output Structure**:
    * Begin with '<|MATH_TEXT|>' on a new line.
    * State the problem under a `## Problem` heading.
    * Present each solution under a separate subheading (e.g., `### Solution 1: Algebraic Approach`, `### Solution 2: Substitution Method`).
    * Conclude with a `## Comparative Analysis` section.
2.  **Solution Content**:
    * Provide at least two different, valid methods for solving the problem.
    * If the source text only provides one method, you must generate a second, distinct method.
    * Each solution should be clearly explained step-by-step.
3.  **Comparative Analysis Section**:
    * Analyze the trade-offs between the presented solutions.
    * Discuss aspects like computational efficiency, conceptual elegance, risk of errors, and generalizability.
    * Conclude with advice on when one method might be preferable over another.
4.  **Formatting**:
    * Use LaTeX for all mathematical notation ($$ for display, $ for inline).

## Example Input and Example Output:

### Example Input:
```
A system of linear equations can be solved using various techniques. For the system x + y = 7 and x - y = 1, one can use the elimination method. By adding the two equations, the y-variable is eliminated, yielding 2x = 8, so x = 4. Substituting x = 4 back into the first equation gives 4 + y = 7, so y = 3.
```

### Example Output:
<|MATH_TEXT|>
## Problem
Solve the following system of linear equations:
$$
\begin{cases}
x + y = 7 \\
x - y = 1
\end{cases}
$$

### Solution 1: Elimination Method
This method aims to eliminate one of the variables by adding or subtracting the equations.

1.  **Add the two equations together**: We add the left sides and the right sides.
    $$
    (x + y) + (x - y) = 7 + 1
    $$
2.  **Simplify the resulting equation**: The $y$ terms cancel out.
    $$
    2x = 8
    $$
3.  **Solve for $x$**:
    $$
    x = 4
    $$
4.  **Substitute $x$ back into an original equation**: Using the first equation:
    $$
    4 + y = 7
    $$
5.  **Solve for $y$**:
    $$
    y = 3
    $$
The solution is $$(x, y) = (4, 3)$$.

### Solution 2: Substitution Method
This method involves solving one equation for one variable and substituting that expression into the other equation.

1.  **Solve the first equation for $x$**:
    $$
    x = 7 - y
    $$
2.  **Substitute this expression for $x$ into the second equation**:
    $$
    (7 - y) - y = 1
    $$
3.  **Simplify and solve for $y$**:
    $$
    7 - 2y = 1 \\
    -2y = -6 \\
    y = 3
    $$
4.  **Substitute the value of $y$ back to find $x$**: Using the expression from step 1:
    $$
    x = 7 - 3 = 4
    $$
The solution is $$(x, y) = (4, 3)$$.

## Comparative Analysis
Both the **Elimination Method** and the **Substitution Method** yield the correct result, but they have different strengths.

* **Efficiency**: For this specific problem, the Elimination Method is slightly more direct. The coefficients of $y$ are already opposites (+1 and -1), making elimination via addition a single, clean step. Substitution required more algebraic manipulation.
* **Generalizability**: The Substitution Method is arguably more universally applicable. It works easily even when coefficients aren't simple opposites or multiples. Elimination often requires an extra step of multiplying one or both equations to align coefficients, which can introduce arithmetic errors.
* **Recommendation**: When you can immediately spot that adding or subtracting equations will cancel a variable (as in this case), Elimination is often faster. For more complex systems or when one variable is already isolated, Substitution is a very reliable and systematic approach."""
