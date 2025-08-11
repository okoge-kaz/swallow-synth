PLANNING_APPROACH_PROMPT = """
You are an expert in mathematical pedagogy. Your task is to convert the provided text into a "Principle -> Problem -> Execution" format, suitable for teaching abstract problem-solving. The output must begin with '<|MATH_TEXT|>'.

## Instructions:
1.  **Output Structure**:
    * Begin the output with '<|MATH_TEXT|>' on a new line.
    * Structure the content into three distinct sections: `### Principle`, `### Problem`, and `### Step-by-Step Execution`.
2.  **Principle Section**:
    * Identify the core mathematical theorem, formula, or concept that is central to the text.
    * Provide a concise and clear definition or explanation of this principle.
3.  **Problem Section**:
    * If the text contains an explicit problem, rephrase it clearly.
    * If not, generate one original problem that can be solved by applying the identified principle.
4.  **Step-by-Step Execution Section**:
    * Provide a detailed, step-by-step solution to the problem.
    * Explicitly reference how each step applies the principle defined in the first section.
    * Format all equations in LaTeX using $$ for display equations and $ for inline equations.

## Example Input and Example Output:

### Example Input:
```
The area of a circle is given by the formula A = πr², where r is the radius. The circumference is C = 2πr. These formulas are fundamental in geometry for solving problems involving circular shapes, such as finding the area of a sector or the length of an arc. For a sector with angle θ in radians, the area is (θ/2π) * πr² = (1/2)r²θ.
```

### Example Output:
<|MATH_TEXT|>
### Principle
The area of a sector of a circle is derived from the total area of the circle. The area of a full circle is $$A = \pi r^2$$. A sector with a central angle of $\theta$ (in radians) represents a fraction, $$\frac{\theta}{2\pi}$$, of the entire circle. Therefore, the area of the sector, $$A_{\text{sector}}$$, is calculated by multiplying the total area by this fraction:
$$
A_{\text{sector}} = \left( \frac{\theta}{2\pi} \right) \times \pi r^2 = \frac{1}{2} r^2 \theta
$$

### Problem
A circle has a radius of 10 cm. Calculate the area of a sector of this circle that has a central angle of $$\frac{\pi}{3}$$ radians.

### Step-by-Step Execution
1.  **Identify the given values**: The radius $r$ is 10 cm, and the central angle $\theta$ is $$\frac{\pi}{3}$$ radians.
2.  **Recall the principle**: We use the formula for the area of a sector, $$A_{\text{sector}} = \frac{1}{2} r^2 \theta$$.
3.  **Substitute the values into the formula**: This step directly applies the principle using the given problem data.
    $$
    A_{\text{sector}} = \frac{1}{2} (10)^2 \left( \frac{\pi}{3} \right)
    $$
4.  **Calculate the final result**:
    $$
    A_{\text{sector}} = \frac{1}{2} \cdot 100 \cdot \frac{\pi}{3} = \frac{100\pi}{6} = \frac{50\pi}{3}
    $$
The area of the sector is $$\frac{50\pi}{3}$$ square centimeters.
"""
