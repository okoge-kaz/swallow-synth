SOCRATIC_METHOD_PROMPT = """
You are an insightful math tutor who uses the Socratic method. Your task is to convert the provided text into a dialogue between a "Teacher" and a "Student" to solve a mathematical problem. The output must begin with '<|MATH_TEXT|>'.

## Instructions:
1.  **Output Structure**:
    * Begin the output with '<|MATH_TEXT|>' on a new line.
    * First, state the problem clearly under a `## Problem` heading.
    * Then, write a dialogue with alternating `**Teacher:**` and `**Student:**` roles.
2.  **Dialogue Content**:
    * The Teacher should guide the Student by asking questions, providing hints, and prompting for the next step, rather than giving direct answers.
    * The Student should articulate their thought process, occasionally making a plausible mistake or expressing uncertainty.
    * The Teacher should help the Student identify and correct their own mistakes.
    * The dialogue should naturally progress to the correct final answer.
3.  **Problem Generation**:
    * If the text contains a problem, use it as the basis for the dialogue.
    * If not, create an original problem from the text's context that allows for a rich, guided conversation.
4.  **Formatting**:
    * Use LaTeX for all mathematical notation ($$ for display, $ for inline).

## Example Input and Example Output:

### Example Input:
```
To solve the equation 3(x - 2) + 5 = 14, one must first apply the distributive property to expand the term 3(x - 2). Then, isolate the variable x by combining like terms and performing inverse operations. First, subtract 5 from both sides, then add 6, and finally divide by 3.
```

### Example Output:
<|MATH_TEXT|>
## Problem
Solve the following linear equation for $x$: $$3(x - 2) + 5 = 14$$

**Teacher:**
Alright, let's take a look at this equation. What's your first instinct for how to approach it?

**Student:**
I see parentheses, so I should probably use the distributive property first. That would give me $$3x - 6 + 5 = 14$$.

**Teacher:**
Excellent start. That's a very solid first step. Now what do you see on the left side of the equation?

**Student:**
There are two constant numbers, -6 and 5. I can combine them. $$-6 + 5$$ is $$-1$$. So the equation becomes $$3x - 1 = 14$$.

**Teacher:**
Perfect. The equation looks much simpler now. What's the goal when we're trying to solve for $x$?

**Student:**
To get $x$ by itself on one side. So, I need to get rid of the -1 and the 3. I should probably divide by 3 first?

**Teacher:**
That's a common thought. But remember the order of operations. We usually handle addition and subtraction before multiplication and division when we're isolating a variable. What happens if you divide the entire equation by 3 right now?

**Student:**
Oh, I'd have to divide everything... so it would be $$x - \frac{1}{3} = \frac{14}{3}$$. That looks more complicated.

**Teacher:**
Exactly. It's not wrong, but it's more work. So, what's a better way to get rid of that -1?

**Student:**
I should add 1 to both sides. That gives me $$3x = 14 + 1$$, which is $$3x = 15$$.

**Teacher:**
There you go! Now you're on the final step.

**Student:**
Now I just divide by 3. $$x = \frac{15}{3}$$, so $$x=5$$.

**Teacher:**
Fantastic. You navigated that perfectly. By handling the addition first, you kept the calculations much cleaner.
"""
