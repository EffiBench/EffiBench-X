from effibench.utils import EFFIBENCH_REGISTRY, EFFIBENCH_LANGS, get_md_lang

SEPARATOR = "\n\n"

def wrap_code_block(lang: str, code: str) -> str:
    return f"```{get_md_lang(lang)}\n{code}\n```"

TEST_RUNNER_EXAMPLES = {
    "cpp": {
        "instructions": """
- Put placeholder at global scope before the entire test runner
- DO NOT include header files or things like `using namespace std`, as they will be auto-injected during evaluation""",
        "code": """```cpp
// Possibly other helper classes seen in the starter code (e.g., ListNode, TreeNode, etc.). Place them BEFORE the placeholder

==Code Submission==

auto deserialize_stdin(const string& input) -> pair<vector<int>, int> {
    auto spacePos = input.find(' ');
    int target = stoi(input.substr(spacePos + 1));
    vector<int> nums;
    stringstream ss(input.substr(0, spacePos));
    for (string token; getline(ss, token, ',');)
        nums.push_back(stoi(token));
    return {nums, target};
}

auto serialize_stdout(const vector<int>& result) -> string {
    return to_string(result[0]) + "," + to_string(result[1]);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string line;
    getline(cin, line);
    
    auto [nums, target] = deserialize_stdin(line);
    Solution sol;
    auto ans = sol.twoSum(nums, target);
    cout << serialize_stdout(ans) << endl;
    return 0;
}
```""",
    },
    "java": {
        "instructions": """Java Version: OpenJDK 21.
- ⚠️ The Main class should be the first class in the file. The placeholder should be placed AFTER the entire Main class.
- ⚠️ AVOID unchecked/unsafe operations warnings.
- ⚠️ AVOID `unchecked conversion` warnings, either by using `@SuppressWarnings(\"unchecked\")` or by taking care of the type castings""",
        "code": """```java
public class Main {
    private static int[] deserializeIntArray(String input) {
        return Arrays.stream(input.split(","))
                     .mapToInt(Integer::parseInt)
                     .toArray();
    }
    
    private static String serializeOutput(int[] result) {
        return result[0] + "," + result[1];
    }
    
    public static void main(String[] args) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String[] input = br.readLine().trim().split(" ", 2);
        int target = Integer.parseInt(input[1]);
        int[] nums = deserializeIntArray(input[0]);
        
        Solution sol = new Solution();
        int[] ans = sol.twoSum(nums, target);
        System.out.println(serializeOutput(ans));
    }
}

==Code Submission==
```""",
    },
    "javascript": {
        "instructions": """
- ⚠️ Put placeholder at global scope before the test runner
- ⚠️ Ensure console.log outputs only the required result, nothing else (e.g., ANSI escape codes)
- ⚠️ Use exact numeric parsing (parseInt/Number as appropriate)""",
        "code": """```javascript
==Code Submission==

const deserialize_stdin = (input) => {
    const [arrPart, t] = input.trim().split(' ', 2);
    const target = +t;
    const nums = arrPart.split(',').map(Number);
    return { nums, target };
};

const serialize_stdout = (result) => {
    return result[0] + ',' + result[1];
};

process.stdin.resume();
process.stdin.setEncoding('utf8');
let input = '';
process.stdin.on('data', chunk => input += chunk);
process.stdin.on('end', () => {
    const { nums, target } = deserialize_stdin(input);
    const ans = twoSum(nums, target);
    console.log(serialize_stdout(ans));
});
```""",
    },
    "ruby": {
        "instructions": "Put placeholder at global scope before the test runner",
        "code": """```ruby
==Code Submission==

input = gets.strip
arr_part, target_str = input.split(' ', 2)
nums = arr_part.split(',').map(&:to_i)
ans = two_sum(nums, target_str.to_i)
puts "#{ans[0]},#{ans[1]}"
```""",
    },
    "golang": {
        "instructions": "Put placeholder at global scope before the entire test runner; DO NOT include imports and things like `package main`, as they will be auto-injected during evaluation",
        "code": """```go
==Code Submission==

func main() {
    reader := bufio.NewReader(os.Stdin)
    line, _ := reader.ReadString('\n')
    line = strings.TrimSpace(line)
    parts := strings.SplitN(line, " ", 2)
    target, _ := strconv.Atoi(parts[1])
    arrItems := strings.Split(parts[0], ",")
    nums := make([]int, len(arrItems))
    for i, s := range arrItems {
        nums[i], _ = strconv.Atoi(s)
    }
    ans := twoSum(nums, target)
    fmt.Printf("%d,%d\n", ans[0], ans[1])
}
```""",
    },
    "rust": {
        "instructions": "Put placeholder at global scope before the test runner",
        "code": """```rust
==Code Submission==

fn main() {
    let line = io::stdin().lock().lines().next().unwrap().unwrap();
    let (arr_part, target_str) = line.split_once(' ').unwrap();
    let target: i32 = target_str.parse().unwrap();

    let nums: Vec<i32> = arr_part.split(',')
        .map(|s| s.parse().unwrap())
        .collect();

    let ans = Solution::two_sum(nums, target);
    println!("{},{}", ans[0], ans[1]);
}
```""",
    },
    "python3": {
        "instructions": "Put placeholder at global scope before the test runner",
        "code": """```python
==Code Submission==

def deserialize_stdin(input_str):
    arr_str, target_str = input_str.split(' ', 1)
    nums = list(map(int, arr_str.split(',')))
    target = int(target_str)
    return nums, target

def serialize_stdout(result):
    return f"{result[0]},{result[1]}"

line = sys.stdin.read().strip()
nums, target = deserialize_stdin(line)
sol = Solution()
ans = sol.twoSum(nums, target)
print(serialize_stdout(ans))
```""",
    },
}

def prompt_solve_problem_functional(problem: str, target_lang: str, starter_code: str) -> str:
    """
    Prompt for solving a functional problem.
    
    Args:
        problem: Problem description in markdown format
        target_lang: Target programming language for solution
        starter_code: Starter code in the target language
        
    Returns:
        Full prompt for solving the functional problem
    """
    return f"""## Instructions
You are a competitive programming expert skilled in multiple programming languages.
Your task is to solve the following problem in {target_lang}:

## Problem Description
{problem}

## Starter Code
{wrap_code_block(target_lang, starter_code)}

## Output Format
- Provide the complete solution code in **one markdown code block** with appropriate language identifier. If your response has multiple code blocks, only the last one will be used.
- Implement the function with the exact signature (name, parameters, etc.) specified in the starter code.
- EXCLUDE ALL explanations, code comments, import/package/library statements, additional classes or functions outside of the starter code scope, or starting code like `if __name__ == "__main__":` or `func main()` or `package main` or `using namespace std;`.
- Use but do not redefine any helper data structures provided in the starter code (even if commented out).
"""

def prompt_solve_problem_io(problem: str, target_lang: str) -> str:
    """
    Prompt for solving an I/O problem.
    
    Args:
        problem: Problem description in markdown format
        target_lang: Target programming language for solution
        
    Returns:
        Full prompt for solving the I/O problem
    """
    return f"""## Instructions
You are a competitive programming expert skilled in multiple programming languages.
Your task is to solve the following problem in {target_lang}:

## Problem Description
{problem}

## Allowed Imports Scope
You may only import libraries within the scope defined below. Feel free to import specific functions/classes rather than entire modules (e.g., `from module import function` instead of importing the whole module).
{wrap_code_block(target_lang, EFFIBENCH_REGISTRY[target_lang]['imports'])}

## Output Format
- Provide the complete solution code in **one markdown code block** with appropriate language identifier. If your response has multiple code blocks, only the last one will be used.
- Your solution should read input directly from standard input (stdin) and write output directly to standard output (stdout).
- Your solution must handle all input parsing and output formatting as specified in the problem.
- Your solution should be directly runnable as a complete program. Include all necessary imports, package declarations, and a main function if required by the language.
"""

def prompt_translate_solution_functional(problem: str, target_lang: str, starter_code: str, reference_solutions: dict[str, str]) -> str:
    """
    Prompt for translating a solution for a functional problem to a different language.
    
    Args:
        problem: Problem description in markdown format
        target_lang: Target programming language for solution
        starter_code: Starter code in the target language
        reference_solutions: Dictionary of reference solutions in different languages
        
    Returns:
        Full prompt for translating the solution
    """
    return f"""{prompt_solve_problem_functional(problem, target_lang, starter_code)}

## Your solution should refer to the following solutions in other languages and translated from them. Your task is to create an equivalent solution in {target_lang}.
{SEPARATOR.join(f"{wrap_code_block(lang, code)}" for lang, code in reference_solutions.items() if code)}

## Allowed Imports Scope
You may only import libraries within the scope defined below. Feel free to import specific functions/classes rather than entire modules (e.g., `from module import function` instead of importing the whole module).
{wrap_code_block(target_lang, EFFIBENCH_REGISTRY[target_lang]['imports'])}

## Guidelines
- Ensure the solution is correct.
- Ensure the solution use the same algorithm and implementation approach as the reference solution (or one of the best in your opinion, if multiple solutions are provided).
- Write code that is idiomatic and follows best practices for {target_lang}.
"""

def prompt_translate_solution_io(problem: str, target_lang: str, reference_solutions: dict[str, str]) -> str:
    """
    Prompt for translating a solution for an I/O problem to a different language.
    
    Args:
        problem: Problem description in markdown format
        target_lang: Target programming language for solution
        reference_solutions: Dictionary of reference solutions in different languages
        
    Returns:
        Full prompt for translating the solution
    """
    return f"""{prompt_solve_problem_io(problem, target_lang)}

## Your solution should refer to the following solutions in other languages and translated from them. Your task is to create an equivalent solution in {target_lang}.
{SEPARATOR.join(f"{wrap_code_block(lang, code)}" for lang, code in reference_solutions.items() if code)}

## Guidelines
- Ensure the solution is correct (reads input and produces output exactly as specified).
- Ensure the solution use the same algorithm and implementation approach as the reference solution (or one of the best in your opinion, if multiple solutions are provided).
- Write code that is idiomatic and follows best practices for {target_lang}, especially for input/output handling.
"""

_TEST_RUNNER_REQUIREMENTS = """- **Core Workflow**: 
  1. Read raw input string from stdin until EOF
  2. Parse input string using consistent format-specific deserialization
  3. Invoke solution function/method with correctly typed parameters
  4. Serialize the return value to stdout using format-consistent encoding

- **Placeholder Requirement**:
  - ⚠️ Include EXACTLY one line containing ONLY the text `==Code Submission==`
  - During evaluation, this line will be replaced with submitted solution code
  - Solution code will contain complete implementation (e.g., "class Solution:\\n...")
  - Position placeholder at global/module scope, not within other code blocks

- **Serialization/Deserialization Requirements**:
  - ⚠️ Implement identical serialization/deserialization logic as Tasks 1-2
  - ⚠️ Minimize parsing complexity while maintaining correctness
  - ⚠️ Process whitespace, newlines, and edge cases consistently across languages
  - ⚠️ Maintain consistent numeric precision handling in all implementations

- **CRITICAL REQUIREMENTS**:
  - ⚠️ EXCLUDE ALL explanations and code comments
  - ⚠️ DO NOT INCLUDE ANY IMPORT STATEMENTS - all necessary imports will be auto-injected during evaluation
  - ⚠️ INCLUDE helper data structures (e.g., ListNode, TreeNode, etc., but NOT Solution which will be injected later) from starter code BEFORE the placeholder
  - ⚠️ DO NOT implement the solution yourself, as it comes from the placeholder"""

def prompt_test_suite_functional(problem: str, starter_code_dict: dict[str, str], solution: str, solution_lang: str = "python3") -> str:
    """
    Prompt for generating a test suite for a functional problem.
    
    Args:
        problem: Problem description in markdown format
        starter_code_dict: Dictionary of starter code for each language
        solution: Solution code in the solution language
        solution_lang: Language of the solution
        
    Returns:
        Full prompt for generating a test suite
    """
    langs = list(starter_code_dict.keys())
    test_runner_example_str = SEPARATOR.join(
        f"""#### Language: {lang}

**Auto-injected imports to expect**:
{wrap_code_block(lang, EFFIBENCH_REGISTRY[lang]['imports'])}

**Language-specific instructions**:
{TEST_RUNNER_EXAMPLES[lang]['instructions']}

**Test Runner Example**:
{TEST_RUNNER_EXAMPLES[lang]['code']}""" for lang in langs)
    
    return f"""## Instructions
Your task is to create a comprehensive test suite for the following programming problem to validate solutions in multiple languages. The test suite should consist of THREE components:

1. A Python function to generate test cases
2. A Python function to evaluate outputs
3. Test runner templates for each of the supported languages: {langs}

Complete each component in order, following specifications exactly.
DO NOT solve the coding problem itself.
Use identical serialization/deserialization logic across all components for consistency, but implement this logic independently in each component (no code sharing or imports between components).
Output in total 1 (Component 1) + 1 (Component 2) + {len(langs)} (Component 3) code blocks.

## Problem Description
{problem}

## Starter Code
{SEPARATOR.join(wrap_code_block(lang, code) for lang, code in starter_code_dict.items())}

## Sample Solution
{wrap_code_block(solution_lang, solution)}

## Component 1: Test Case Generator (Python)

**Implement** `generate_test_cases(num_cases: int, seed: int = 42) -> list[dict]`:

- **Purpose**: Create diverse test cases that thoroughly test the problem solution.
- **Arguments**: 
  - `num_cases`: Any valid integer number of test cases to generate, e.g., 10, 100, 1000, etc.
  - `seed`: Optional seed for reproducibility
- **Return**: List of dictionaries, each with exactly two keys:
  - `input`: String containing serialized test input
  - `output`: String containing serialized expected output
- **Serialization Requirements**:
  - ⚠️ Do NOT use JSON serialization
  - ⚠️ Use the simplest possible serialization format (e.g., plain string formats with basic delimiters).
  - Should be safely and easily parsed across all programming languages ({", ".join(langs)})
  - Serialize extreme cases carefully (e.g., empty string (""), empty list ([]), None/null, etc), as the test input will be passed to the program by stdin.
- **Implementation Notes**:
  - You should only implement the `generate_test_cases` function
  - You may create helper functions inside `generate_test_cases` function
  - You may import libraries inside the function
  - ⚠️ You MUST use the provided sample solution to generate expected outputs (adapt it into a function inside your code)
  - ⚠️ You MUST acknowledge the possible Constraints mentioned in the problem description, otherwise a correct solution may not be able to pass tests.
  - ⚠️ You MUST deduplicate test cases, so that each test case is unique and not redundant (should return `num_cases` number of DISTINCT test cases).

### Test Case Categories: Consider the following categories when generating test cases if applicable.
- **Boundary Cases**: Min/max values, empty inputs, single elements
- **Core Functionality**: Typical inputs testing main requirements
- **Stress Tests**: Inputs at maximum allowed constraints
- **Edge Cases**: Complex combinations that may cause unexpected behavior
- **Special Patterns**: Structured inputs (sorted, reversed, symmetrical)
- **Performance Traps**: Inputs that penalize inefficient algorithms
- **Data Structure Edge Cases**: Inputs challenging specific data structures

## Component 2: Test Case Evaluator (Python)

**Implement** `evaluate(expected_output: str, program_output: str) -> bool`:

- **Purpose**: Determine if a submitted solution is correct by comparing outputs.
- **Arguments**:
  - `expected_output`: Serialized expected output from test case generator
  - `program_output`: Serialized actual output from submitted solution
- **Return**: `True` if outputs match logically (not necessarily string equality), otherwise `False`
- **Comparison Logic**:
  - ⚠️ Implement the same deserialization logic used in Task 1
  - After deserializing, compare values appropriately according to the problem description (accounting for potential order differences in lists, etc.)
  - Keep comparison logic as simple as possible while being accurate
  - ⚠️ Since we generate test cases in Python, floating point numbers that save and load from JSON may have precision issues for different languages, and may cause evaluator to fail. Do not compare floating point numbers directly.
- **Implementation Notes**:
  - `program_output` may include trailing whitespace, newlines, etc. - handle appropriately
  - You may create helper functions inside the main function
- **Output**: Single Python code block

## Component 3: Test Case Runners ({len(langs)} languages)

For each language in: {langs}, implement a test case runner.
Create {len(langs)} code blocks, one for each language.

⚠️ **IMPORTANT**: DO NOT INCLUDE ANY IMPORT STATEMENTS in your test runners. All necessary imports will be auto-injected during evaluation. Test runners containing import statements will fail.

{_TEST_RUNNER_REQUIREMENTS}

### Language-Specific Test Runner Examples
{test_runner_example_str}

## Instructions (Reemphasized)
Complete each component in order, following specifications exactly. DO NOT solve the coding problem itself.
Use identical serialization/deserialization logic across all components for consistency, but implement this logic independently in each component (no code sharing or imports between components).
Output in total 1 (Component 1) + 1 (Component 2) + {len(langs)} (Component 3) code blocks.
"""

def prompt_test_suite_io(problem: str, solution: str, solution_lang: str = "python3") -> str:
    """Generates a prompt for creating test cases and evaluators for I/O problems.
    These problems read directly from stdin and write to stdout, without needing separate test runners.
    No starter code is provided for I/O problems.
    """

    return f"""## Instructions
Your task is to create a comprehensive test suite for the following programming problem to validate solutions in multiple languages. The test suite should consist of TWO components:

1. A Python function to generate test cases
2. A Python function to evaluate outputs

Complete each component in order, following specifications exactly. DO NOT solve the coding problem itself.
Use identical serialization/deserialization logic across all components for consistency, but implement this logic independently in each component (no code sharing or imports between components).
Output in total 1 (Component 1) + 1 (Component 2) code blocks.

## Problem Description
{problem}

## Sample Solution
{wrap_code_block(solution_lang, solution)}

## Component 1: Test Case Generator (Python)

**Implement** `generate_test_cases(num_cases: int, seed: int = 42) -> list[dict]`:

- **Purpose**: Create diverse test cases that thoroughly test the problem solution.
- **Arguments**: 
  - `num_cases`: Any valid integer number of test cases to generate, e.g., 10, 100, 1000, etc.
  - `seed`: Optional seed for reproducibility
- **Return**: List of dictionaries, each with exactly two keys:
  - `input`: String containing serialized test input
  - `output`: String containing serialized expected output
- **Serialization Requirements**:
  - ⚠️ Do NOT use JSON serialization
  - ⚠️ Use the simplest possible serialization format (e.g., plain string formats with basic delimiters).
  - Should be safely and easily parsed across all programming languages ({", ".join(EFFIBENCH_LANGS)})
  - Serialize extreme cases carefully (e.g., empty string (""), empty list ([]), None/null, etc), as the test input will be passed to the program by stdin.
- **Implementation Notes**:
  - You should only implement the `generate_test_cases` function
  - You may create helper functions inside `generate_test_cases` function
  - You may import libraries inside the function
  - ⚠️ You MUST refer to the provided sample solution to generate expected output for each test case. Adapt it into a function inside your code. If the provided sample solution is not in Python, you can translate it into Python carefully.
  - ⚠️ You MUST acknowledge the possible Constraints mentioned in the problem description, otherwise a correct solution may not be able to pass tests.
  - ⚠️ You MUST deduplicate test cases, so that each test case is unique and not redundant (should return `num_cases` number of DISTINCT test cases).

### Test Case Categories: Consider the following categories when generating test cases if applicable.
- **Boundary Cases**: Min/max values, empty inputs, single elements
- **Core Functionality**: Typical inputs testing main requirements
- **Stress Tests**: Inputs at maximum allowed constraints
- **Edge Cases**: Complex combinations that may cause unexpected behavior
- **Special Patterns**: Structured inputs (sorted, reversed, symmetrical)
- **Performance Traps**: Inputs that penalize inefficient algorithms
- **Data Structure Edge Cases**: Inputs challenging specific data structures

## Component 2: Test Case Evaluator (Python)

**Implement** `evaluate(expected_output: str, program_output: str) -> bool`:

- **Purpose**: Determine if a submitted solution is correct by comparing outputs.
- **Arguments**:
  - `expected_output`: Serialized expected output from test case generator
  - `program_output`: Serialized actual output from submitted solution
- **Return**: `True` if outputs match logically (not necessarily string equality), otherwise `False`
- **Comparison Logic**:
  - ⚠️ Implement the same deserialization logic used in Task 1
  - After deserializing, compare values appropriately according to the problem description (accounting for potential order differences in lists, etc.)
  - Keep comparison logic as simple as possible while being accurate
  - ⚠️ Since we generate test cases in Python, floating point numbers that save and load from JSON may have precision issues for different languages, and may cause evaluator to fail. Do not compare floating point numbers directly, use `math.isclose` instead. According to the problem description, use a proper floating point precision that is sufficient for distinguishing between correct and incorrect solutions to avoid numerical issues.
- **Implementation Notes**:
  - `program_output` may include trailing whitespace, newlines, etc. - handle appropriately
  - You may create helper functions inside the main function

## Important Notes
- Solutions will directly read from stdin and write to stdout
- Solutions must implement their own input/output handling
- Test cases should provide input formatted exactly as expected by stdin
- Expected output should match exactly what solutions should print to stdout
- Pay special attention to whitespace, newlines, and formatting in both input and output

## Instructions (Reemphasized)
Complete each component in order, following specifications exactly. DO NOT solve the coding problem itself.
Use identical serialization/deserialization logic across all components for consistency, but implement this logic independently in each component (no code sharing or imports between components).
Output in total 1 (Component 1) + 1 (Component 2) code blocks.
"""

def prompt_generate_new_test_runner(
    lang: str,
    problem: str,
    starter_code: str,
    test_case_generator: str,
    evaluator: str,
    test_runners: dict[str, str | None],
) -> str:
    """
    Prompt for generating a new test runner for a specific language.
    
    Args:
        lang: Target language for the test runner
        problem: Problem description in markdown format
        starter_code: Starter code in the target language
        test_case_generator: Test case generator code
        evaluator: Evaluator code
        test_runners: Dictionary of existing test runners for other languages
        
    Returns:
        Full prompt for generating a new test runner
    """
    return f"""## Instructions
Create a test runner for a programming problem in {lang}.
Study the serialization/deserialization logic from the provided test runners in other languages and the test case generator and evaluator.

## Problem Description
{problem}

## Starter Code
{starter_code}

## Test Case Generator
{wrap_code_block('python', test_case_generator)}

## Test Case Evaluator
{wrap_code_block('python', evaluator)}

## Valid Test Runners in Other Languages for THIS PROBLEM
{SEPARATOR.join([wrap_code_block(lang, code) for lang, code in test_runners.items() if code is not None])}

## Example Test Runner in {lang} (NOT FOR THIS PROBLEM BUT FOR REFERENCE):
{TEST_RUNNER_EXAMPLES[lang]['code']}

## Rules

⚠️ **IMPORTANT**: DO NOT INCLUDE ANY IMPORT STATEMENTS in your test runner. All necessary imports will be auto-injected during evaluation. Test runners containing import statements will fail.

{_TEST_RUNNER_REQUIREMENTS}

## Language-specific instructions**:
{TEST_RUNNER_EXAMPLES[lang]['instructions']}

## Auto-injected imports to expect**:
{wrap_code_block(lang, EFFIBENCH_REGISTRY[lang]['imports'])}

## Instructions (Reemphasized)
Please write a test runner for {lang}.
Your response should only contain one code block with the test runner code.
"""

def prompt_fix_test_runner(lang: str, err_msg: str, test_runner: str, full_code: str) -> str:
    """
    Prompt for fixing a test runner that encountered errors.
    
    Args:
        lang: Language of the test runner
        err_msg: Error message that occurred during testing
        test_runner: Current test runner code
        full_code: Full code including test runner and solution
        
    Returns:
        Full prompt for fixing the test runner
    """
    return f"""## Instructions
The previous test runner code for {lang} is encountering errors.
You can assume that the test cases and the solution are CORRECT - the problem is in the test runner implementation.

## Error Message
{err_msg}

## Test Runner Instructions for {lang} (Reemphasized)
{TEST_RUNNER_EXAMPLES[lang]['instructions']}

## Common Errors
- **PLACEHOLDER ISSUES**: Incorrect location or format of the placeholder
- **⚠️ NO IMPORTS ALLOWED**: DO NOT INCLUDE ANY IMPORT STATEMENTS. All imports are auto-injected before execution. The test runner should NOT contain any import, package, using, or require statements.
- **NUMERIC PRECISION ISSUES**: Serialization/deserialization is not robust to different languages so that evaluator fails on comparing the test case output and the program output despite the solution is correct
- **Serialization/Deserialization Logic**: Ensure that the serialization/deserialization logic is suitable for the input/output format defined in the test case generator and test case evaluator.

## The actual running code after injecting imports and replacing `==Code Submission==`
{wrap_code_block(lang, full_code)}

## Output Format
Provide the fixed test runner code in a single code block.
"""