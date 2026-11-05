# Exam Prep Toolkit + AI Exam Agent (JEE / NEET / Boards / CUET)

Concept notes, solved examples, practice questions, calculation tools, and an AI agent that
solves questions step by step using those tools.

## The AI agent (new)

| File | Role |
|---|---|
| `exam_agent.py` | The agent. Reads a question (text or photo), looks up the chapter, calculates with tools, checks, and answers |
| `agent_tools.py` | The tools: `math_tool`, `calc`, `molar_mass`, `ph_calc`, `lookup_concept` |
| `concepts_db.py` | Core concept, key formulas, method and trap for 67 chapters across Maths, Physics, Chemistry, Biology |

### Set up

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your key"        # Windows PowerShell: $env:ANTHROPIC_API_KEY="your key"
```

### Use

```bash
python exam_agent.py "Find the area under min{f,g} where f = x^2 + 5/12 and g = 2(1 - 4|x|/3) for |x| <= 3/4, else 0. Find 9*area."
python exam_agent.py --image question.png            # photo or screenshot
python exam_agent.py --chat                          # follow-up questions
python exam_agent.py --selftest                      # offline test, no key needed
python agent_tools.py --selftest                     # test the tools
python concepts_db.py "projectile motion"            # look up a chapter's core concept
python concepts_db.py --list                         # list all chapters
```

Set `EXAM_AGENT_MODEL` to change the model name.

### How the agent works

1. Restates the question and how it read any ambiguous formula.
2. Looks up the chapter (`lookup_concept`) for the method, formulas and common trap.
3. Does every real calculation with exact tools instead of mental arithmetic.
4. Checks units, domain, a second method, and the "clean answer" test.
5. Gives GIVEN, CONCEPT, SOLUTION, FINAL ANSWER, CHECK.

## Study material

| File | What it is |
|---|---|
| `exam_concepts_JEE_NEET.md` | Concept notes: Maths, Physics, Chemistry, Biology |
| `solved_examples.md` | 30 worked examples with steps and common traps |
| `practice_questions.md` | 42 practice questions with answer key |
| `boards_CUET_and_other_exams.md` | Strategy for boards, CUET and other exams |
| `exam_solver_master_prompt.md` | Instructions to give any AI before it solves a question |
| `area_under_min_f_g_guide.md` | Method guide for area under min{f, g} |

## Python tools (no API key needed)

| File | What it is |
|---|---|
| `exam_practice.py` | Interactive practice quiz, 42 questions, standard library only |
| `exam_toolkit.py` | Command-line maths helper (limits, integrals, areas, ODEs, matrices, probability, projectile) |
| `area_min_solver.py` | Focused solver for area under min{f, g} |

## Honest limits

* The agent was tested offline (tools, loop logic, image encoding). A live run needs the `anthropic` package,
  an API key, and internet; it costs money and sends your question to Anthropic's servers.
* No system can solve every question correctly. Check the final answer, especially for blurry images.
* The concept database is a revision summary, not a full textbook. Use NCERT alongside it.
* Syllabi and exam patterns change. Confirm with the official source for your exam.
* The tools parse expressions with sympy after safety checks, but this is not a full sandbox. Run on your own machine.
