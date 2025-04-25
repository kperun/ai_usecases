# Building Code Agents with Hugging Face smolagents

This code is based on the Deeplearning Ai course "Building Code Agents with Hugging Face smolagents" [1]. All credits
to the original creators of the content.

Used Frameworks:
- smolagent: For coding and code execution. Smolagent runs code as python script instead of generating json which is used to invoke 
functions. The difference being that code executed as script is generated in one step, i.e., the agent has to create only one script
which is then executed, thus not several invocations for intermediate steps are required. The final result is generated directly.

[1] https://learn.deeplearning.ai/courses/building-code-agents-with-hugging-face-smolagents