# LLM-practice
Practice for local LLM deployment and querying via Python code 

This README will be used mainly to track progress and write some basic notes, not a standard README

## Preamble
- Special thanks to the authors of Warner et al. 2025, and their GitHub repository located at https://github.com/ubcbraincircuits/SPIT_Generation/tree/main for all pre-generated llm patient files, patient prompt, interviewer question bank, and code snippet to start with


## Completed:
- Link VS Code to GitHub and successfully push edits to cloud
- Create basic document to interact with local LLM
- Created basic structure of final planned code/functions
- Borrowed code to query local LLM from paper
- Created variables for the pre-interview patient (LLM) prompt, transcript storage directory, and to stop interview
- Started writing main function, starting by reading the desired CSV based on the input variable
- Wrote code to randomly pick and extract a row of data, in addition to the respective patient name
- Wrote code to insert patient info into the initial LLM prompt document
- Created a variable that keeps track of the message history for the LLM to reference for each new question
- Wrote loop with assistance of Claude to conduct interview
- Wrote code with assistance of Claude to save transcript of interview as JSON file
- Adds some interview information to the front of the same JSON
- Claude suggested to add an additional safeguard at the end to prevent accidental calls for the main function
- Added helper functions to convert the respective spreadsheet row into a string
- Used Claude to create a helper function which can show the model's thinking when set to True, adjusted other code to incorporate
- Tested the file and adjusted various smaller issues and bugs (e.g. outputted line for random person not matching up with document)
- Compiled individual code blocks and readied for GitHub Commit


# Notes:

## Temperature, Top-K, and Top-P Sampling in LLMs (1)

### Temperature Sampling
- typical range 0 to 2
- controls randomness of output
- low = predictable text, favours high-probability words
- high = creative text, equalizes probability distribution, allows less probable words to appear more
- used for control over randomness


### Top-K Sampling 
- typical range 10 to 100
- limits model to choosing only from top *K* most likely words (i.e. limits to high probability words)
- high = more variety
- used for strict limits
- steps:
    - model ranks words by probability
    - keeps only top K words
    - randomly selects from list

### Top-P (Nucleus) Sampling in LLMs
- typical range 0.8 to 0.95
- selects words based on cumulative probability (not fixed options)
- high = more creative
- used for flexible control
- steps:
    - words ranked by probability
    - added to selection one by one until combined probability reaches/exceeds P
    - randomly select word from selection


## Sources:
1. geeksforgeeks.org