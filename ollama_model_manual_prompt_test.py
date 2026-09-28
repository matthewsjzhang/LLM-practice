import random
import json

from pathlib import Path
from datetime import datetime

import pandas as pd

from langchain_ollama import OllamaLLM

##### Documentation used
"""
- pandas guides -> isna(), read.csv(), iloc()
- some stack overflow to clarify code and functions
- official Python documentation -> various built in operators and functions (.items, .split, .strip, .randint, etc.)
"""

##### Model setup

"""
Only model used to play patient. Human will be interviewer, manually
entering questions on by one from "questionbank_v2.txt."
"""

model = OllamaLLM(
    base_url = "localhost:11434", # default Ollama port, 11435 used in paper, but possibly due to remote Ollama servers with SSH tunneling (setting adjacent port)
    model = "llama3.3-8B:latest", # as detailed in "ollama list"
    temperature = 0.9,  # randomness in token predictions
    num_ctx = 6144,     # max tokens for the context window
    top_k = 40,         # limits choices to most probably tokens
    top_p = 0.9)        # max cumulative probability for selecting tokens

##### Prompt templates

patient_prompt_path = 'patient_prompt_v2.txt' # to access the prompt provided initially for the LLM
transcript_dir = "transcripts"                # sets directory to save new interview transcripts at the end

stop_keyword = "<STOP>"                       # sets keyword to enter into chat to end interview and generate interview transcript

show_thinking = False # when set to True -> prints model's reasoning, otherwise hides it (need to debug not always working)


##### HELPER FUNCTION:

"""
turn a CSV row into a list of key values for the prompt to reply with
"""

def patient_row_to_str(patient_row, ignore_keys=("Clinician Name", "Appointment Date")): # removing two non-patient relevant details
    lines = []
    for key, value in patient_row.items():
        if key in ignore_keys:
            continue
        if pd.isna(value): # True if value is NaN
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


##### HELPER FUNCTION:

"""
extracts answers from the raw model output from the paper code. makes it
easier to read the outputs on the terminal

NOTE: Used Claude to code this helper function
"""

def split_response(raw_response):
    split_key = "RESPONSE:"
    if split_key in raw_response:
        thinking, response = raw_response.split(split_key, 1)
        return thinking.strip(), response.strip()
    return "", raw_response.strip()  


##### MAIN FUNCTION:

def run_interview(csv_path):
    """
    loads one llm_patients_######.csv file and randomly picks patient
    row. then insert patient info into patient_prompt_v2.txt, and run 
    interview in terminal window.

    at end, typing message containing stop keyword ends interview. full 
    conversation, including CSV row used saved to JSON file in transcript 
    directory so accuracy of info extraction can be verified
    """

    # load CSV
    patients_df = pd.read_csv(csv_path, sep = "|")

    # randomly pick row

    row_index = random.randint(0, len(patients_df) - 1) # generates random data row for the spreadsheet
    patient_row = patients_df.iloc[row_index] # outputs each row into pairs? of data, with each header being paired with the respective data entry (or NaN if nothing/blank)
    patient_name = patient_row.get("Full Name", "Unknown")

    print(f"Selected patient: '{patient_name}' - row {row_index + 2} in {csv_path}") # initial output of information before any interaction to help fact check

    # create patient info for llm

    patient_info_str = patient_row_to_str(patient_row)
    prompt_template = Path(patient_prompt_path).read_text()
    patient_prompt = prompt_template.replace("{patient_info}", patient_info_str) # replaces placeholder [patient info] section in template with string that contains all paired patient info for the LLM to use

    # send stuff to LLM
    message_list = [{"role": "system", "content": patient_prompt}] 
    
    """
    tradeoff here. opted for "memory" of entire conversation so follow up 
    questions can be asked in consecutive queries (using .append instead of 
    resetting the message and having the model reply to a single query each 
    time), rather than complex "chunking" algorithm used in paper. however, 
    each individual question will add up so prolonged interview will slow 
    model down since it needs to process the entire message history for each 
    new question
    """

    transcript = []

    print(f"Interview started, ask questions one at at time. Type {stop_keyword} to end")

    while True: # loops interview "state" until the interviewer ends using <STOP>
        user_input = input("You: ") # space for human interviewer to type

        if stop_keyword in user_input:
            print("\nInterview ended") 
            break   # ends interview if human types <STOP>
        
        message_list.append({"role": "user", "content": user_input}) # adds interviewer question into prompt inputted to LLM 
        transcript.append({"role": "user", "content": user_input}) # adds the question to the final transcript
 
        raw_response = model.invoke(message_list) # queries LLM with question from interviewer
        thinking, patient_reply = split_response(raw_response) # splits response into the model thinking? and reply -> Claude helper function
 
        if show_thinking and thinking:
            print(f"Thinking:\n{thinking}\n") # if asked to show thinking, model will also output thinking in response, otherwise just outputs the in character response
 
        print(f"Patient: {patient_reply}\n") # prints LLM response for interviewer to see
 
        message_list.append({"role": "assistant", "content": patient_reply}) # adds LLM response to message list
        transcript.append({"role": "assistant", "content": patient_reply})   # adds LLM response to the final transcript

    # save convo
    Path(transcript_dir).mkdir(parents=True, exist_ok=True) # set path to inside parent directory "transcripts" only creating new folder once
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")    # takes current timestamp for file name, in format YearMonthDay-24HourMinuteSecond
    safe_name = str(patient_name).replace(" ", "_")         # creates a new "safe" for Windows explorer name, replacing all spaces in the patient's full name with underscores
    out_path = Path(transcript_dir, f"interview_{safe_name}_{timestamp}.json") # sets the final path for the JSON file and names it according to previous parameters
 
    # formatting final JSON file contents (want to add output_obj to the existing out_path JSON)

    output_obj = {
        "csv_file": str(csv_path),
        "row_index": row_index,
        "patient_name": patient_name,
        "conversation": transcript,
    }
 
    with open(out_path, "w") as f: # opens the formatted and renamed JSON transcript to write, and redefine as variable "f"
        json.dump(output_obj, f, indent=4) # updates JSON by adding interview details to the front of the transcript, and indents all transcript conversation to the 4th indent line
 
    print(f"Transcript saved to {out_path}")


if __name__ == "__main__": # prevents run_interview() function from being imported and run elsewhere
    
    csv_path = "llm_patients_031925.csv"

    # KEY: run LLM start batch file if not done already
    
    run_interview(csv_path)