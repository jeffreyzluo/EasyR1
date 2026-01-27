import os
import json
import openai
import regex
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--responses_file', type=str, help="model responses")
args = parser.parse_args()

PROMPT = """You are ImageTaskEvaluatorGPT, an expert language model at judging whether or not a response is correct given an instruction in the context of an image. More specifically, you will be given the following:

1. An instruction: This is a question, an imperative request, or something similar about the image which requires a response.
2. A ground-truth response: This is the ground-truth response to the instruction.
3. A predicted response: This response is a model's answer to address the instruction in the context of the image without having access to the ground-truth response.

Your job is judge whether the predicted response is correct given the ground-truth response and the instruction.
 
Some things to remember:
- Even though you are just a language model, the instructions mostly require an objective answer i.e., the ground-truth response and instruction should be sufficient for you to judge the correctness of the predicted response. You do not need to have access to the complete image description. 
- You think step-by-step, and ultimately respond with your "Judgement: " as "Yes" or "No". Here, "Yes" implies that the predicted response is correct according to you, and "No" implies that the predicted response is not correct.
- Focus on whether the ground-truth response is equivalent to the predicted response or not. 

Instruction: {instruction}
Ground-truth Response: {groundtruth}
Predicted Response: {prediction}"""

def get_prompt(instruction, groundtruth, prediction):
    prompt = PROMPT.format(instruction=instruction, groundtruth=groundtruth, prediction=prediction)
    messages = [{"role": "user", "content": prompt}]
    return messages

def main():
    with open(args.responses_file, 'r') as f:
        data = json.load(f)
    model_name = args.responses_file.split("/")[-1].replace(".json","").strip()
    
    # Convert string answers to lists if needed
    if isinstance(data[0]['predicted_answers'], str):
        for i in range(len(data)):
            data[i]['predicted_answers'] = [data[i]['predicted_answers']]

    judgement_data = []
    outfile = f"evaluation/outputs/gpt4_{model_name}.json"
    print(f"saving to {outfile}")
    correct = 0

    for j in tqdm(range(len(data))):
        new_example = data[j].copy()
        instruction = data[j]['instruction']
        ground_truth = data[j]['ground_truth']
        model_response = data[j]['predicted_answers']
            
        completion = openai.chat.completions.create(
                model="gpt-4",
                messages=get_prompt(instruction, ground_truth, model_response)
        )
        output = completion.choices[0].message.content
        # print(output)
        search = regex.search("Judgement: ", output)
        if search == None:
            if output == "Yes":
                correct += 1
                new_example["gpt_correct"] = 1
                judgement_data.append(new_example)
            elif output == "No":
                new_example["gpt_correct"] = 0
                judgement_data.append(new_example)
            continue
        index = search.span()[1]                
        output = output[index:]
        if "Yes" in output:
            correct += 1
            new_example["gpt_correct"] = 1
            judgement_data.append(new_example)
        elif "No" in output:
            new_example["gpt_correct"] = 0
            judgement_data.append(new_example)
        else:
            continue
    
        if (j + 1) % 10 == 0:
            with open(outfile, 'w') as f:
                json.dump(judgement_data, f, indent=2)
    
    with open(outfile, 'w') as f:
        json.dump(judgement_data, f, indent=2)

    print(correct/len(data))

if __name__ == "__main__":
    main()