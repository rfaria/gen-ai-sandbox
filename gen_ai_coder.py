import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_response(prompt, system = '', temperature = 1.0):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content": prompt},
                   {"role": "system", "content": system}],
        temperature = temperature
    )
     return response['choices'][0]['message']['content']
 

if __name__ == '__main__':
  system_prompt = '''
    You are a helpful AI assistant for data analysis, providing commented code for human review.
    You put any code inside a Python code block in Markdown.
    You include a bulleted code explanation after the code.'''

  message_prompt = ''' 
    Generate Seaborn code for producing a single bar graph.
    Include matplotlib and seaborn import statements. Pandas has already been imported.
    Create a single bar graph for each Team for their Win and Loss columns. I want the Wins and Losses on the same bar. Win should be green and Loss should be red.
    Let's only include teams that have a higher win rate than their loss rate
    Include code for title and axis labels.'''

  gpt_response = get_response(message_prompt, system_prompt)
  print(gpt_response)
