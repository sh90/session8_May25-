"""

Critic ------>(Write an engaging blog about GenAI) Writer
Critic <------(GenAI is an ........) Writer
Critic -----> SEO reviewer
       -----> Legal reviewer
       -----> Ethical consideration reviewer
Critic will get suggestions from these reviewers and generate the possible enhancements to writer

Can you think of this in corporate terms?

"""
# In this demo we will using assistance type agent

import autogen

import data_info

print("Imported successfully")
configuration =  {"model": "gpt-4o-mini","api_key": data_info.open_ai_key    }

task = ''' Write a concise and engaging blogpost about power of GenAI within 250 words. '''

writer = autogen.AssistantAgent(
    name="Writer",
    system_message="You are a writer who crafts engaging and concise blog posts, "
                   "complete with titles, on assigned topics. "
                   "Based on any feedback you receive, revise and polish your writing. "
                   "Return only the final version—no extra commentary or explanations.",
    llm_config=configuration,
)


critic = autogen.AssistantAgent(
    name="Critic",
    llm_config=configuration,
    system_message="You are a critic. Your role is to review the writer’s work and provide clear, constructive feedback aimed at improving the quality, clarity, and impact of the content.",
)
chat_result = critic.initiate_chat(
    recipient=writer,
    message=task,
    max_turns=2,
    summary_method="last_msg"
)

import pprint

pprint.pprint(chat_result.summary)
