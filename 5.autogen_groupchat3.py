"""

Critic ------>(Write and engaing blog about GenAI) Writer
Critic <------(GenAI is an ........) Writer
Critic -----> SEO reviewer
       -----> Legal reviewer
       -----> Ethical consideration reviewer
Critic will get suggestions from these reviewers and generate the possible enhancements to writer

Can you think of this in corporate terms?

Why do we need this?
One of the key limitations of LLMs like ChatGPT—as you may have noticed—is their struggle with handling complex tasks that involve multiple requirements. Typically, they miss or overlook parts of the task.

Nested chats help solve this by ensuring each requirement is addressed individually and thoroughly.

Additionally, LLMs rarely deliver the perfect result on the first try. You often have to guide them with follow-up prompts and refinements. With nested chats, you can automate these repeated refinement steps, saving time and improving consistency.


"""
# In this demo we will using assistance type agent

import autogen

import data_info

print("Impored successfully")
configuration =  {"model": "gpt-4o-mini", "api_key": data_info.open_ai_key    }

task = ''' Write a concise and engaging blogpost about power of GenAI within 250 words. '''

critic = autogen.AssistantAgent(
    name="Critic",
    llm_config=configuration,
    system_message="You are a critic. Your role is to review the writer’s work and provide clear, constructive feedback aimed at improving the quality, clarity, and impact of the content.",
)

writer = autogen.AssistantAgent(
    name="Writer",
    system_message="You are a writer who crafts engaging and concise blog posts, "
                   "complete with titles, on assigned topics. "
                   "Based on any feedback you receive, revise and polish your writing. "
                   "Return only the final version—no extra commentary or explanations.",
    llm_config=configuration,
)


SEO_reviewer = autogen.AssistantAgent(
    name="SEO_Reviewer",
    llm_config=configuration,
    system_message="You are an SEO reviewer, known for "
        "your ability to optimize content for search engines, "
        "ensuring that it ranks well and attracts organic traffic. " 
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role.",
)

legal_reviewer = autogen.AssistantAgent(
    name="Legal_Reviewer",
    llm_config=configuration,
    system_message="You are a legal reviewer, known for "
        "your ability to ensure that content is legally compliant "
        "and free from any potential legal issues. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role.",
)

ethics_reviewer = autogen.AssistantAgent(
    name="Ethics_Reviewer",
    llm_config=configuration,
    system_message="You are an ethics reviewer, known for "
        "your ability to ensure that content is ethically sound "
        "and free from any potential ethical issues. " 
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role. ",
)

meta_reviewer = autogen.AssistantAgent(
    name="Meta_Reviewer",
    llm_config=configuration,
    system_message="You are a meta reviewer, you aggregate and review "
    "the work of other reviewers and give a final suggestion on the content.",
)

"""
Chat orchestration
Nested chats
The way our chat is going to work is that when the writer will answer the critic, the critic will actually trigger a series of nested chats with each specialized reviewer (Critic -> Reviewer and then Reviewer -> Critic). We are also going to request from each reviewer that they send back their review in a specific format. Each review will send back a LLM generated summary of their review in the following JSON format:
{'Reviewer': '', 'Review': ''}
This will make it easier for the meta-reviewer to summarize all reviews.

We will also define here is a simple function called reflection_message() that will create the following nessage:

Review the following content.

"BLOGPOST PROPOSED BY WRITER"
We will call this function to create the message sent by the Critic to each specialized reviewer sequentially.

"""

def reflection_message(recipient, messages, sender, config):
    return f'''Review the following content. 
            \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''

"""

We are now going to define a new type of chat, a nested chat, that will trigger when the critic receives an answer. You can think about it like the inner monologue the Critic is having with other Reviewers that will help him provide the best possible criticism of the blogpost written by the reviewer. This is the structure our chat will follow:

Main chat:

Critic -> Writer : Initial task ("Write a concise but engaging blogpost ...")
Writer -> Critic : First version of the blogpost, this will trigger the nested chat
Nested chat:

Critic -> SEO reviewer: "Review the following content: blogpost"
SEO reviewer -> Critic: SEO review with context {'Reviewer': '', 'Review': ''}
Critic -> Legal reviewer: "Review the following content: blogpost"
Legal reviewer -> Critic: Legal review with context {'Reviewer': '', 'Review': ''}
Critic -> Ethics reviewer: "Review the following content: blogpost"
Ethics reviewer -> Critic: Ethics review with context {'Reviewer': '', 'Review': ''}
Critic -> Meta reviewer: "Aggregrate feedback from all reviewers and give final suggestions on the writing."
Meta reviewer -> Critic: Summary of all reviews with all contexts {'Reviewer': '', 'Review': ''}
Enf of nested chat

Back to the main chat:

Critic -> Writer : Summary of all reviews with all contexts {'Reviewer': '', 'Review': ''}
Writer -> Critic : Refined version of the blogpost based on all reviews.

"""

review_chats = [  # This is our nested chat
    {
        "recipient": SEO_reviewer,
        "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "summary_args":
            {
                "summary_prompt":
                    "Return review into as JSON object only:"
                    "{'Reviewer': '', 'Review': ''}. Here Reviewer should be your role",
            },
        "max_turns": 1},

    {
        "recipient": legal_reviewer,
        "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "summary_args": {"summary_prompt":
                             "Return review into as JSON object only:"
                             "{'Reviewer': '', 'Review': ''}.", },
        "max_turns": 1},

    {"recipient": ethics_reviewer,
     "message": reflection_message,
     "summary_method": "reflection_with_llm",
     "summary_args": {"summary_prompt":
                          "Return review into as JSON object only:"
                          "{'reviewer': '', 'review': ''}", },
     "max_turns": 1},

    {"recipient": meta_reviewer,
     "message": "Aggregate feedback from all reviewers and give final suggestions on the writing.",
     "max_turns": 1},
]

"""
Note how the message for each nested chat is going to be constructed by the `reflection_message() function we previously defined
Note how each specialized reviewer will send back their review in the requested JSON format {'reviewer': '', 'review': ''}

We now need to save and register this nested chat as a chat that will be triggered when the writer will contact the critic:
"""

critic.register_nested_chats(
    review_chats,
    trigger=writer,
)

"""
Main chat
Ok, we are now ready to start this chat. We will start this with the main chat that will trigger the Critic's nested chat as soon as the writed send back an first proposal blogpost answer to the critic.

Pay attention to the order in which the exchanges will happen and feel free to go back to the orchestration structure presented above to ensure that you understand how the nested chat works:

"""

chat_results = critic.initiate_chat(
    recipient=writer,
    message=task,
    max_turns=2,
    summary_method="last_msg"
)
