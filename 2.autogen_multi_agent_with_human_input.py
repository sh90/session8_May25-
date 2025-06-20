"""Objective:This example demonstrates how to integrate large language models (LLMs) into an application to enable complex, natural interactions with humans and gather specific information. This capability was difficult to achieve before the advent of LLMs. By assigning agents to focused, single-purpose tasks, we can keep the LLM on track—reducing divergence and significantly improving the likelihood of accurate task completion."""

from autogen import ConversableAgent
import data_info
print("Imported successfully")
configuration =  {"model": "gpt-4o-mini",   "api_key": data_info.open_ai_key    }

## Sequntial agents
# Step1. Personal_Info_Agent: Agent to get personal information
# Step2. issue agent: Agent to get customer feedback on issues
# Step3. engagement agent: engage customer
personal_info_agent = ConversableAgent(
    name="Personal_Info_Agent",
    system_message='''You are a helpful customer onboarding agent,
    you work for a phone provider called ACME.
    Your job is to gather the customer's name and location.
    Do not ask for any other information, only ask about the customer's name and location.
    After the customer gives you their name and location, repeat them 
    and thank the user, and ask the user to answer with TERMINATE to move on to describing their issue.''',
    llm_config=configuration,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "terminate" in msg.get("content").lower()

)
issue_agent = ConversableAgent(
    name="Issue_Agent",
    system_message='''You are a helpful customer onboarding agent,
    you work for a phone provider called ACME,
    you are here to help new customers get started with our product.
    Your job is to gather the product the customer use and the issue they currently 
    have with the product,
    Do not ask for other information.
    After the customer describes their issue, repeat it and add
    "Please answer with 'TERMINATE' if I have correctly understood your issue." ''',
    llm_config=configuration,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "terminate" in msg.get("content").lower()
)
engagement_agent = ConversableAgent(
    name="Engagement_Agent",
    system_message='''You are a helpful customer service agent.
    Your job is to gather customer's preferences on news topics.
    You are here to provide fun and useful information to the customer based on the user's
    personal information and topic preferences.
    This could include fun facts, jokes, or interesting stories.
    Make sure to make it engaging and fun!
    Return 'TERMINATE' when you are done.''',
    llm_config=configuration,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "terminate" in msg.get("content").lower(),
)
customer_proxy_agent = ConversableAgent(
    name="customer_proxy_agent",
    llm_config=False,
    code_execution_config=False,
    human_input_mode="ALWAYS",
)
#Conversation orchestration: Define how the conversation will happen

#  This means that we will define in which order agents will interact and who'll interact with who when.
#  To define this, we will use a list, that will contain several elements, each one corresponding to a chat.
#  The chats will then happen in that specific order.
chats = [] # Store all the covnersations

##1. Onboarding Agent with Customer
# We will now define the first chat and add it to this list.
# The first chat will be between our first agent, the  Personal Information Agent and the customer, who is going to be us.
# The first message will be sent by the Onboarding Personal Information Agent and will be:

# Carrying data to the next chat
"""
1. Onboarding Agent with Customer:
We will now define the first chat and add it to this list. 
The first chat will be between our first agent, the Onboarding Personal Information Agent and the customer, who is going to be us.

The first message will be sent by the Onboarding Personal Information Agent and will be:

Hello, I'm here to help you get started with our product. Could you tell me your name?

Carrying data to the next chat
In order to make the transition easier with the next agent, we are going to ask for a slightly different type of summary
 than we did before with this agent.
  We are going to request a summary generated by the LLM, but we will specify that the summary should return 
  the name and location of the customer in a JSON format: {'name': '', 'location': ''}. 
  This is a structured data format that can be easily read by another agent but also by another app or protocol. 
  This shows how an LLM agent can be used to interact with other apps.

Since we only want to transfer name and location to the next chat and we specifically specified how we want to transfer this data,
 we are going to add a new parameter, the clear_history to True which means that no data other than the one specified in the summary
  will be sent to the next chat. If we set it to False the agent from the next chat will be aware about the previous exchange with the user. 
  We'll use that later."""

chats.append(
    {
        "sender": personal_info_agent,
        "recipient": customer_proxy_agent,
        "message":
            "Hello, I'm here to help you solve any issue you have with our products. Could you tell me your name?",
        "summary_method": "reflection_with_llm",
        "summary_args": {
        "summary_prompt" : "Return the customer information into a JSON object only: "
                             "{'name': '', 'location': ''}",
        },
        "clear_history" : True
    }
)
""" 2. Issue Agent with Customer
We will now define the second chat and add it to this list. The second chat will be between our second agent, the Issue Agent and the customer, who is going to be us again.
The second message will be sent by the Onboarding Personal Information Agent and will be:
Great! Could you tell me what issue you're currently having and with which product?
This time we're going to generate a summary, but we won't specify any format or specifc data that must be carried over because we do not know what the exchange will yield specifically. We are also going to specify that we want to transfer the chat history to the next chat/agent."""

chats.append(
    {
        "sender": issue_agent,
        "recipient": customer_proxy_agent,
        "message":
                "Great! Could you tell me what issue you are currently having and with which product?",
        "summary_method": "reflection_with_llm",
        "clear_history" : False
    }
)

"""

3. Customer Engagement Agent with Customer
We will now define the third chat and add it to this list. The third chat will be between our third agent, the Customer Engagement Agent and the customer, who is going to be us again.

The third message will be sent by the Customer Engagement Agent and will be:

Can you tell me more about how you use our products or some topics interesting for you?

This time we're going to generate a summary so that the human agent can get this information in an easy and quick way when they take over the conversation, but we won't specify any format or specifc fata that must be carried over because we do not know what the exchange will yield specifically.

"""

chats.append(
        {
        "sender": customer_proxy_agent,
        "recipient": engagement_agent,
        "message": "While we're waiting for a human agent to take over and help you solve "
        "your issue, can you tell me more about how you use our products or some "
        "topics interesting for you?",
        "max_turns": 2,
        "summary_method": "reflection_with_llm",
    }
)
#chats


##Initiate the sequential chat
"""Now that we finished orchestrating the chat, we can get it started!
For this to work, you, the customer, will have to roleplay as a customer that currently have an issue with your phone provider. Let's say that your internet does not work, or that you want more bandwidth, or that you want some help to setup port forwarding to play a game with some friends or some other thing. Have fun doing some roleplay!"""

from autogen import initiate_chats

chat_results = initiate_chats(chats)
import pprint

for chat_result in chat_results:
    #pprint.pprint(chat_result.chat_history) # We could also get the whole chat history with this command
    pprint.pprint(chat_result.summary)

