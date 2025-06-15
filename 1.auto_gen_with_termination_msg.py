from autogen import ConversableAgent

import data_info

print("Impored successfully")
configuration =  {"model": "gpt-4o-mini", "api_key": data_info.open_ai_key }

# Configure a Conversable Agent that will never ask for our input and uses the config provided
nba_fan = ConversableAgent(
    name="nba", # Name of the agent
    llm_config=configuration,
    system_message="Your name an nba fan in an discussion with soccer fan "
                   "When you're ready to end the conversation, say 'I gotta go'.",
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "I gotta go" in msg["content"],

)

soccer_fan = ConversableAgent(
    name="soccer", # Name of the agent
    llm_config=configuration, # The Agent will use the LLM config provided to answer
    system_message="Your name an soccer fan in an discussion with nba fan "
                   "When you're ready to end the conversation, say 'I gotta go'.",
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "I gotta go" in msg["content"],
)

chat_result = nba_fan.initiate_chat(
    recipient = soccer_fan,
    message="convince me that soccer is better than nba",
)

# send method
#nba_fan.send(message="What was the last argument I made?",recipient=soccer_fan)
