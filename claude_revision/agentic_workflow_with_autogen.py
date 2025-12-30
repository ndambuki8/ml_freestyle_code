from autogen import AssistantAgent, UserProxyAgent,  GroupChat, GroupChatManager

# Configure LLM
llm_config = {
    "model": "gpt-4",
    "api_key": "your_key",
    "temperature": 0.7
}

# create specialized agents 
researcher = AssistantAgent(
    name="Researcher",
    system_message="You research topics and gather information from the web."
    llm_config=llm_config
)

analyst = AssistantAgent(
    name="Analyst",
    system_message="You analyze data and provide insights",
    llm_config=llm_config
)

writer =  AssistantAgent(
    name="Writer",
    system_message="You write clear comprehensive reports",
    llm_config=llm_config
)

# user proxy for execution
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="TERMINATE",
    code_execution_config={"work_dir":"coding", "use_docker":False}
)

# create a group chat
groupchat = GroupChat(
    agents=[researcher, analyst, writer, user_proxy],
    messages=[],
    max_round=10
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# execute the multiagent workflow
user_proxy.initiate_chat(
    manager,
    message="Research latest trends in llms and, analyze data and write a report"
)



