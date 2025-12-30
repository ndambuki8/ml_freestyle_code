from crewai import Agent, Task, Crew

researcher_agent = Agent(
    role='Researcher',
    goal='Find accurate information about {topic}',
    backstory='Expert at finding and verifying information',
    verbose=True
)

research_task = Task(
    description='Research {topic} and compile findings',
    agent=researcher_agent,
    expected_output='Detailed research report'
)

crew = Crew(
    agents=[researcher_agent],
    tasks=[research_task],
    verbose=True
)
result = crew.kickoff(input={'topic':'RAG Systems'})