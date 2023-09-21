from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

from dotenv import load_dotenv
load_dotenv()

# # my first langchain
# chat_model = ChatOpenAI()
# print(chat_model.predict("hi!"))

# create templates
system_template = """You are a fraud detection investigator for a big insurance company in the USA. 
Your primary job is to read an insurance claim report and classify the type of insurance report failed as one of three domains:  Medical fraud, Vehicle accident fraud, Home insurance fraud. 
Then once the domain is identified,  identify some key red flags in a report that raise suspicion of fraud. 
Finally summarize the findings in the following format. 
Output: Classification: , Red flags:
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(
    system_template)

input_claim = "{text}"
human_template = f"Insurance claim report: {input_claim} \n Output: "
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# create full prompt
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])

# process output


class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")


# build a chain
chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=chat_prompt,
)

claim = "Jamie is a 40-year-old man who met with an accident. Jamie’s car was in “Park” at the stop sign and in a blink of an eye, Jamie looked in the rear view mirror and boom. Jamie goes into shock and wakes up to 911 assistance. Jamie was asked to perform the basic concussion tests to make sure everything was alright. Jamie feels a slight pain in his lower back and since he is not at fault, he wants to claim it from the insurance company (ABC). Now, Manny works at ABC insurance agency and wants to make sure Jamie is provided the right service. Jamie wants the money upfront and he wants to get treated at his friend’s facility. Manny’s team did not agree for upfront payment as it was against policy; instead, they asked Jamie to visit the physicians that are part of ABC insurance chain. A week or two later, Jamie comes and submits a claim with all the receipts from another physician who is not part of the ABC program. Now, Manny’s team is suspicious about Jamie’s behaviour and wants to find out whether Jamie was treated at the facility. For this, the team did a manual search on Jamie’s social media profiles and found out that on the day Jamie submitted that he was being treated, he was at the bar with his friends at a party."
print(chain.run(text=claim))

# response received from chatbot is pretty impressive
'''
Classification: Vehicle accident fraud
Red flags:
1. Inconsistencies in the accident story - Jamie claims his car was in "Park" at a stop sign, but suddenly got hit from behind. This raises suspicion as it is unlikely for a parked car to be hit with such force.
2. Desire for upfront payment - Jamie's request for upfront payment instead of following the insurance company's policy raises suspicion. Fraudsters often try to get money upfront before providing any actual treatment or services.
3. Submitting receipts from a non-approved physician - Jamie submits receipts from a physician who is not part of the ABC insurance program, despite being advised to visit the approved physicians. This suggests a potential attempt to deceive the insurance company by seeking treatment elsewhere.
4. Evidence of fraudulent activity on social media - The team's manual search on Jamie's social media profiles reveals that he was at a party on the same day he claimed to be receiving treatment. This discrepancy raises doubts about the truthfulness of Jamie's claim.

'''
