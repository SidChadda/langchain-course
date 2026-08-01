from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama



load_dotenv()


def main():
    print("Hello from langchain-course!")
    print("sid")

    information = """Warranty Claim – Vehicle Failure Notification

    To whom it may concern,

    I am submitting a formal warranty claim regarding a mechanical failure in my vehicle. The details are as follows:

    VIN: 1HGCM82633A004521

    Manufacturer: Honda Motor Co.

    Make & Model: Honda Accord

    Model Year: 2023

    Reported Error / Failure: The vehicle is displaying repeated Powertrain Control Module (PCM) Failure – Error Code P0606. This issue results in sudden loss of acceleration, intermittent stalling, and failure of onboard diagnostic communication. The malfunction occurs without warning and poses a significant safety concern.

    Given that the vehicle is still within the manufacturer’s warranty period, I request an immediate inspection, diagnostic confirmation, and full warranty repair or replacement of the defective components.

    Please confirm receipt of this claim and advise on the next steps for processing.

    Thank you for your prompt attention."""


    warranty_claim_template = PromptTemplate(
        input_variables=["information"],
        template="""
    You are a customer service representative for an automotive company. 
    A customer has submitted the following warranty claim information:

    {information}

    From the information provided, identify the make, model, year, vin of the car and the error it gave. 
    Give output in the below format:

    Make: {{make}}
    Model: {{model}}
    Year: {{year}}
    VIN: {{vin}}
    Error: {{error}}
    """
    )


    #llm=ChatOpenAI(model="gpt-5", temperature=0)
    llm=ChatOllama(model="gemma3:270m", temperature=0, format="json")
    chain=warranty_claim_template | llm
    response=chain.invoke(input={"information": information})
    print(response.content)



if __name__ == "__main__":
    main()
