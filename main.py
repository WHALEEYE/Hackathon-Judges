from pydantic import BaseModel
from openai import OpenAI
from textwrap import dedent

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.tasks import Task
from camel.types import ModelPlatformType, ModelType
from camel.workforce import Workforce
from camel.toolkits import SearchToolkit, OpenAIFunction

import json


def make_judge(
    persona: str,
    example_feedback: str,
    criteria: str,
) -> ChatAgent:
    msg_content = dedent(
        f"""\
        You are a judge in a hackathon.
        This is your persona that you MUST act with: {persona}
        Here is an example feedback that you might give with your persona, you MUST try your best to align with this:
        {example_feedback}
        When evaluating projects, you must use the following criteria:
        {criteria}
        You also need to give scores based on these criteria, from 1-4. The score given should be like 3/4, 2/4, etc.
        """  # noqa: E501
    )

    sys_msg = BaseMessage.make_assistant_message(
        role_name="Hackathon Judge",
        content=msg_content,
    )

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict=ChatGPTConfig().as_dict(),
    )

    agent = ChatAgent(
        system_message=sys_msg,
        model=model,
    )

    return agent


def create_judge_wf() -> Workforce:
    search_toolkit = SearchToolkit()
    search_tools = [
        OpenAIFunction(search_toolkit.search_google),
        OpenAIFunction(search_toolkit.search_duckduckgo),
    ]

    researcher_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict=ChatGPTConfig().as_dict(),
    )

    researcher_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Researcher",
            content="You are a researcher who does research on AI and Open"
            "Sourced projects. You use web search to stay updated on the "
            "latest innovations and trends.",
        ),
        model=researcher_model,
        tools=search_tools,
    )

    vc_persona = (
        "You are a venture capitalist who is obsessed with how projects can "
        'be scaled into "unicorn" companies. You peppers your speech with '
        'buzzwords like "disruptive," "synergistic," and "market penetration."'
        " You do not concerned with technical details or innovation unless "
        "it directly impacts the business model."
    )

    vc_example_feedback = (
        '"Wow, this project is absolutely disruptive in the blockchain-enabled'
        " marketplace! I can definitely see synergistic applications in the "
        "FinTech ecosystem. The scalability is through the roof--this is "
        "revolutionary!"
    )

    vc_criteria = dedent(
        """\
        ### **Applicability to Real-World Usage (1-4 points)**
        - **4**: The project directly addresses a significant real-world problem with a clear, scalable application.
        - **3**: The solution is relevant to real-world challenges but requires more refinement for practical or widespread use.
        - **2**: Some applicability to real-world issues, but the solution is not immediately practical or scalable.
        - **1**: Little or no relevance to real-world problems, requiring substantial changes for practical use.
        """  # noqa: E501
    )

    vc_agent = make_judge(
        vc_persona,
        vc_example_feedback,
        vc_criteria,
    )

    eng_persona = (
        "You are an experienced engineer and a perfectionist. You are highly "
        "detail-oriented and critical of any technical flaw, no matter how "
        "small. He evaluates every project as though it were going into a "
        "mission-critical system tomorrow, so his feedback is thorough but "
        "often harsh."
    )

    eng_example_feedback = (
        "There are serious code inefficiencies in this project. The "
        "architecture is unstable, and the memory management is suboptimal. "
        "I expect near-perfect performance, but this solution barely functions"
        " under stress tests. It has potential, but it is nowhere near "
        "deployment-ready."
    )

    eng_criteria = dedent(
        """\
        ### **Technical Implementation (1-4 points)**
        - **4**: Flawless technical execution with sophisticated design, efficient performance, and robust architecture.
        - **3**: Strong technical implementation, though there may be areas for improvement or further development.
        - **2**: The project works, but technical limitations or inefficiencies hinder its overall performance.
        - **1**: Poor technical implementation with major issues in functionality, coding, or structure.
        """  # noqa: E501
    )

    eng_agent = make_judge(
        eng_persona,
        eng_example_feedback,
        eng_criteria,
    )

    founder_persona = (
        "You are a well-known AI startup founder who is always looking for the"
        ' "next big thing" in AI. You value bold, inventive ideas and '
        "prioritizes projects that break new ground over those that improve "
        "existing systems."
    )

    founder_example_feedback = (
        "This is interesting, but I have seen similar approaches before. I am "
        "looking for something that pushes boundaries and challenges norms. "
        "What is the most revolutionary part of this project? Let us see what "
        "is trending on Internet to make sure this is not already out there!"
    )

    founder_criteria = dedent(
        """\
        ### **Innovation (1-4 points)**
        - **4**: The project showcases a groundbreaking concept or a unique approach that significantly departs from existing methods.
        - **3**: The project demonstrates a novel twist on known solutions or introduces some innovative aspects.
        - **2**: Some level of innovation is present, but the project largely builds on existing ideas without major new contributions.
        - **1**: Little or no innovation; the project is based on standard approaches with minimal creativity.
        """  # noqa: E501
    )

    founder_agent = make_judge(
        founder_persona,
        founder_example_feedback,
        founder_criteria,
    )

    contributor_persona = (
        "You are a contributor to the CAMEL-AI project and is always excited "
        "to see how people are using it. You are kind and optimistic, always "
        "offering positive feedback, even for projects that are still rough "
        "around the edges."
    )

    contributor_example_feedback = (
        "Oh, I love how you have implemented CAMEL-AI here! The use of its "
        "adaptive learning capabilities is fantastic, and you have really "
        "leveraged the contextual reasoning in a great way! Let me just pull "
        "up the GitHub README to check if there is any more potential "
        "optimizations."
    )

    contributor_criteria = dedent(
        """\
        ### **Use of CAMEL-AI (1-4 points)**
        - **4**: Excellent integration of CAMEL-AI, fully leveraging its advanced features like in-context learning, adaptability, or multi-domain applications.
        - **3**: Good use of CAMEL-AI, but there are opportunities to exploit more of its advanced capabilities.
        - **2**: Limited use of CAMEL-AI, relying mostly on basic features without taking advantage of its full potential.
        - **1**: CAMEL-AI integration is minimal or poorly implemented, adding little value to the project.
        """  # noqa: E501
    )

    contributor_agent = make_judge(
        contributor_persona,
        contributor_example_feedback,
        contributor_criteria,
    )

    workforce = Workforce("Hackathon Judges")

    workforce.add_single_agent_worker(
        "Visionary Veronica (Judge), a venture capitalist who is "
        'obsessed with how projects can be scaled into "unicorn" companies',
        worker=vc_agent,
    ).add_single_agent_worker(
        "Critical John (Judge), an experienced engineer and a" " perfectionist.",
        worker=eng_agent,
    ).add_single_agent_worker(
        "Innovator Iris (Judge), a well-known AI startup founder who"
        ' is always looking for the "next big thing" in AI.',
        worker=founder_agent,
    ).add_single_agent_worker(
        "Friendly Frankie (Judge), a contributor to the CAMEL-AI "
        "project and is always excited to see how people are using it.",
        worker=contributor_agent,
    ).add_single_agent_worker(
        "Researcher Rachel (Helper), a researcher who does online searches to"
        " find the latest innovations and trends on AI and Open Sourced "
        "projects.",
        worker=researcher_agent,
    )

    return workforce


def genarate_feedback(project_description: str) -> str:
    workforce = create_judge_wf()

    task = Task(
        content="Evaluate the hackathon project. First, do some research on "
        "the infomation related to the project, then each judge should give a"
        " score accordingly. Finally, list the opinions from each judge while"
        " preserving the judge's unique identity, along with the score and"
        " judge name, and also give a final summary of the opinions.",
        additional_info=project_description,
        id="0",
    )

    task = workforce.process_task(task)
    return task.result


class Opinion(BaseModel):
    judge_name: str
    score: int
    comment: str


class Feedback(BaseModel):
    opinions: list[Opinion]
    summary: str


def parse_feedback(feedback_text: str, openai_client: OpenAI) -> Feedback:

    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract the feedback information."},
            {"role": "user", "content": feedback_text},
        ],
        response_format=Feedback,
    )

    feedback = completion.choices[0].message.parsed
    return feedback


def main():
    client = OpenAI()
    project_id = "Goodheart"
    project_description = dedent(
        """\
        The WasteNot Market project incorporates innovative aspects that set it apart from traditional food distribution methods, particularly through the utilization of blockchain technology. Here’s an exploration of these innovative features:

        Innovative Aspects of WasteNot Market
        Blockchain Technology Utilization:
        Transparency and Traceability: WasteNot Market employs blockchain technology to create a transparent and traceable supply chain for food products. Each transaction and movement of food items is recorded on the blockchain, allowing consumers to trace the origin of their food. This transparency builds trust among consumers regarding the quality and safety of the food they purchase.
        Smart Contracts: The use of smart contracts on the blockchain automates transactions between food producers, retailers, and consumers. These contracts can ensure that surplus food is sold at predetermined prices, reducing the need for intermediaries and streamlining the distribution process.
        Real-Time Data Sharing: Blockchain enables real-time sharing of data regarding food availability, pricing, and expiration dates. This information helps consumers make informed decisions and encourages timely purchases of surplus food, reducing waste.
        Decentralized Distribution Model:
        Unlike traditional food distribution methods that rely heavily on centralized systems, WasteNot Market adopts a decentralized approach. This model allows local producers and retailers to connect directly with consumers, minimizing the distance food travels and reducing carbon footprints associated with transportation.
        By empowering local communities, the platform fosters a sense of ownership and responsibility towards food waste reduction and sustainability.
        Community-Driven Solutions:
        WasteNot Market encourages community involvement through volunteer programs and local partnerships. This grassroots approach contrasts with traditional food distribution systems that often overlook community engagement. By involving local volunteers in food collection and distribution, the project strengthens community ties and raises awareness about food waste issues.
        Dynamic Pricing Models:
        The platform utilizes dynamic pricing strategies based on supply and demand, which is facilitated by blockchain data analytics. This approach allows for flexible pricing of surplus food, ensuring that it remains affordable while also providing fair compensation to producers.
        Educational Integration:
        WasteNot Market integrates educational initiatives within its platform, utilizing blockchain to provide consumers with information about food waste, sustainability practices, and the environmental impact of their food choices. This educational aspect is often absent in traditional food distribution models. ### Conclusion The innovative aspects of the WasteNot Market project, particularly its use of blockchain technology, represent a significant departure from traditional food distribution methods. By enhancing transparency, fostering community engagement, and implementing decentralized solutions, WasteNot Market not only addresses food waste but also promotes a sustainable and equitable food system. This approach exemplifies how technology can be leveraged to create positive social and environmental impacts in the food industry.
        """  # noqa: E501
    )

    feedback_text = genarate_feedback(project_description)

    feedback = parse_feedback(feedback_text, client)

    with open(f"output/{project_id}.json", "w") as f:
        result_json_str = feedback.json()
        # format the json string
        result_json_str = json.dumps(json.loads(result_json_str), indent=4)
        f.write(result_json_str)


if __name__ == "__main__":
    main()
