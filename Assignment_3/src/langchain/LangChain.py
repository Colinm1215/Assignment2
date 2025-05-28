from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import RouterOutputParser, LLMRouterChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0)


def first_chain():
    lang_prompt = ChatPromptTemplate.from_template(
        "Identify the language of the following: {review}"
    )
    lang_chain = LLMChain(
        llm=llm,
        prompt=lang_prompt,
        output_key="language"
    )

    translate_prompt = ChatPromptTemplate.from_template(
        "Translate the following from {language} into English: {review}"
    )
    translate_chain = LLMChain(
        llm=llm,
        prompt=translate_prompt,
        output_key="english_review"
    )

    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize the following: {english_review}"
    )
    summarize_chain = LLMChain(
        llm=llm,
        prompt=summarize_prompt,
        output_key="summary"
    )

    reply_prompt = ChatPromptTemplate.from_template(
        "Draft a reply to {review} in {language}, based on the summary: {summary}."
    )
    reply_chain = LLMChain(
        llm=llm,
        prompt=reply_prompt,
        output_key="followup_message"
    )

    final_prompt = ChatPromptTemplate.from_template(
        "Translate the following from {language} into English: {followup_message}"
    )
    final_chain = LLMChain(
        llm=llm,
        prompt=final_prompt,
        output_key="English_followup_message"
    )

    return SequentialChain(
        chains=[lang_chain, translate_chain, summarize_chain, reply_chain, final_chain],
        input_variables=["review"],
        output_variables=["language", "english_review", "summary", "followup_message", "English_followup_message"],
        verbose=True,
    )


def second_chain():
    router_template = """
    You are an expert classifier. Decide which subject best fits the question.

    Subjects:
    - math
    - history
    - physics

    <question>
    {input}
    </question>

    <output>
    Return a markdown code snippet with a JSON object formatted to look like:
    '''json
    {{
        "destination": string \ Subject or None if none apply
        "next_inputs": string \ the original input : {input}
    }}
    '''
    """
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser()
    )

    router_chain = LLMRouterChain.from_llm(
        llm=llm,
        prompt=router_prompt
    )

    dest = {
        "math": LLMChain(llm=llm,
                         prompt=ChatPromptTemplate.from_template(
                             f"You are a top‑tier math professor. "
                             f"Answer the question in concise, rigorous terms.\n\n{{input}}"
                         ), output_key="text"),
        "history": LLMChain(llm=llm,
                            prompt=ChatPromptTemplate.from_template(
                                f"You are a top‑tier history professor. "
                                f"Answer the question in concise, rigorous terms.\n\n{{input}}"
                            ), output_key="text"),
        "physics": LLMChain(llm=llm,
                            prompt=ChatPromptTemplate.from_template(
                                f"You are a top‑tier physics professor. "
                                f"Answer the question in concise, rigorous terms.\n\n{{input}}"
                            ), output_key="text"),
        "None": LLMChain(llm=llm,
                         prompt=ChatPromptTemplate.from_template(
                             f"You are a top‑tier professor. "
                             f"Answer the question in concise, rigorous terms.\n\n{{input}}"
                         ), output_key="text"),
    }

    return MultiPromptChain(
        router_chain=router_chain,
        destination_chains=dest,
        default_chain=dest["None"],
        verbose=True,
    )


def third_chain_helper(llm_third_chain, prefix):
    desc_prompt = ChatPromptTemplate.from_template(
        f"{prefix} Write a detailed and informative product description for: {{input}}"
    )
    desc_chain = LLMChain(
        llm=llm_third_chain,
        prompt=desc_prompt,
        output_key="desc"
    )

    ad_prompt = ChatPromptTemplate.from_template(
        f"{prefix} Write a compelling and persuasive advertisement using this product description: {{desc}}"
    )
    ad_chain = LLMChain(
        llm=llm,
        prompt=ad_prompt,
        output_key="ad"
    )

    merge_prompt = ChatPromptTemplate.from_template(
        """
        Here is a product description: {desc}
        Here is an advertisement: {ad}
        Combine them into a single informative and persuasive text.
        """
    )
    merge_chain = LLMChain(
        llm=llm,
        prompt=merge_prompt,
        output_key="text"
    )

    return SequentialChain(
        chains=[desc_chain, ad_chain, merge_chain],
        input_variables=["input"],
        output_variables=["desc", "ad", "text"],
        verbose=True
    )


def third_chain(llm):
    router_template = """
    You are an expert classifier. Decide which category best fits the product.

    Categories:
    - tech
    - fashion
    - home

    <product_name>
    {input}
    </product_name>

    <output>
    Return a markdown code snippet with a JSON object formatted to look like:
    '''json
    {{
        "destination": string \ Category or None if none apply
        "next_inputs": string \ the original product name : {input}
    }}
    '''
    """
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser()
    )

    router_chain = LLMRouterChain.from_llm(
        llm=llm,
        prompt=router_prompt
    )

    dest_chains = {
        "tech": third_chain_helper(llm, "You are a tech business expert."),
        "fashion": third_chain_helper(llm, "You are a fashion business expert."),
        "home": third_chain_helper(llm, "You are a home business expert."),
        "None": third_chain_helper(llm, "You are a general business expert.")
    }

    return MultiPromptChain(
        router_chain=router_chain,
        destination_chains=dest_chains,
        default_chain=dest_chains["None"],
        verbose=True
    )


if __name__ == "__main__":
    review = (
        "Je trouve que la glace dans cette crèmerie est assez ordinaire.On sent un fort goût d'arômes artificiels, certaines glaces sont trop acides, voire pas très fraîches. Est‑ce que le personnel ici triche sur la qualité et ne prend pas soin de faire des produits de qualité ?"
    )
    chain = first_chain()
    result = chain.invoke({"review": review})
    for k, v in result.items():
        print(f"\n{k.upper()}:\n{v}")
    chain = second_chain()
    i = "What is the capital of France?"
    result = chain.invoke({"input": i})
    for k, v in result.items():
        print(f"\n{k.upper()}:\n{v}")
    chain = third_chain(ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7))
    product_name = "Smartphone with AI Camera"
    result = chain.invoke({"input": product_name})
    for k, v in result.items():
        print(f"\n{k.upper()}:\n{v}")