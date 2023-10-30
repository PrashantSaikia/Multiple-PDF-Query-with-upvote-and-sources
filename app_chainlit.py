from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import chainlit as cl
import os, weaviate
from langchain.vectorstores.weaviate import Weaviate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def set_custom_prompt():
    system_template = """Use the following pieces of context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.
    The "SOURCES" part should be a reference to the source of the document from which you got your answer.

    Example of your response should be:

    ```
    The answer is foo
    SOURCES: xyz
    ```

    Begin!
    ----------------
    {summaries}"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt):

    # Setup remote client with openai authentication
    client = weaviate.Client(url='http://localhost:80', 
                             additional_headers={
                                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
                            })

    # Use the schema with 'LangChain' for index_name and 'text' for text_key
    # as defined for this document query application in the backend.py file.
    vectorstore = Weaviate(client, "LangChain2", "text", embedding=OpenAIEmbeddings(), attributes=["source"])

    llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0,
            openai_api_key= os.getenv("OPENAI_API_KEY"),
        )

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#QA Model Function
def qa_bot():
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


@cl.on_chat_start
async def init():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, you can ask me anything about Expireon. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])

    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
