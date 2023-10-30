import gradio as gr
from langchain.vectorstores.weaviate import Weaviate
import weaviate, os, json, pandas, datetime
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

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

system_template = """
Use the provided articles delimited by triple quotes to answer questions. If the answer cannot be found in the articles, write "I could not find an answer."
If you don't know the answer, just say "Hmm..., I'm not sure.", don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

The answer is foo
SOURCES:
1. abc
2. xyz

Begin!
----------------
{summaries}
"""

def create_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                                    chain_type="stuff",
                                                    reduce_k_below_max_tokens=True,
                                                    return_source_documents=True,
                                                    retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
                                                    chain_type_kwargs={"prompt": create_prompt_template()})

def append_row(df, row):
    return pandas.concat([
                df, 
                pandas.DataFrame([row], columns=row.index)]
           ).reset_index(drop=True)

def vote(data: gr.LikeData, history):
    df = pandas.DataFrame(columns=['Query', 'Response', 'Upvote', 'Time'])

    if data.liked:
        query = history[data.index[0]][data.index[1]-1]
        response = data.value
        upvote = 1
        time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

        new_row = pandas.Series({'Query': query, 'Response': response, 'Upvote': upvote, 'Time': time})
        df = append_row(df, new_row)

    else:
        query = history[data.index[0]][data.index[1]-1]
        response = data.value
        upvote = -1
        time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

        new_row = pandas.Series({'Query': query, 'Response': response, 'Upvote': upvote, 'Time': time})
        df = append_row(df, new_row)

    # append data frame to CSV file
    df.to_csv('vote_history.csv', mode='a', index=False, header=False)

with gr.Blocks(title='Expireon Documenatation GPT', theme=gr.themes.Soft(primary_hue=gr.themes.colors.lime), 
               css="footer {visibility: hidden}, .gradio-container {background-color: white}") as ui:

    gr.HTML("<div style='text-align: right'> <img src='/file=logo.jpg' width=300> </div>")
    gr.Markdown(
        """
        <h1><center>Documentation GPT v1.1</h1>
        I have been trained on all the PDF documents supplied to me. Ask me any question you may have on them!
    """)

    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(label="", placeholder="Enter a question")
            examples = gr.Examples(examples=["What is the product?",
                                             "Who are its competitors?",
                                             "What are its advantages compared to its competitors?",
                                             "Tell me about a case study"],
                                   inputs=[msg]
                                  )
            
        with gr.Column():
            chatbot = gr.Chatbot(label=" ")
            chatbot.like(vote, chatbot, None)  # Adding this line causes the like/dislike icons to appear in your chatbot
            clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history):
            response = chain({"question": message}, return_only_outputs=True)
            sources = [response['source_documents'][i].page_content for i in range(len(response['source_documents']))]
            sources_str = 'SOURCES:\n\n'
            for i in range(len(sources)):
                sources_str +=  f'\n================================================================\n\n{i+1}. ' + sources[i] + '\n-----------------------------------------------------------------\n'
            answer = response['answer']
            response = answer + '\n\n\n================================================================\n\n\n' + sources_str
            chat_history.append((message, response))            

            with open('chat_history.json', 'w') as file:
                    json.dump(chat_history, file)

            yield "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=True)

ui.queue().launch(server_name="0.0.0.0", server_port=8080)
