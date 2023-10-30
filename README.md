# Multiple-PDF-Query-with-upvote-and-sources

App to query on multiple PDFs and repond. The sources are also displayed along with the bot's responses.

`convert_webpages_to_PDFs.py` extracts text from webpages in `URLs.txt`, converts them to PDFs and saves them in `Docs`

`backend.py` creates and stores the vector embeddings of the PDF docs supplied in `Docs`

The gradio bot has the following added functionalities -
1. the user can upvote or downvote the bot's responses
2. the chat history and voting history are saved in separate files

The chainlit bot has the following added functionalities -
1. Oauth enabled (need to set up the Oauth Client ID, Oauth secret, Chainlit Auth Secret and the Chainlit API key in `.env`)
2. the chat history and voting history are saved in https://cloud.chainlit.io/
    
