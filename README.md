# Multiple-PDF-Query-with-upvote-and-sources

`convert_webpages_to_PDFs.py` extracts text from webpages in `URLs.txt`, converts them to PDFs and saves them in `Docs`

`backend.py` creates and stores the vector embeddings of the PDF docs supplied in `Docs`

The bot has the following functionalities -
1. can query on multiple PDFs and repond
2. the user can upvote or downvote the bot's responses
3. the chat history and voting history are saved in separate files
4. the sources are also displayed along with the bot's responses
