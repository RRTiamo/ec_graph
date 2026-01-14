import uvicorn
from fastapi import FastAPI

from web.schema import Question, Answer
from web.service import ChatService

app = FastAPI()


@app.post("/api/chat")
def read_item(question: Question) -> Answer:
    reply = ChatService().chat(question)
    return Answer(message=reply)


if __name__ == '__main__':
    uvicorn.run("web.app:app", port=9999, host="0.0.0.0")
