from pydantic import BaseModel, Field


class Question(BaseModel):
    message: str = Field(description="发送信息")


class Answer(BaseModel):
    message: str = Field(description="回答信息")
