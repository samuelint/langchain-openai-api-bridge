from langchain_core.runnables.schema import StreamEvent, EventData


class ChunkStub:
    def __init__(self, content: str):
        self.content = content


def create_stream_event(content: str = "", name: str = "", event: str = ""):
    event_data = EventData(chunk=ChunkStub(content=content))
    return StreamEvent(event=event, name=name, data=event_data)


def create_on_chat_model_stream_event(content: str = "", name: str = ""):
    return create_stream_event(content=content, name=name, event="on_chat_model_stream")
