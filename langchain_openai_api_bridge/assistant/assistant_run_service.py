from openai.types.beta.threads import Run


class AssistantRunService:

    def __init__(
        self,
    ) -> None:
        pass

    def create(
        self, thread_id: str, assistant_id: str, model: str, stream: bool = False
    ) -> Run:
        # client.beta.threads.runs.create(
        #     thread_id="thread_123",
        #     assistant_id="asst_123",
        #     stream=True
        # )

        raise NotImplementedError
