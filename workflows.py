# workflows.py
from temporalio import workflow
from datetime import timedelta

@workflow.defn
class CNOTrainingWorkflow:
    @workflow.run
    async def run(self, epochs: int = 50):
        result = await workflow.execute_activity(
            "train_cno_model",
            epochs,
            start_to_close_timeout=timedelta(hours=6),
        )
        return result
