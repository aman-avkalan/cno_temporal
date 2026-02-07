# run_workflow.py
import asyncio
from temporalio.client import Client

async def main():
    client = await Client.connect("localhost:7233")

    result = await client.execute_workflow(
        "CNOTrainingWorkflow",
        50,
        id="cno-ldc-training",
        task_queue="cno-training-queue",
    )

    print("Workflow finished")
    print("Saved plot at:", result)

if __name__ == "__main__":
    asyncio.run(main())
