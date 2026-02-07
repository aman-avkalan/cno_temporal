# worker.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker

from workflows import CNOTrainingWorkflow
from activities import train_cno_model


async def main():
    client = await Client.connect("localhost:7233")

    # Executor for blocking (GPU / PyTorch) activities
    activity_executor = ThreadPoolExecutor(max_workers=1)

    worker = Worker(
        client,
        task_queue="cno-training-queue",
        workflows=[CNOTrainingWorkflow],
        activities=[train_cno_model],
        activity_executor=activity_executor,
    )

    print("Temporal worker started (GPU-safe)")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
