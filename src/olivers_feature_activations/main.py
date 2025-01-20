import pandas as pd
import goodfire
import time
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv
import os
import logging
from typing import List, Dict, Any
import pickle

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/feature_extraction_{time.strftime("%Y%m%d-%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

RATE_LIMIT = 400  # Adjust based on your API limits
BATCH_SIZE = 50   # Adjust based on your needs

class AsyncRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.semaphore = asyncio.Semaphore(requests_per_minute)
        self.request_times: List[float] = []
        self.window_size = 60
        self.requests_per_minute = requests_per_minute
        self.lock = asyncio.Lock()
        logger.info(f"Initialized rate limiter with {requests_per_minute} rpm")

    async def acquire(self):
        while True:
            current_time = time.time()
            should_wait = False
            wait_time = 0

            async with self.lock:
                self.request_times = [t for t in self.request_times 
                                    if current_time - t < self.window_size]

                if len(self.request_times) >= self.requests_per_minute:
                    wait_time = self.request_times[0] + self.window_size - current_time
                    if wait_time > 0:
                        should_wait = True
                else:
                    self.request_times.append(current_time)
                    await self.semaphore.acquire()
                    return

            if should_wait:
                logger.info(f"Rate limit reached, waiting for {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                continue

    async def release(self):
        self.semaphore.release()

async def process_entity(
    client: goodfire.AsyncClient,
    variant: goodfire.Variant,
    record: Dict,
    key_name: str,
    k: int,
    rate_limiter: AsyncRateLimiter
) -> Dict:
    """Process a single entity with rate limiting"""
    try:
        item = record.get(key_name)
        if not item:
            logger.warning(f"Skipping item: {record}")
            return None

        await rate_limiter.acquire()
        try:
            inspector = await client.features.inspect(
                [{"role": "user", "content": item}],
                model=variant,
            )
            features = []

            for activation in inspector.top(k=k):
                features.append({
                    "uuid": activation.feature.uuid,
                    "activation": activation.activation,
                })

            return {
                "query": record.get("query"),
                "entity": record.get("entity"),
                "known_unknown": record.get("known_unknown"),
                "red_herring": record.get("red_herring"),
                "entity_features": features,
            }
        finally:
            await rate_limiter.release()

    except Exception as e:
        logger.error(f"Failed to process entity: {str(e)}")
        return None

async def get_feature_activations_v2(
    client: goodfire.AsyncClient,
    variant: goodfire.Variant,
    entities: pd.DataFrame,
    known_unknown_filter: str,
    key_name: str = "query",
    k: int = 100,
    batch_size: int = BATCH_SIZE
):
    """Process entities in batches with rate limiting"""
    rate_limiter = AsyncRateLimiter(RATE_LIMIT)
    feature_activations = []
    feature_library = set()

    # Filter entities
    filtered_entities = [
        entity for entity in entities.to_dict('records')
        if entity.get("known_unknown") == known_unknown_filter
    ]
    
    logger.info(f"Processing {len(filtered_entities)} entities")

    # Process in batches
    for i in tqdm(range(0, len(filtered_entities), batch_size)):
        batch = filtered_entities[i:i + batch_size]
        
        logger.info(f"Processing batch of {len(batch)} entities")
        tasks = [
            process_entity(client, variant, record, key_name, k, rate_limiter)
            for record in batch
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {str(result)}")
                continue
            if result:
                feature_activations.append(result)
                # Update feature library
                for feature in result["entity_features"]:
                    feature_library.add((feature["uuid"], None))  # Adjust if you need labels

    logger.info(f"Processed {len(feature_activations)} entities successfully")
    return feature_activations, feature_library

async def main():
    # Load data
    queries_for_feature_extraction = pd.read_parquet(
        "./queries_for_feature_extraction.parquet"
    )

    client_gf = goodfire.AsyncClient(os.getenv("GOODFIRE_API_KEY"))
    variant = goodfire.Variant("meta-llama/Meta-Llama-3.1-8B-Instruct")
    filter = "unknown"

    feature_activations_known_1Q, feature_library_known_1Q = await get_feature_activations_v2(
        client_gf,
        variant,
        queries_for_feature_extraction,
        known_unknown_filter=filter,
        key_name="query",
        k=100,
    )

    # Pickle results
    with open(f"feature_activations_{filter}_1Q.pkl", "wb") as f:
        pickle.dump(feature_activations_known_1Q, f)

    with open(f"feature_library_{filter}_1Q.pkl", "wb") as f:
        pickle.dump(feature_library_known_1Q, f)

if __name__ == "__main__":
    asyncio.run(main())