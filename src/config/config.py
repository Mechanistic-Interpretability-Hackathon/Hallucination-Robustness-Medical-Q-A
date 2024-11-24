import os
from dotenv import load_dotenv
import goodfire

load_dotenv()
gf_api_key = os.getenv('GOODFIRE_API_KEY')
client = goodfire.Client(gf_api_key)
variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")