def split_prompt(prompts):
  system_prompt = ''
  user_prompts = [] 
  
  for prompt in prompts:
    split_index = prompt.find('[0,1,2,3].') + len('[0,1,2,3].')
    system_prompt = prompt[:split_index].strip()
    user_prompt = prompt[split_index:].strip()
    user_prompts.append(user_prompt)

  return system_prompt, user_prompts

mock_responses = {
    "What are the symptoms of diabetes?": "The common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, and blurry vision.",
    "How to treat high blood pressure?": "High blood pressure can be managed with lifestyle changes such as reducing salt intake, regular exercise, and medications like ACE inhibitors.",
    "What are common causes of fatigue?": "Common causes of fatigue include stress, poor sleep quality, anemia, and chronic medical conditions like hypothyroidism."
}

activated_features = ["Feature A", "Feature B", "Feature C"]