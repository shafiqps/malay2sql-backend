from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import openai

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the Malay to English translation model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('mesolitica/nanot5-small-malaysian-translation-v2')
model = T5ForConditionalGeneration.from_pretrained('mesolitica/nanot5-small-malaysian-translation-v2')
model = model.to(device)  # Move model to GPU if available
print(f"Model loaded. Number of parameters: {model.num_parameters():,}")

def translate_malay_to_english(malay_text):
    # Add the translation prefix and prepare input
    prefix = 'terjemah ke Inggeris: '
    input_text = f"{prefix}{malay_text}"
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    input_ids = input_ids.to(device)  # Move input to GPU if available
    
    # Generate translation
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model.generate(
            input_ids,
            max_length=1024,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            do_sample=True
        )
    
    # Remove special tokens and decode
    all_special_ids = [0, 1, 2]  # Special token IDs to remove
    outputs = [i for i in outputs[0] if i not in all_special_ids]
    translation = tokenizer.decode(outputs, spaces_between_special_tokens=False).strip()
    
    return translation

def generate_sql_query(english_query, schema):
    prompt = f"""
    Given the following database schema:
    {schema}

    Generate an SQL query for the following request:
    {english_query}

    SQL Query:
    """
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    return response.choices[0].text.strip()

def process_schema_file(schema_content):
    # Process the schema file content
    # This is a placeholder implementation
    return schema_content.decode("utf-8")

def process_malay_query(malay_query):
    # Translate Malay to English
    english_query = translate_malay_to_english(malay_query)
    
    # Generate SQL query using OpenAI
    # Note: In a real implementation, you would retrieve the user's schema here
    schema = "CREATE TABLE users (id INT, name VARCHAR(255), email VARCHAR(255));"
    sql_query = generate_sql_query(english_query, schema)
    
    return sql_query