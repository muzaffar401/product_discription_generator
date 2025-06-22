import os
import time
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
import openai
import re
from PIL import Image
import io
import mimetypes
import json
import base64
import requests
from urllib.parse import quote_plus
import urllib.parse

# Load environment variables
load_dotenv()

# Configure API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GEMINI_API_KEY and not OPENAI_API_KEY:
    raise ValueError("Either GEMINI_API_KEY or OPENAI_API_KEY must be set in environment variables")

# Configure Gemini API if key is available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Configure OpenAI API if key is available
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

class ProductDescriptionGenerator:
    def __init__(self, use_openai=False):
        self.use_openai = use_openai
        if self.use_openai:
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not self.client.api_key:
                raise ValueError("OPENAI_API_KEY not found or is invalid.")
        else:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables.")
            genai.configure(api_key=self.gemini_api_key)

    def _make_api_call(self, prompt, image_bytes=None, mime_type=None, retries=3, delay=30):
        if self.use_openai:
            for attempt in range(retries):
                try:
                    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                    if image_bytes:
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        messages[0]["content"].append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                        })
                    
                    response = self.client.chat.completions.create(
                        model="gpt-4o", 
                        messages=messages, 
                        max_tokens=400,
                        timeout=60  # Add timeout
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"OpenAI API call failed on attempt {attempt + 1}: {e}")
                    if attempt < retries - 1:
                        time.sleep(delay * (attempt + 1))
                    else:
                        return "API_CALL_FAILED"
            return "API_CALL_FAILED"
        else:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            for attempt in range(retries):
                try:
                    content = [prompt]
                    if image_bytes:
                        image_parts = [{"mime_type": mime_type, "data": image_bytes}]
                        content.append(image_parts[0])
                    
                    # Add timeout and safety settings
                    response = model.generate_content(
                        content,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=400,
                            temperature=0.7
                        ),
                        safety_settings=[
                            {
                                "category": "HARM_CATEGORY_HARASSMENT",
                                "threshold": "BLOCK_NONE"
                            },
                            {
                                "category": "HARM_CATEGORY_HATE_SPEECH",
                                "threshold": "BLOCK_NONE"
                            },
                            {
                                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                "threshold": "BLOCK_NONE"
                            },
                            {
                                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                "threshold": "BLOCK_NONE"
                            }
                        ]
                    )
                    
                    if response.text:
                        return response.text
                    else:
                        return "API_CALL_FAILED"
                        
                except Exception as e:
                    print(f"Gemini API call failed on attempt {attempt + 1}: {e}")
                    if attempt < retries - 1:
                        time.sleep(delay * (attempt + 1))
                    else:
                        return "API_CALL_FAILED"
            return "API_CALL_FAILED"

    def search_product_online(self, sku):
        """
        Search for product information online using the SKU
        Returns a list of search results and product information
        """
        try:
            # Clean the SKU for search
            search_query = sku.replace('_', ' ').replace('-', ' ')
            
            # Use a simple web search approach (you can enhance this with proper search APIs)
            search_url = f"https://www.google.com/search?q={quote_plus(search_query + ' product')}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Extract basic information from search results
                # This is a simplified approach - in production, you might want to use proper search APIs
                search_info = {
                    'search_query': search_query,
                    'search_url': search_url,
                    'status': 'success',
                    'found_results': True
                }
                return search_info
            else:
                return {
                    'search_query': search_query,
                    'status': 'failed',
                    'error': f'HTTP {response.status_code}',
                    'found_results': False
                }
                
        except Exception as e:
            return {
                'search_query': sku,
                'status': 'error',
                'error': str(e),
                'found_results': False
            }

    def enhanced_image_validation(self, sku, image_bytes, mime_type):
        """
        Highly detailed, step-by-step validation protocol to ensure accuracy.
        Forces the model to deconstruct the SKU and image before comparing.
        """
        try:
            clean_sku = sku.replace('_', ' ').replace('-', ' ').replace('__', ' ')
            
            # Create a highly detailed, step-by-step validation prompt
            enhanced_prompt = f"""
HIGHLY DETAILED IMAGE-SKU VALIDATION PROTOCOL:

You are an expert image analyst. Your task is to perform a detailed, step-by-step analysis to determine if the provided image matches the SKU. Follow this protocol precisely.

**SKU for analysis:** "{sku}"
**Cleaned SKU words for analysis:** "{clean_sku}"

---
**STEP 1: Deconstruct the SKU**
Break down the cleaned SKU "{clean_sku}" into a list of individual keywords.

**STEP 2: Analyze Text Content from the Image**
Carefully read ALL text visible on the product packaging in the image. List all significant words you can identify.

**STEP 3: Analyze Visual Content of the Image**
Describe the object in the image. What is it? What is its category?

**STEP 4: Compare and Decide (The Matching Logic)**
Based *only* on your analysis from Steps 1, 2, and 3, answer the following.

*   **Brand Match Check:** Do any brand-related keywords from the SKU (Step 1) appear in the text from the image (Step 2)? (e.g., 'shan', 'national')
*   **Product Name Match Check:** Do any product name keywords from the SKU (Step 1) appear in the text from the image (Step 2)? (e.g., 'karahi', 'baisan')
*   **Category Match Check:** Is the visual category from Step 3 consistent with the SKU from Step 1?

**FINAL CONCLUSION:**
If you answered YES to AT LEAST ONE of the checks in Step 4, the final result is a MATCH. Otherwise, it is a MISMATCH.

---
**STEP 5: Generate Final JSON Output**
Now, provide your final answer in a single, raw JSON object. Do not add any text before or after the JSON. The JSON must contain your step-by-step analysis.

Return ONLY this JSON:
{{
  "match": true/false,
  "analysis": {{
    "sku_keywords": ["list", "of", "keywords", "from", "step1"],
    "image_text": ["list", "of", "words", "from", "step2"],
    "image_category": "description from step3"
  }},
  "decision_logic": {{
    "brand_match": "YES/NO. Reason...",
    "name_match": "YES/NO. Reason...",
    "category_match": "YES/NO. Reason..."
  }},
  "reason": "Final summary of your decision based on the matching logic."
}}

Example for SKU 'SHAN_KARAHI_MIX':
{{
  "match": true,
  "analysis": {{
    "sku_keywords": ["shan", "karahi", "mix"],
    "image_text": ["shan", "recipe", "masala", "mix", "karahi"],
    "image_category": "A box of Shan Karahi recipe and masala mix, a food product."
  }},
  "decision_logic": {{
    "brand_match": "YES. The SKU keyword 'shan' is found in the image text.",
    "name_match": "YES. The SKU keyword 'karahi' is found in the image text.",
    "category_match": "YES. The SKU suggests a recipe mix, and the image shows a recipe mix."
  }},
  "reason": "The image is a MATCH because the brand, name, and category all align with the SKU."
}}
"""
            
            # Make the validation call
            validation_response = self._make_api_call(enhanced_prompt, image_bytes=image_bytes, mime_type=mime_type)
            
            try:
                clean_response = validation_response.strip().lstrip('```json').rstrip('```').strip()
                validation_data = json.loads(clean_response)
                validation_data['web_search_used'] = False
                return validation_data
            except json.JSONDecodeError:
                # Fallback to simple validation if JSON parsing fails
                return self.simple_image_validation(sku, image_bytes, mime_type)
                
        except Exception as e:
            print(f"Enhanced validation failed: {str(e)}")
            # Fallback to simple validation
            return self.simple_image_validation(sku, image_bytes, mime_type)

    def simple_image_validation(self, sku, image_bytes, mime_type):
        """
        Highly detailed, step-by-step fallback validation protocol.
        """
        readable_sku = sku.replace('_', ' ').replace('__', ' ')
        
        simple_prompt = f"""
HIGHLY DETAILED IMAGE-SKU VALIDATION PROTOCOL:

You are an expert image analyst. Your task is to perform a detailed, step-by-step analysis to determine if the provided image matches the SKU. Follow this protocol precisely.

**SKU for analysis:** "{sku}"
**Cleaned SKU words for analysis:** "{readable_sku}"

---
**STEP 1: Deconstruct the SKU**
Break down the cleaned SKU "{readable_sku}" into a list of individual keywords.

**STEP 2: Analyze Text Content from the Image**
Carefully read ALL text visible on the product packaging in the image. List all significant words you can identify.

**STEP 3: Analyze Visual Content of the Image**
Describe the object in the image. What is it? What is its category?

**STEP 4: Compare and Decide (The Matching Logic)**
Based *only* on your analysis from Steps 1, 2, and 3, answer the following.

*   **Brand Match Check:** Do any brand-related keywords from the SKU (Step 1) appear in the text from the image (Step 2)? (e.g., 'shan', 'national')
*   **Product Name Match Check:** Do any product name keywords from the SKU (Step 1) appear in the text from the image (Step 2)? (e.g., 'karahi', 'baisan')
*   **Category Match Check:** Is the visual category from Step 3 consistent with the SKU from Step 1?

**FINAL CONCLUSION:**
If you answered YES to AT LEAST ONE of the checks in Step 4, the final result is a MATCH. Otherwise, it is a MISMATCH.

---
**STEP 5: Generate Final JSON Output**
Now, provide your final answer in a single, raw JSON object. Do not add any text before or after the JSON. The JSON must contain your step-by-step analysis.

Return ONLY this JSON:
{{
  "match": true/false,
  "analysis": {{
    "sku_keywords": ["list", "of", "keywords", "from", "step1"],
    "image_text": ["list", "of", "words", "from", "step2"],
    "image_category": "description from step3"
  }},
  "decision_logic": {{
    "brand_match": "YES/NO. Reason...",
    "name_match": "YES/NO. Reason...",
    "category_match": "YES/NO. Reason..."
  }},
  "reason": "Final summary of your decision based on the matching logic."
}}
"""
        
        validation_response = self._make_api_call(simple_prompt, image_bytes=image_bytes, mime_type=mime_type)
        
        try:
            clean_response = validation_response.strip().lstrip('```json').rstrip('```').strip()
            validation_data = json.loads(clean_response)
            validation_data['web_search_used'] = False
            validation_data['confidence'] = 'medium'
            return validation_data
        except json.JSONDecodeError:
            return {
                'match': True,  # Default to accept if validation fails
                'analysis': {
                    'sku_keywords': [],
                    'image_text': [],
                    'image_category': 'Unknown'
                },
                'decision_logic': {
                    'brand_match': 'unknown',
                    'name_match': 'unknown',
                    'category_match': 'unknown'
                },
                'reason': 'Validation parsing failed, defaulting to accept',
                'web_search_used': False,
                'confidence': 'low'
            }

    def generate_product_description(self, sku):
        prompt = f"""Generate a compelling product description for a product with SKU: {sku}. 

The description should be marketing-friendly, EXACTLY 1000 characters (including spaces), and highlight key features and benefits.

IMPORTANT: Format the description in 2-3 paragraphs with EXACTLY 1000 characters:
- First paragraph: Introduce the product and its main benefits
- Second paragraph: Describe key features and specifications  
- Third paragraph (optional): Add any additional benefits, usage tips, or call-to-action

Make each paragraph engaging and informative. Use bullet points or numbered lists within paragraphs if needed for better readability.

CRITICAL: The final description must be EXACTLY 1000 characters - no more, no less."""
        return self._make_api_call(prompt)

    def find_related_products(self, current_sku_or_title, all_skus, num_related=3):
        skus_to_search = [s for s in all_skus if s and s != current_sku_or_title]
        if not skus_to_search:
            return []
        
        prompt = f"""You are a product recommendation engine. Based on the target product, find the {num_related} most similar products from the provided list of SKUs.

Target Product: "{current_sku_or_title}"

List of available SKUs:
{', '.join(skus_to_search)}

Return ONLY the SKUs of the most related products, separated by a pipe '|'. Do not include the target product in the result. If no products are related, return an empty string.
"""
        response = self._make_api_call(prompt)
        if response and response != "API_CALL_FAILED":
            potential_skus = [sku.strip() for sku in response.split('|')]
            return [sku for sku in potential_skus if sku in skus_to_search]
        return []

    def generate_product_description_with_image(self, sku, image_name, image_bytes, mime_type):
        prompt = """You are an expert product marketer. Analyze this product image to generate a product title and a compelling description.

Instructions:
1.  **Product Title**: Create a concise, SEO-friendly, and accurate title for the product in the image. If the image is unclear or you cannot confidently identify the product, return "Unknown Product".
2.  **Product Description**: Write a marketing-friendly description of EXACTLY 1000 characters (including spaces) in 2-3 paragraphs:
    - First paragraph: Introduce the product and its main benefits
    - Second paragraph: Describe key features and specifications
    - Third paragraph (optional): Add any additional benefits, usage tips, or call-to-action
    If the title is "Unknown Product", the description should be "Could not generate description from image.".

CRITICAL: The final description must be EXACTLY 1000 characters - no more, no less.

Return the result as a single raw JSON object with two keys: "title" and "description". Do not wrap it in markdown or any other text.
Example for a clear image: {"title": "Shan Achar Ghost Masala 50g", "description": "Introducing the authentic Shan Achar Ghost Masala, a premium spice blend that brings the traditional flavors of South Asian cuisine to your kitchen. This carefully crafted masala combines the finest quality spices to create a rich, aromatic seasoning that elevates any dish to new heights of flavor.\n\nOur signature blend features a perfect balance of coriander, cumin, turmeric, and other hand-selected spices that have been roasted and ground to perfection. Each 50g pack contains the ideal proportions of ingredients, ensuring consistent taste and quality in every use. The masala is free from artificial preservatives and additives, maintaining the natural goodness of pure spices.\n\nWhether you're preparing traditional curries, marinating meats, or adding depth to vegetarian dishes, this versatile masala is your go-to seasoning. Its robust flavor profile makes it perfect for both everyday cooking and special occasions. Experience the authentic taste that has made Shan a trusted name in households worldwide."}
Example for an unclear image: {"title": "Unknown Product", "description": "Could not generate description from image."}
"""
        if sku:
            prompt += f"\n\nUse the following SKU for context: '{sku}'."
        if image_name:
            prompt += f" The original image file name is '{image_name}'."
        
        response_text = self._make_api_call(prompt, image_bytes=image_bytes, mime_type=mime_type)
        
        try:
            clean_response = response_text.strip().lstrip('```json').rstrip('```').strip()
            data = json.loads(clean_response)
            return data
        except (json.JSONDecodeError, AttributeError, TypeError):
            if response_text == "API_CALL_FAILED":
                 return {"title": "API_CALL_FAILED", "description": "API_CALL_FAILED"}
            return {"title": "", "description": response_text}

def process_products(use_openai: bool = False):
    # Initialize the generator
    generator = ProductDescriptionGenerator(use_openai=use_openai)
    
    # Check if enriched_products.csv exists
    if os.path.exists('enriched_products.csv'):
        print("Found existing enriched_products.csv, continuing from last processed product...")
        df = pd.read_csv('enriched_products.csv')
    else:
        # Read the input Excel file
        try:
            df = pd.read_excel('sample_products.xlsx')
        except FileNotFoundError:
            print("sample_products.xlsx not found, trying sample_products.xls...")
            try:
                df = pd.read_excel('sample_products.xls')
            except FileNotFoundError:
                raise FileNotFoundError("No Excel file found. Please ensure sample_products.xlsx or sample_products.xls exists.")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['sku'])
        # Create new columns
        df['description'] = ''
        df['related_products'] = ''
    
    # Get list of all products for related products search
    all_products = df['sku'].tolist()
    
    # Find products that need processing
    unprocessed_products = df[
        (df['description'].isna()) | 
        (df['description'] == '') | 
        (df['description'] == 'Description generation failed.') |
        (df['related_products'].isna()) | 
        (df['related_products'] == '')
    ]
    
    if len(unprocessed_products) == 0:
        print("All products have been processed!")
        return
    
    print(f"\nFound {len(unprocessed_products)} products that need processing")
    
    # Process each unprocessed product
    for idx, row in unprocessed_products.iterrows():
        print(f"\nProcessing product {idx + 1} of {len(df)}: {row['sku']}")
        
        # Generate description if needed
        if pd.isna(row['description']) or row['description'] == '' or row['description'] == 'Description generation failed.':
            description = generator.generate_product_description(row['sku'])
            df.at[idx, 'description'] = description
            # Add delay between description and related products
            time.sleep(30)
        
        # Find related products if needed
        if pd.isna(row['related_products']) or row['related_products'] == '':
            related = generator.find_related_products(row['sku'], all_products)
            df.at[idx, 'related_products'] = '|'.join(related)
            # Add delay between products
            time.sleep(30)
        
        # Save progress after each product
        df.to_csv('enriched_products.csv', index=False)
        print(f"Progress saved for product {idx + 1}")
    
    print(f"\nResults saved to enriched_products.csv")

if __name__ == "__main__":
    # Check which API key is available and use that
    use_openai = bool(OPENAI_API_KEY) and not bool(GEMINI_API_KEY)
    process_products(use_openai=use_openai)