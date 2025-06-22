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
        Flexible visual validation that accepts images if ANY criteria match
        Checks for brand word, shape, color, or any hint of similarity
        """
        try:
            # Step 1: Search for product information and images online
            search_result = self.search_product_online(sku)
            clean_sku = sku.replace('_', ' ').replace('-', ' ').replace('__', ' ')
            
            # Step 2: Create flexible validation prompt with multiple matching criteria
            enhanced_prompt = f"""
FLEXIBLE VISUAL VALIDATION WITH MULTIPLE MATCHING CRITERIA:

You are performing a FLEXIBLE visual analysis to determine if the uploaded image matches the product SKU.

SKU: "{sku}" (clean: "{clean_sku}")
WEB SEARCH CONTEXT: {search_result.get('search_query', sku)}
USER'S IMAGE: [Image will be provided]

VALIDATION PROCESS:
1. Analyze the user's uploaded image for:
   - Product packaging and design
   - Color scheme and brand colors
   - Text/labeling on packaging (brand names, product names)
   - Product type and category
   - Visual characteristics and appearance

2. Compare with what the SKU "{clean_sku}" should represent:
   - Expected product type (spice, food, ingredient, etc.)
   - Typical packaging style for this product
   - Common brand colors and design elements
   - Expected visual characteristics

3. Check for ANY of these matching criteria (ACCEPT if ANY match):
   - BRAND WORD MATCH: Does the image contain brand names or words that match the SKU?
   - SHAPE MATCH: Does the packaging shape/style match typical products in this category?
   - COLOR MATCH: Do the colors match typical brand/product colors for this SKU?
   - HINT MATCH: Are there any visual hints or elements that suggest this is the right product?
   - CATEGORY MATCH: Is it the same product category (food, spice, ingredient, etc.)?

ACCEPTANCE CRITERIA (FLEXIBLE - ANY MATCH IS ENOUGH):
- Brand word/name matches → ACCEPT
- Shape/packaging style matches → ACCEPT
- Color scheme matches → ACCEPT
- Any visual hint suggests it's the right product → ACCEPT
- Same product category with similar characteristics → ACCEPT
- Similar packaging design → ACCEPT
- Matching brand elements → ACCEPT

REJECTION CRITERIA (ONLY IF NO MATCHES):
- Completely wrong product category (electronics, furniture, clothing, etc.)
- No brand words, shape, color, or hint matches
- Non-product images (people, landscapes, etc.)
- Blank or corrupted images

EXAMPLES:
- SKU: "SHAN_MASALA" + Image: any Shan brand product → ACCEPT (brand match)
- SKU: "BAISAN" + Image: similar shaped packaging → ACCEPT (shape match)
- SKU: "NATIONAL_SPICE" + Image: similar color scheme → ACCEPT (color match)
- SKU: "SHAN_MASALA" + Image: any spice product → ACCEPT (category match)
- SKU: "BAISAN" + Image: electronics/phone → REJECT (no matches)

Return ONLY this JSON:
{{
  "match": true/false,
  "sku_type": "description of what SKU suggests",
  "image_type": "description of what image shows",
  "matching_criteria": ["brand_word", "shape", "color", "hint", "category"],
  "matched_elements": "which specific elements matched",
  "reason": "explanation of decision",
  "confidence": "high/medium/low"
}}

Be FLEXIBLE - accept if ANY criteria match (brand, shape, color, hint, category).
"""
            
            # Make the validation call
            validation_response = self._make_api_call(enhanced_prompt, image_bytes=image_bytes, mime_type=mime_type)
            
            try:
                clean_response = validation_response.strip().lstrip('```json').rstrip('```').strip()
                validation_data = json.loads(clean_response)
                validation_data['web_search_used'] = True
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
        Flexible validation as fallback when enhanced validation fails
        Accepts images if ANY criteria match (brand, shape, color, hint)
        """
        readable_sku = sku.replace('_', ' ').replace('__', ' ')
        
        simple_prompt = f"""
FLEXIBLE IMAGE VALIDATION:

You are performing a FLEXIBLE visual analysis to validate if the uploaded image matches the product SKU.

SKU: "{sku}" (suggests: {readable_sku})
Image: [Image will be provided]

VISUAL ANALYSIS PROCESS:
1. Examine the uploaded image for:
   - Product packaging and design
   - Color scheme and branding
   - Text/labeling on packaging (brand names, product names)
   - Product type and category
   - Visual characteristics and appearance

2. Compare with SKU expectations:
   - What product type should "{readable_sku}" represent?
   - What visual elements are expected?
   - What colors and packaging style are typical?

3. Check for ANY of these matching criteria (ACCEPT if ANY match):
   - BRAND WORD MATCH: Does the image contain brand names or words that match the SKU?
   - SHAPE MATCH: Does the packaging shape/style match typical products in this category?
   - COLOR MATCH: Do the colors match typical brand/product colors for this SKU?
   - HINT MATCH: Are there any visual hints or elements that suggest this is the right product?
   - CATEGORY MATCH: Is it the same product category (food, spice, ingredient, etc.)?

ACCEPTANCE CRITERIA (FLEXIBLE - ANY MATCH IS ENOUGH):
- Brand word/name matches → ACCEPT
- Shape/packaging style matches → ACCEPT
- Color scheme matches → ACCEPT
- Any visual hint suggests it's the right product → ACCEPT
- Same product category with similar characteristics → ACCEPT
- Similar packaging design → ACCEPT
- Matching brand elements → ACCEPT

REJECTION CRITERIA (ONLY IF NO MATCHES):
- Completely wrong product category (electronics, furniture, clothing, etc.)
- No brand words, shape, color, or hint matches
- Non-product images (people, landscapes, etc.)
- Blank or corrupted images

Return ONLY this JSON:
{{
  "match": true/false,
  "sku_type": "description of what SKU suggests",
  "image_type": "description of what image shows",
  "matching_criteria": ["brand_word", "shape", "color", "hint", "category"],
  "matched_elements": "which specific elements matched",
  "reason": "explanation of decision",
  "confidence": "high/medium/low"
}}

Be FLEXIBLE - accept if ANY criteria match (brand, shape, color, hint, category).
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
                'match': True,  # Default to accept if validation fails (flexible approach)
                'sku_type': 'Unknown',
                'image_type': 'Unknown',
                'matching_criteria': ['unknown'],
                'matched_elements': 'Validation parsing failed',
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