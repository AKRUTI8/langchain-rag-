"""
Essential Prompt Engineering Techniques for LLMs
Demonstrates 7 core prompting strategies with practical examples
"""

class PromptTechniques:
    """Collection of 7 essential prompt engineering techniques"""
    
    def __init__(self, model_name="claude-sonnet-4-20250514"):
        self.model_name = model_name
    
    # 1. INSTRUCTION PROMPT
    def instruction_prompt(self):
        """Clear, direct instructions for the task"""
        prompt = """Analyze the following customer review and provide:
1. Overall sentiment (positive/negative/neutral)
2. Key themes mentioned
3. Specific pain points or praise
4. Actionable recommendations

Review: "The product arrived late and the packaging was damaged, 
but once I set it up, it worked great. Customer service was helpful."
"""
        return prompt
    
    # 2. ROLE-BASED PROMPT
    def role_based_prompt(self):
        """Assign a specific role or expertise to the model"""
        prompt = """You are an experienced cybersecurity expert with 15 years in the field.
A small business owner asks:

"I run an online store with 10 employees. What are the top 3 security measures 
I should implement immediately to protect customer data?"

Provide practical, actionable advice based on your expertise."""
        return prompt
    
    # 3. FEW-SHOT PROMPT
    def few_shot_prompt(self):
        """Provide examples to guide the model's responses"""
        prompt = """Classify movie reviews as positive, negative, or neutral.

Example 1:
Review: "This film was absolutely brilliant! The acting was superb and the plot kept me engaged."
Classification: Positive

Example 2:
Review: "Terrible waste of time. Poor acting and confusing storyline."
Classification: Negative

Example 3:
Review: "The movie was okay. Some good scenes but nothing memorable."
Classification: Neutral

Now classify this review:
Review: "An incredible masterpiece! Every scene was perfectly crafted and the ending left me in tears."
Classification:"""
        return prompt
    
    # 4. ZERO-SHOT PROMPT
    def zero_shot_prompt(self):
        """Direct task without any examples"""
        prompt = """Translate the following English text to French:

English: "The weather is beautiful today. Would you like to go for a walk in the park?"
French:"""
        return prompt
    
    # 5. CHAIN-OF-THOUGHT PROMPT
    def chain_of_thought_prompt(self):
        """Encourage step-by-step reasoning"""
        prompt = """Solve this problem step by step:

A bakery sells cupcakes for $3 each. If you buy a dozen, you get a 15% discount. 
If you buy 2 dozen, you get a 25% discount. How much would you save by buying 
2 dozen instead of buying them individually?

Let's solve this step by step:
1) Calculate the cost of buying 24 cupcakes individually
2) Calculate the cost with the 2 dozen discount
3) Find the difference (savings)
4) State the final answer"""
        return prompt
    
    # 6. CONVERSATIONAL PROMPT
    def conversational_prompt(self):
        """Natural back-and-forth dialogue style"""
        prompt = """User: Hi! I'm planning a trip to Japan and I'm a bit overwhelmed. 
Where should I start?