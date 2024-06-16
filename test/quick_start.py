import torch
from src.ir import Retriever

# Initialize the retriever
vdr_text2text = Retriever.from_pretrained("vsearch/vdr-nq")

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vdr_text2text = vdr_text2text.to(device)

# Define a query and a list of passages
query = "What are the benefits of drinking green tea?"
passages = [
    "Green tea is known for its antioxidant properties, which can help protect cells from damage caused by free radicals. It also contains catechins, which have been shown to have anti-inflammatory and anti-cancer effects. Drinking green tea regularly may help improve overall health and well-being.",
    "The history of coffee dates back to ancient times, with its origins in Ethiopia. Coffee is one of the most popular beverages in the world and is enjoyed by millions of people every day.",
    "Yoga is a mind-body practice that combines physical postures, breathing exercises, and meditation. It has been practiced for thousands of years and is known for its many health benefits, including stress reduction and improved flexibility.",
    "Eating a balanced diet that includes a variety of fruits, vegetables, whole grains, and lean proteins is essential for maintaining good health. It provides the body with the nutrients it needs to function properly and can help prevent chronic diseases."
]

# Embed the query and passages
q_emb = vdr_text2text.encoder_q.embed(query, topk=768)  # Shape: [1, V]
p_emb = vdr_text2text.encoder_p.embed(passages, topk=768)  # Shape: [4, V]

# Query-passage Relevance
scores = q_emb @ p_emb.t()
print(scores)

# Output: 
# tensor([[91.1257, 17.6930, 13.0358, 12.4576]], device='cuda:0')


vdr_cross_modal = Retriever.from_pretrained("vsearch/vdr-cross-modal") # Note: encoder_p for images, encoder_q for text.

image_file = './examples/images/mars.png'
texts = [
    "Four thousand Martian days after setting its wheels in Gale Crater on Aug. 5, 2012, NASA’s Curiosity rover remains busy conducting exciting science. The rover recently drilled its 39th sample then dropped the pulverized rock into its belly for detailed analysis.",
    "ChatGPT is a chatbot developed by OpenAI and launched on November 30, 2022. Based on a large language model, it enables users to refine and steer a conversation towards a desired length, format, style, level of detail, and language."
]
image_emb = vdr_cross_modal.encoder_p.embed(image_file, topk=768) # Shape: [1, V]
text_emb = vdr_cross_modal.encoder_q.embed(texts, topk=768)  # Shape: [2, V]

# Image-text Relevance
scores = image_emb @ text_emb.t()
print(scores)

# Output: 
# tensor([[0.3209, 0.0984]])