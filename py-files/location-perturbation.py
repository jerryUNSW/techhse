import torch
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from openai import OpenAI
from dotenv import load_dotenv
import os

deep_seek_ky = os.getenv("DEEP_SEEK_KEY")


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def get_embedding(phrase):
    """Generate BERT embedding for a given phrase."""
    inputs = tokenizer(phrase, return_tensors="pt", 
        padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)
    return (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

# Precompute embeddings for candidate phrases
def precompute_candidate_embeddings(candidate_phrases):
    """Precompute embeddings for all candidate phrases."""
    # return {phrase: get_embedding(phrase).numpy() for phrase in candidate_phrases}
    return {phrase: get_embedding(phrase).cpu().numpy() for phrase in candidate_phrases}


def differentially_private_replacement(target_phrase, epsilon, candidate_phrases, candidate_embeddings):
    """
    Selects a differentially private replacement for a given target phrase.

    Parameters:
    - target_phrase (str): The phrase to be replaced.
    - epsilon (float): Privacy budget (permissible privacy loss)
    - candidate_phrases (list of str): List of candidate replacement phrases.
    - candidate_embeddings (dict): Precomputed embeddings for candidate phrases.

    Returns:
    - str: A differentially private replacement phrase.
    """
    # Compute embedding only for the target phrase
    target_embedding = get_embedding(target_phrase).cpu().numpy()

    # Stack precomputed candidate embeddings
    candidate_embeddings_matrix = np.vstack([candidate_embeddings[phrase] for phrase in candidate_phrases])

    # Compute cosine similarity
    similarities = cosine_similarity(target_embedding, candidate_embeddings_matrix)[0]
    
    # Convert similarity to distance
    distances = 1 - similarities
    
    # Apply the exponential mechanism
    p_unnorm = np.exp(-epsilon * distances)
    p_norm = p_unnorm / np.sum(p_unnorm)  # Normalize to make it a probability distribution

    # Sample a replacement
    return np.random.choice(candidate_phrases, p=p_norm)


# input_prompt = "I am a machine learning engineer in Tokyo."

input_prompt = "Sam lives in downtown Boston"

nlp = spacy.load("en_core_web_sm")

print("NER detecting sensitive phrases")

# Process the text
doc = nlp(input_prompt)

# Extract locations and occupations
extracted_locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]

print("Detected Locations:", extracted_locations)

# Extract occupations:
# File path (replace with your actual file path)
file_path = 'occupation.json'

# Open and read the file
with open(file_path, 'r') as file:
    # Parse the JSON data from the file
    data = json.load(file)

# Extract the occupation labels into a list
occupation_list = [entry['occupationLabel'] for entry in data]

# Print the list of occupations
occupation_list = [occupation for occupation in occupation_list if not occupation.startswith('Q') or not occupation[1:].isdigit()]

# for item in occupation_list:
#     print(item)



# Example candidate jobs
candidate_jobs = [
    "Software Engineer", "Data Scientist", "Product Manager", "Teacher", "Nurse",
    "Doctor", "Lawyer", "Architect", "Civil Engineer", "Electrical Engineer",
    "Marketing Manager", "Graphic Designer", "Accountant", "Financial Analyst", "Sales Manager",
    "Human Resources Manager", "Construction Worker", "Chef", "Pharmacist", "Web Developer",
    "Dentist", "Veterinarian", "Research Scientist", "Electrician", "Plumber",
    "Teacher Assistant", "Retail Manager", "Police Officer", "Firefighter", "Journalist",
    "Editor", "Writer", "Photographer", "Artist", "Musician", "Actor", "Law Enforcement Officer",
    "Social Worker", "Real Estate Agent", "Psychologist", "Physician Assistant", "Biologist",
    "Nuclear Engineer", "Environmental Scientist", "Pilot", "Flight Attendant", "Air Traffic Controller",
    "Surveyor", "Urban Planner", "Computer Programmer", "Network Administrator", "Database Administrator",
    "Occupational Therapist", "Speech Therapist", "Physical Therapist", "Yoga Instructor", "Fitness Trainer",
    "Event Planner", "Travel Agent", "Insurance Agent", "HR Specialist", "SEO Specialist",
    "UX/UI Designer", "Marketing Analyst", "Business Analyst", "Entrepreneur", "Financial Advisor",
    "Investment Banker", "Risk Manager", "Operations Manager", "Retail Associate", "Customer Service Representative",
    "Stock Broker", "Legal Advisor", "Public Relations Specialist", "Catering Manager", "Butcher",
    "Florist", "Housekeeper", "Construction Manager", "Security Guard", "Cleaner", "Janitor", "Massage Therapist",
    "Speech Pathologist", "Chiropractor", "Film Director", "Animator", "Fashion Designer", "Interior Designer",
    "Personal Assistant", "Taxi Driver", "Bus Driver", "Truck Driver", "Librarian", "Bookkeeper", "Coach",
    "Athletic Trainer", "Barista", "Bartender", "Waiter", "Waitress", "Cook", "Baker", "Mechanic",
    "Auto Body Technician", "Painter", "Sculptor", "Tattoo Artist", "Hair Stylist", "Nail Technician",
    "Cosmetologist", "Jeweler", "Carpenter", "Handyman", "Tour Guide", "Language Translator", "Interpreter"
]

# Example locations:
candidate_addresses = [
    "UNSW Sydney High St Kensington, NSW 2052", "320-346 Barker Street, Randwick, NSW 2031", 
    "1600 Amphitheatre Parkway, Mountain View, CA 94043, USA", "221B Baker Street, London NW1 6XE, United Kingdom", 
    "4 Privet Drive, Little Whinging, Surrey, England", "1 Infinite Loop, Cupertino, CA 95014, USA", 
    "350 Fifth Avenue, New York, NY 10118, USA", "1600 Pennsylvania Avenue NW, Washington, DC 20500, USA", 
    "10 Downing Street, London SW1A 2AA, United Kingdom", "Eiffel Tower, Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France", 
    "Colosseum, Piazza del Colosseo, 1, 00184 Roma RM, Italy", "Great Wall of China, Huairou, China", 
    "Sydney Opera House, Bennelong Point, Sydney NSW 2000, Australia", "Taj Mahal, Dharmapuri, Forest Colony, Tajganj, Agra, Uttar Pradesh 282001, India", 
    "Christ the Redeemer, Parque Nacional da Tijuca, Rio de Janeiro - RJ, Brazil", "Burj Khalifa, 1 Sheikh Mohammed bin Rashid Blvd, Dubai, United Arab Emirates", 
    "Red Square, Moscow, Russia, 109012", "Pyramids of Giza, Al Haram, Giza Governorate, Egypt", 
    "Statue of Liberty, New York, NY 10004, USA", "Machu Picchu, 08680, Peru", "Santorini, Thira 847 00, Greece", 
    "Chichen Itza, Yucatan, Mexico", "Stonehenge, Salisbury SP4 7DE, United Kingdom", 
    "Louvre Museum, Rue de Rivoli, 75001 Paris, France", "Buckingham Palace, London SW1A 1AA, United Kingdom", 
    "Times Square, Manhattan, NY 10036, USA", "Golden Gate Bridge, San Francisco, CA, USA", "Niagara Falls, NY 14303, USA", 
    "Mount Fuji, Kitayama, Fujinomiya, Shizuoka 418-0112, Japan", "Acropolis of Athens, Athens 105 58, Greece", 
    "Kremlin, Moscow, Russia, 103073", "Petra, Wadi Musa, Jordan", "Angkor Wat, Krong Siem Reap, Cambodia", 
    "Sagrada Familia, Carrer de Mallorca, 401, 08013 Barcelona, Spain", "Grand Canyon, Arizona 86023, USA", 
    "Great Barrier Reef, Queensland, Australia", "Banff National Park, Alberta, Canada", "Galapagos Islands, Ecuador", 
    "Serengeti National Park, Tanzania", "Bora Bora, French Polynesia", "Maldives", "Mount Kilimanjaro, Tanzania", 
    "Antarctica", "Victoria Falls, Zimbabwe", "Iguazu Falls, Misiones Province, Argentina", 
    "Blue Lagoon, Grindavík, Iceland", "Cinque Terre, 19016 Vernazza SP, Italy", "Plitvice Lakes National Park, Croatia", 
    "Bagan, Myanmar (Burma)", "Salar de Uyuni, Bolivia", "Ha Long Bay, Quảng Ninh, Vietnam", "Pamukkale, Denizli, Turkey", 
    "Bali, Indonesia", "Kyoto, Japan", "Prague Castle, 119 08 Prague 1, Czechia", 
    "Neuschwanstein Castle, Neuschwansteinstraße 20, 87645 Schwangau, Germany", 
    "Alhambra, Calle Real de la Alhambra, s/n, 18009 Granada, Spain", "Edinburgh Castle, Castlehill, Edinburgh EH1 2NG, United Kingdom", 
    "Dubrovnik Old Town, 20000 Dubrovnik, Croatia", "Marrakech, Morocco", "Cappadocia, Turkey", 
    "Petronas Twin Towers, Kuala Lumpur City Centre, 50088 Kuala Lumpur, Malaysia", 
    "Table Mountain, Cape Town, South Africa", "Uluru, Petermann NT 0872, Australia", "Lake Louise, Alberta, Canada", 
    "Zermatt, Switzerland", "Hallstatt, Austria", "Reykjavik, Iceland", "Cinque Terre, Italy", "Santorini, Greece", 
    "Queenstown, New Zealand", "Banff, Alberta, Canada", "Kyoto, Japan", "Siem Reap, Cambodia", "Cusco, Peru", 
    "Lisbon, Portugal", "Seoul, South Korea", "Havana, Cuba", "Buenos Aires, Argentina", "Rio de Janeiro, Brazil", 
    "Lima, Peru", "Santiago, Chile", "Bogotá, Colombia", "Quito, Ecuador", "La Paz, Bolivia", "Caracas, Venezuela", 
    "Panama City, Panama", "San José, Costa Rica", "Managua, Nicaragua", "Tegucigalpa, Honduras", 
    "San Salvador, El Salvador", "Guatemala City, Guatemala", "Belize City, Belize", "Mexico City, Mexico", 
    "Kingston, Jamaica", "Nassau, Bahamas", "Port-au-Prince, Haiti", "Santo Domingo, Dominican Republic", 
    "San Juan, Puerto Rico", "Bridgetown, Barbados", "Castries, Saint Lucia", "Port of Spain, Trinidad and Tobago"
]


# candidate_locations = [
#     "UNSW Sydney High St Kensington, NSW 2052", 
#         "320-346 Barker Street, Randwick, NSW 2031", 
#     "1600 Amphitheatre Parkway, Mountain View, CA 94043, USA",
#     "221B Baker Street, London NW1 6XE, United Kingdom",
#     "4 Privet Drive, Little Whinging, Surrey, England",
#     "1 Infinite Loop, Cupertino, CA 95014, USA",
#     "350 Fifth Avenue, New York, NY 10118, USA",
#     "1600 Pennsylvania Avenue NW, Washington, DC 20500, USA",
#     "New York City",
#     "10 Downing Street, London SW1A 2AA, United Kingdom",
#     "Eiffel Tower, Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
#     "Colosseum, Piazza del Colosseo, 1, 00184 Roma RM, Italy",
#     "Great Wall of China, Huairou, China",
#     "Sydney Opera House, Bennelong Point, Sydney NSW 2000, Australia",
#     "Taj Mahal, Dharmapuri, Forest Colony, Tajganj, Agra, Uttar Pradesh 282001, India",
#     "Christ the Redeemer, Parque Nacional da Tijuca, Rio de Janeiro - RJ, Brazil",
#     "Burj Khalifa, 1 Sheikh Mohammed bin Rashid Blvd, Dubai, United Arab Emirates",
#     "Red Square, Moscow, Russia, 109012",
#     "Pyramids of Giza, Al Haram, Giza Governorate, Egypt",
#     "Statue of Liberty, New York, NY 10004, USA",
#     "Machu Picchu, 08680, Peru",
#     "Santorini, Thira 847 00, Greece",
#     "Chichen Itza, Yucatan, Mexico",
#     "Stonehenge, Salisbury SP4 7DE, United Kingdom",
#     "Louvre Museum, Rue de Rivoli, 75001 Paris, France",
#     "Buckingham Palace, London SW1A 1AA, United Kingdom",
#     "Times Square, Manhattan, NY 10036, USA",
#     "Golden Gate Bridge, San Francisco, CA, USA",
#     "Niagara Falls, NY 14303, USA",
#     "Mount Fuji, Kitayama, Fujinomiya, Shizuoka 418-0112, Japan",
#     "Acropolis of Athens, Athens 105 58, Greece",
#     "Kremlin, Moscow, Russia, 103073",
#     "Petra, Wadi Musa, Jordan",
#     "Angkor Wat, Krong Siem Reap, Cambodia",
#     "Sagrada Familia, Carrer de Mallorca, 401, 08013 Barcelona, Spain",
#     "Grand Canyon, Arizona 86023, USA",
#     "Great Barrier Reef, Queensland, Australia",
#     "Banff National Park, Alberta, Canada",
#     "Galapagos Islands, Ecuador",
#     "Serengeti National Park, Tanzania",
#     "Bora Bora, French Polynesia",
#     "Maldives",
#     "Mount Kilimanjaro, Tanzania",
#     "Antarctica",
#     "Victoria Falls, Zimbabwe",
#     "Iguazu Falls, Misiones Province, Argentina",
#     "Blue Lagoon, Grindavík, Iceland",
#     "Cinque Terre, 19016 Vernazza SP, Italy",
#     "Plitvice Lakes National Park, Croatia",
#     "Bagan, Myanmar (Burma)",
#     "Salar de Uyuni, Bolivia",
#     "Ha Long Bay, Quảng Ninh, Vietnam",
#     "Pamukkale, Denizli, Turkey",
#     "Bali, Indonesia",
#     "Kyoto, Japan",
#     "Prague Castle, 119 08 Prague 1, Czechia",
#     "Neuschwanstein Castle, Neuschwansteinstraße 20, 87645 Schwangau, Germany",
#     "Alhambra, Calle Real de la Alhambra, s/n, 18009 Granada, Spain",
#     "Edinburgh Castle, Castlehill, Edinburgh EH1 2NG, United Kingdom",
#     "Dubrovnik Old Town, 20000 Dubrovnik, Croatia",
#     "Marrakech, Morocco",
#     "Cappadocia, Turkey",
#     "Petronas Twin Towers, Kuala Lumpur City Centre, 50088 Kuala Lumpur, Malaysia",
#     "Table Mountain, Cape Town, South Africa",
#     "Uluru, Petermann NT 0872, Australia",
#     "Lake Louise, Alberta, Canada",
#     "Zermatt, Switzerland",
#     "Hallstatt, Austria",
#     "Reykjavik, Iceland",
#     "Cinque Terre, Italy",
#     "Santorini, Greece",
#     "Queenstown, New Zealand",
#     "Banff, Alberta, Canada",
#     "Kyoto, Japan",
#     "Siem Reap, Cambodia",
#     "Cusco, Peru",
#     "Lisbon, Portugal",
#     "Seoul, South Korea",
#     "Havana, Cuba",
#     "Buenos Aires, Argentina",
#     "Rio de Janeiro, Brazil",
#     "Lima, Peru",
#     "Santiago, Chile",
#     "Bogotá, Colombia",
#     "Quito, Ecuador",
#     "La Paz, Bolivia",
#     "Caracas, Venezuela",
#     "Panama City, Panama",
#     "San José, Costa Rica",
#     "Managua, Nicaragua",
#     "Tegucigalpa, Honduras",
#     "San Salvador, El Salvador",
#     "Guatemala City, Guatemala",
#     "Belize City, Belize",
#     "Mexico City, Mexico",
#     "Kingston, Jamaica",
#     "Nassau, Bahamas",
#     "Port-au-Prince, Haiti",
#     "Santo Domingo, Dominican Republic",
#     "San Juan, Puerto Rico",
#     "Bridgetown, Barbados",
#     "Castries, Saint Lucia",
#     "Port of Spain, Trinidad and Tobago"
# ]

# Open the loc.txt file and read the contents into a list
candidate_locations = []

with open("loc.txt", "r") as file:
    # Read each line (location) from the file and append it to the list
    candidate_locations = [line.strip() for line in file.readlines()]

# Remove duplicates by converting the list to a set and then back to a list
candidate_locations = list(set(candidate_locations))
# Now candidate_locations is a list containing all the location entities
# print(candidate_locations[:10])  # Print the first 10 locations as a sample

# Precompute embeddings for candidate phrases
# jobs_embeddings = precompute_candidate_embeddings(candidate_jobs)

loc_embeddings = precompute_candidate_embeddings(candidate_locations)


# # Query target phrase
target_job = "Machine Learning Engineer"

# target_loc = "Tokyo"
target_loc = "Boston"




client = OpenAI(api_key=deep_seek_ky, base_url="https://api.deepseek.com")


# Apply the exponential mechanism
for epsilon in range(1, 100, 10):
    # job_replacement = differentially_private_replacement(target_job, 
    #     epsilon=epsilon, 
    #     candidate_phrases=candidate_jobs, 
    #     candidate_embeddings=jobs_embeddings)

    loc_replacement = differentially_private_replacement(target_loc, 
        epsilon=epsilon, 
        candidate_phrases=    candidate_locations, 
        candidate_embeddings= loc_embeddings)

    # sanitized_prompt = f"I am a {job_replacement} in {loc_replacement}."

    sanitized_prompt = f"Sam lives in downtown {loc_replacement}."

    prompt_ = f"Text completion task. Sentence: {sanitized_prompt} Do the task and make sure the added text does not heavily depend on \"{loc_replacement}\". Keep it concise."

    print("Epsilon = ", epsilon, prompt_)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt_},
        ],
        stream=False
    )

    response_content = response.choices[0].message.content


    print("Response = ", response_content)