import requests
import pandas as pd
from bs4 import BeautifulSoup
import openai
import time
import os
from dotenv import load_dotenv
from urllib.parse import urljoin

load_dotenv()

def fetch_companies_data(start_url):
    response = requests.get(start_url)

    soup = BeautifulSoup(response.content, 'html.parser')

    companies_data = []
    while True:
        companies = soup.find_all('div', {'class': 'views-row'})
        for company in companies:
            name = company.find('div', {'class': 'views-field views-field-field-clubsnsw-trading-name'}).text.strip()
            address_line1 = company.find('div', {'class': 'views-field-field-clubsnsw-address-line-1'}).text.strip()
            address_line2 = company.find('div', {'class': 'views-field-field-clubsnsw-address-line-2'}).text.strip()
            city = company.find('div', {'class': 'views-field-field-clubsnsw-city'}).text.strip()
            state = company.find('div', {'class': 'views-field-field-clubsnsw-state'}).text.strip()
            phone = company.find('div', {'class': 'views-field-field-clubsnsw-phone'}).text.strip()
            website = company.find('div', {'class': 'views-field-field-clubsnsw-web-site'}).text.strip()
            companies_data.append({
                'name': name,
                'address_line1': address_line1,
                'address_line2': address_line2,
                'city': city,
                'state': state,
                'phone': phone,
                'website': website
            })

        next_link = soup.find('a', {'title': 'Go to next page'})
        if next_link is None:
            break

        next_url = urljoin(start_url, next_link['href'])

        response = requests.get(next_url)

        soup = BeautifulSoup(response.content, 'html.parser')

    return pd.DataFrame(companies_data)



def generate_descriptions(companies_data, api_key, model_engine):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    descriptions = []
    for _, company in companies_data.iterrows():
        prompt = f"Generate a short description for {company['name']} located at {company['address_line1']}, {company['address_line2']}, {company['city']}, {company['state']}."
        
        while True:
            try:
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt,
                    max_tokens=7,
                    n=1,
                    stop=None,
                    temperature=0.5,
                )
                break
            except Exception as e:
                print(e)
                time.sleep(500)

        description = response.choices[0].text.strip()
        descriptions.append(description)

    # Add descriptions to companies data
    companies_data_with_descriptions = companies_data.copy()
    companies_data_with_descriptions['description'] = descriptions

    return companies_data_with_descriptions



def main():
    start_url = os.getenv("API_URL")
    
    companies_data = fetch_companies_data(start_url)
    df = pd.DataFrame(companies_data)

    df.to_csv('companies_data.csv', index=False)

    api_key = os.getenv("OPENAI_API_KEY")
    model_engine = "text-davinci-003"  

    companies_data_with_descriptions = generate_descriptions(companies_data, api_key, model_engine)

    # Export the updated DataFrame to a CSV file
    companies_data_with_descriptions.to_csv("companies_data_with_descriptions.csv", index=False)


if __name__ == '__main__':
    main()
