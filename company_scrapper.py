import requests
import pandas as pd
from bs4 import BeautifulSoup
import openai
import time
import os
from dotenv import load_dotenv
from urllib.parse import urljoin
import googlemaps

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

def test_cases_scrapped_data(start_url):
    companies_data = fetch_companies_data(start_url)
    for _, company in companies_data.iterrows():
        assert company['name'] != '', "Company name missing"
        assert company['address_line1'] != '', "Address line 1 missing"
        assert company['address_line2'] != '', "Address line 2 missing"
        assert company['city'] != '', "City missing"
        assert company['state'] != '', "State missing"
        assert company['phone'] != '', "Phone number missing"
        phone = company['phone']
        assert phone.startswith('0'), f"Invalid phone number format: {phone}"
        assert len(phone) == 10, f"Invalid phone number length: {phone}"
        assert phone.isdigit(), f"Phone number contains non-numeric characters: {phone}"
        website = company['website']
        if website != '':
            assert website.startswith('http'), f"Invalid website URL format: {website}"
            assert requests.get(website).status_code == 200, f"Invalid website URL: {website}"



def search_business(query):
    gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))

    # Geocoding an address
    geocode_result = gmaps.geocode(query)

    # Search for business based on geolocation
    location = geocode_result[0]['geometry']['location']
    places_result = gmaps.places_nearby(location=location, radius=500, type='establishment')

    # Find the first result that matches the query
    for result in places_result['results']:
        if result['name'] == query:
            # Get details for the business
            place_id = result['place_id']
            place_result = gmaps.place(place_id=place_id)
            website = place_result['result']['website']
            phone_number = place_result['result']['formatted_phone_number']
            return website, phone_number

    return None, None


def main():
    start_url = os.getenv("API_URL")
    
    companies_data = fetch_companies_data(start_url)

    # test_cases_scrapped_data(start_url)

    df = pd.DataFrame(companies_data)

    websites = []
    phone_numbers = []
    for _, company in df.iterrows():
        query = f"{company['name']}, {company['address_line1']}, {company['address_line2']}, {company['city']}, {company['state']}"
        website, phone_number = search_business(query)
        websites.append(website)
        phone_numbers.append(phone_number)

    df['website'] = websites
    df['phone'] = phone_numbers

    # Test Case 1
    # Test with a valid business query
    query = "Starbucks, New York"
    expected_website = "https://www.starbucks.com/"
    expected_phone_number = "+1 212-989-4016"
    assert search_business(query) == (expected_website, expected_phone_number)

    # Test Case 2
    # Test with a valid business query but no website or phone number available
    query = "Statue of Liberty, New York"
    expected_website = None
    expected_phone_number = None
    assert search_business(query) == (expected_website, expected_phone_number)

    # Test Case 3
    # Test with an invalid business query
    query = "Some random business"
    expected_website = None
    expected_phone_number = None
    assert search_business(query) == (expected_website, expected_phone_number)

    # Test Case 4
    # Test with a query containing special characters
    query = "McDonald's, San Francisco"
    expected_website = "https://www.mcdonalds.com/us/en-us.html"
    expected_phone_number = "+1 415-864-0337"
    assert search_business(query) == (expected_website, expected_phone_number)

    # Test Case 5
    # Test with a query containing non-ASCII characters
    query = "Café du Monde, New Orleans"
    expected_website = "https://www.cafedumonde.com/"
    expected_phone_number = "+1 504-581-2914"
    assert search_business(query) == (expected_website, expected_phone_number)


    df.to_csv('companies_data.csv', index=False)

    api_key = os.getenv("OPENAI_API_KEY")
    model_engine = "text-davinci-003"  

    companies_data_with_descriptions = generate_descriptions(companies_data, api_key, model_engine)

    # Export the updated DataFrame to a CSV file
    companies_data_with_descriptions.to_csv("companies_data_with_descriptions.csv", index=False)


if __name__ == '__main__':
    main()
