# ClubsNSW WebScrapping

This project is done using Python's BeautifulSoup4, requests library and have used OpenAI API to get the description of the clubs.
In the second part, i have used Google Maps API(Places API) to get the business's website address and phone number. Proper test cases are written to ensure that the scrapped data is correct and complete.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install this project.

```bash
pip install -r requirements.txt
```
Don't forget to add your API SECRET KEY in the .env file
create a .env file 
1) add OPENAI_API_KEY=<API-KEY>
2) add API_URL=<CLUBS_WEBSITE_URL>
3) add GOOGLE_MAPS_API_KEY=<GOOGLE_MAPS_API_KEY>

## Usage
To Run the project, simply run this command

```python
python company_scrapper.py
```
## Output
You can expect a CSV file with the details of the Club if it passes all the test cases.

