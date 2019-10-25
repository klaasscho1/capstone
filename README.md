# Capstone - Klaas Schoenmaker
**Supervisor:** M.P.J. Repplinger, Universiteit van Amsterdam (mpjrepplinger@gmail.com)

Topic: Predicting News Article Framing with Natural Language Processing

## Scripts

### Scraper

There are a number of scripts contained in this repository, used for retrieving and analysing 
the articles from the Media Frames Corpus (MFC). Here are a few of them, highlighted:

```sh
$ python scraper/scrape_first.py
```
Gets the titles of all the articles from the MFC, searches for them on Nexis Uni, and downloads 
them as a compressed PDF.
Saves the results from scraping in the `results/` folder, where JSON files for every article
will put.
Takes approximately 30 hours to complete.

```sh
$ python scraper/unzip_downloads.py
```
Takes all the ZIP files from your Downloads folder and decompresses them into the `unzipped-articles/`
folder in the project root.

```sh
$ python scraper/datify_pdf_articles.py
```
Takes the unzipped PDF files in the 'unzipped-articles' folder, converts them to plain text, and 
extracts the title and body from them. JSON files containing these will be saved in the `article-contents/`
folder in the project root.

```sh
$ python scraperunify_data_sources.py
```
Takes the annotation data from the MFC, the scraping results, and the article contents, and combines them
into one unified JSON file called `data.json`

### Model
```sh
$ python model.py
```
The main logic for vectorizing and clustering the article data.
