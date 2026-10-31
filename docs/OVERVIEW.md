# What this project does (in plain English)

If you have never worked on a project like this, start here. No programming
knowledge is assumed.

## The problem

Used-car prices are hard to judge. Two cars that look identical on paper can
be listed hundreds or thousands of dollars apart, and the "right" price changes
week to week. Sellers want to know what to ask; buyers want to know whether an
ad is a bargain or a rip-off.

## The idea

Websites such as Kijiji, AutoTrader and Facebook Marketplace publish thousands
of used-vehicle ads. Each ad is a tiny data point: *"this 2016 Honda Civic with
120,000 km was listed for $14,500 on this date."* One ad tells you little.
Tens of thousands of ads tell you a lot.

**Vehicle Intel** is a pipeline that:

1. **Collects** listings automatically from several websites.
2. **Remembers** them in a database, including every price change and the moment
   a listing disappears (which usually means the car sold).
3. **Sorts** each listing into *private seller*, *dealer* or *salvage auction*, because those three
   markets price cars very differently.
4. **Reads** the ad text and, optionally, the photos with a local AI model to
   judge condition (rust, accident hints, interior wear, and so on).
5. **Learns** from all of that history by training price-prediction models.
6. **Answers questions** such as *"what is a 2019 Toyota Camry with 80,000 km worth?"* with
   a predicted price, a likely range and similar cars currently for sale.

Everything runs on your own computer. Nothing is sent to a paid API; the AI
model runs locally through [Ollama](https://ollama.com).

## The pipeline at a glance

```
 ┌────────────┐   ┌───────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
 │  Scrapers  │ → │  Ingest   │ → │  Database  │ → │ AI condition│ → │  Training  │
 │ Kijiji     │   │ clean up, │   │ (Postgres) │   │ reading     │   │  price     │
 │ AutoTrader │   │ classify, │   │ listings + │   │ (optional)  │   │  models    │
 │ Facebook   │   │ dedupe,   │   │ history    │   │             │   │            │
 │ stat.vin   │   │ track     │   │            │   │             │   │            │
 └────────────┘   │ changes   │   └────────────┘   └────────────┘   └─────┬──────┘
                  └───────────┘                                            │
                                                                           ▼
                                                              "What is this car worth?"
                                                               (predicted price, range,
                                                                comparable listings)
```

A **scheduler** can run all of this on a timetable so the database keeps
filling itself while you do other things.

## Walk-through: what happens to one listing

1. The Kijiji scraper downloads a search results page and finds an ad:
   *"2016 Honda Civic EX, 120,000 km, $14,500, one owner, no accidents."*
2. It is turned into a standard record: year 2016, make Honda, model Civic,
   mileage 120000, price 14500, plus the description, photos and location.
3. The **classifier** checks the seller name and wording. No dealership
   language, so it is labelled `private`.
4. **Ingest** looks the listing up by (source, id). It is new, so it is inserted
   and a first *snapshot* (price, status, date) is saved.
5. Next week the seller drops the price to $13,900. The scraper sees the ad
   again, Ingest notices the change, updates the record and stores a second
   snapshot, so you now have a price history.
6. Two weeks later the ad is gone. The **sold sweep** marks it `removed` and records
   how many days it was on the market.
7. Meanwhile the optional **condition extractor** asks the local vision model to
   read the text and look at up to six photos. It writes down things such as "interior: good,
   rust indicated: no" along with its confidence and the evidence it relied on.
8. When you train, the listing becomes one row in the training table: year,
   mileage, make, model, trim, transmission, which keywords appear in the
   description (for example "accident free", "needs work"), and the price it
   asked.

## The four price models

| # | Name | Trained on | Evaluated on |
| --- | --- | --- | --- |
| 1 | `private_csv` | private-seller listings | your own CSV of cars you trust (if you provide one), otherwise a random 15% holdout |
| 2 | `private_traintest` | private-seller listings | the **newest** 15% of listings the model never saw |
| 3 | `dealer_traintest` | dealer listings | the newest 15% |
| 4 | `salvage_traintest` | sold salvage-auction lots | the newest 15% |

Models 2-4 use a stricter, more honest evaluation than Model 1 (see
[MODELS.md](MODELS.md)). Comparing 1 and 2 shows how much a naive evaluation
flatters a model.

## Glossary

| Term | Meaning |
| --- | --- |
| **Listing** | One ad for one vehicle on one website. |
| **Scraper** | A program that reads a website and extracts data from it. |
| **Source** | Where a listing came from: `kijiji`, `autotrader`, `facebook`, `statvin`. |
| **Snapshot** | A saved copy of a listing's price/status at one moment, used for history. |
| **Sold sweep** | A pass that marks listings that vanished from a site as `removed`. |
| **Listing type** | `private`, `dealer` or `salvage`. |
| **Fingerprint** | A short hash of year + make + model + mileage bucket + city, used to spot the same car posted twice. |
| **VIN** | The 17-character vehicle identification number, unique per car. |
| **Salvage** | A vehicle written off by an insurer, usually sold at auction. |
| **Ollama** | A program that runs AI models locally on your machine. |
| **VLM** | Vision-language model: an AI that reads both text and pictures. |
| **XGBoost** | The machine-learning algorithm used for price prediction. |
| **MAE** | Mean absolute error: on average, how many dollars the prediction is off. |
| **R²** | How much of the price variation the model explains (1.0 is perfect, 0 is no better than guessing the average). |
| **Migration** | A numbered SQL file that changes the database structure in a controlled way. |
| **Docker / Compose** | Tools that run the database and AI server in ready-made containers so you do not install them by hand. |

## What this project is *not*

- **Not a finished product.** It is a data and modelling toolkit. There is no
  website or app, only a command-line interface.
- **Not magic.** Prediction quality depends entirely on how much clean data you
  collect. With a few dozen listings the models will be poor; the training
  commands refuse to run with fewer than 50 usable rows.
- **Not set up for every city.** The default search URLs target **Calgary,
  Alberta**. See [ADVANCED_SETUP.md](ADVANCED_SETUP.md#changing-the-city-or-search) to change that.
- **Not a licence to scrape anything.** Every site has terms of service. See the
  [responsible use](../README.md#responsible-use) notes.

## Where to go next

- Want to run it? → [GETTING_STARTED.md](GETTING_STARTED.md)
- Want to understand each part? → [MODULES.md](MODULES.md)
- Want to know every setting? → [CONFIGURATION.md](CONFIGURATION.md)
- Want production-style setup? → [ADVANCED_SETUP.md](ADVANCED_SETUP.md)
