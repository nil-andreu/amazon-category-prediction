# Product Classification

In this test we ask you to build a model to classify products into their categories according to their features.

## Dataset Description

The dataset is a simplified version of [Amazon 2018](https://jmcauley.ucsd.edu/data/amazon/), only containing products and their descriptions.

The dataset consists of a jsonl file where each is a json string describing a product.

Example of a product in the dataset:

```json
{
 "also_buy": ["B071WSK6R8", "B006K8N5WQ", "B01ASDJLX0", "B00658TPYI"],
 "also_view": [],
 "asin": "B00N31IGPO",
 "brand": "Speed Dealer Customs",
 "category": ["Automotive", "Replacement Parts", "Shocks, Struts & Suspension", "Tie Rod Ends & Parts", "Tie Rod Ends"],
 "description": ["Universal heim joint tie rod weld in tube adapter bung. Made in the USA by Speed Dealer Customs. Tube adapter measurements are as in the title, please contact us about any questions you
may have."],
 "feature": ["Completely CNC machined 1045 Steel", "Single RH Tube Adapter", "Thread: 3/4-16", "O.D.: 1-1/4", "Fits 1-1/4\" tube with .120\" wall thickness"],
 "image": [],
 "price": "",
 "title": "3/4-16 RH Weld In Threaded Heim Joint Tube Adapter Bung for 1-1/4&quot; Dia by .120 Wall Tube",
 "main_cat": "Automotive"
}
```

### Field description

- also_buy/also_view: IDs of related products
- asin: ID of the product
- brand: brand of the product
- category: list of categories the product belong to, usually in hierarchical order
- description: description of the product
- feature: bullet point format features of the product
- image: url of product images (migth be empty)
- price: price in US dollars (might be empty)
- title: name of the product
- main_cat: main category of the product

`main_cat` can have one of the following values:

```json
[
  "All Electronics",
  "Amazon Fashion",
  "Amazon Home",
  "Arts, Crafts & Sewing",
  "Automotive",
  "Books",
  "Buy a Kindle",
  "Camera & Photo",
  "Cell Phones & Accessories",
  "Computers",
  "Digital Music",
  "Grocery",
  "Health & Personal Care",
  "Home Audio & Theater",
  "Industrial & Scientific",
  "Movies & TV",
  "Musical Instruments",
  "Office Products",
  "Pet Supplies",
  "Sports & Outdoors",
  "Tools & Home Improvement",
  "Toys & Games",
  "Video Games"
]
```

[Download dataset](https://drive.google.com/file/d/1Zf0Kdby-FHLdNXatMP0AD2nY0h-cjas3/view?usp=sharing)

Data can be read directly from the gzip file as:

```python
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)
```

## Task description

The solution explained inside of [solution](solution) folder.

- You should create a model that predicts `main_cat` using any of the other fields except `category` list. The model should be developed in python, you can use any pre-trained models and thirdparty
  libraries you need (for example huggingface).

- You should create a HTTP API endpoint that is capable of performing inference and return the predicted `main_cat` when receiving the rest of product fields.

- Both the training code (if needed) and the inference API should be dockerized and easy for us to run and test locally (only docker build and docker run commands should be necessary to perform training
  or setting up the inference API).

- You should also provide a detailed analysis of the performance of your model. **In this test we're not looking at the your model performance but we expect a good understanding of your solution
  performance and it's limitations**.

- Answer the following questions:
  - What would you change in your solution if you needed to predict all the categories?
  - If this model was deployed to categorize products without any supervision which metrics would you check to detect data drifting? When would you need to retrain?
