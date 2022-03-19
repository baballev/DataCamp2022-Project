

filtered_categories = ["Beverages", "Sweet snacks", "Dairies", "Cereals and potatoes", "Meats", "Fermented foods", "Fermented milk products", 
                    "Groceries", "Meals", "Cereals and their products", "Cheeses", "Sauces", "Spreads", "Confectioneries", "Prepared meats", 
                    "Frozen foods", "Breakfasts", "Desserts", "Canned foods", "Seafood", "Cocoa and its products", "Fats", "Condiments", 
                    "Fishes", "Breads", "Yogurts", "Cakes", "Biscuits", "Pastas", "Legumes"]
filtered_categories = [s.lower() for s in filtered_categories]

CLASS_TO_INDEX = {filtered_categories[i]:i for i in range(len(filtered_categories))}
INDEX_TO_CLASS = {i:filtered_categories[i] for i in range(len(filtered_categories))}

class ImageClassifier():
    def __init__(self) -> None:
        print("hello")


    def fit(self, img_loader):
        print(img_loader)
        return self

    def predict_proba(self, img_loader):
        pass