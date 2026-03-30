import pandas as pd
import numpy as np
import os

def generate_professional_mapping():
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # 1. Attempt to load your metadata (created in pre-process.py)
    try:
        df = pd.read_csv("data/food_item_metadata.csv")
    except FileNotFoundError:
        print("⚠️ food_item_metadata.csv not found in data/. Creating dummy range for testing...")
        # Create dummy data with dept_ids if metadata is missing
        data = []
        for i in [1, 2, 3]:
            for j in range(1, 250):
                item_id = f'FOODS_{i}_{str(j).zfill(3)}'
                data.append({'item_id': item_id, 'dept_id': f'FOODS_{i}', 'cat_id': 'FOODS'})
        df = pd.DataFrame(data)

    # High-quality mock brands
    brands = [
        "Great Value", "Marketside", "Organic Valley", "Nestle", "Kelloggs", 
        "Kraft", "General Mills", "Dole", "Chobani", "Tyson", "Horizon", 
        "Silk", "Stonyfield", "Annie's", "Nature Valley", "Quaker"
    ]

    # Department lists (30+ items each)
    f1_list = [
        "Whole Milk (1 gal)", "2% Reduced Fat Milk", "Large Grade A Eggs (12ct)", 
        "Greek Strawberry Yogurt", "Unsalted Sweet Cream Butter", "Cheddar Cheese Block", 
        "String Cheese Snacks", "Sour Cream (16oz)", "Heavy Whipping Cream", 
        "Almond Milk (Unsweetened)", "Cream Cheese Spread", "Oat Milk (Barista Blend)",
        "Cottage Cheese (Low Fat)", "Orange Juice (No Pulp)", "Probiotic Drink",
        "Liquid Egg Whites", "Half & Half Creamer", "Margarine Tub", 
        "Shredded Mozzarella", "Swiss Cheese Slices", "Vanilla Bean Yogurt", 
        "Pepper Jack Cubes", "Salted Butter Sticks", "Blueberry Yogurt 4pk",
        "Chocolate Milk", "Soy Milk (Vanilla)", "Cage Free Brown Eggs", 
        "Ricotta Cheese", "Parmesan Shaker", "Whipped Topping"
    ]

    f2_list = [
        "Chicken Breast (Boneless)", "Ground Beef (93% Lean)", "Frozen Pepperoni Pizza", 
        "Beef Patties (1/4 lb)", "Smoked Turkey Deli Slices", "Frozen Broccoli Florets", 
        "Pork Sausage Links", "Frozen Mixed Berries", "Atlantic Salmon Fillet", 
        "Frozen Chicken Nuggets", "Honey Ham Slices", "Beef Chuck Roast", 
        "Frozen French Fries", "Mixed Vegetable Medley", "Breakfast Burritos", 
        "Meatball Sub Kit", "Corned Beef", "Frozen Waffles", 
        "Tilapia Fillets", "Turkey Bacon", "Hot Dogs (8ct)", 
        "Frozen Spinich", "Ice Cream (Vanilla Bean)", "Salami Trio", 
        "Raw Shrimp (Peeled)", "Frozen Lasagna", "Pork Loin Chops", 
        "Breaded Fish Sticks", "Frozen Peaches", "Veggie Burgers"
    ]

    f3_list = [
        "Organic Bananas (Bunch)", "Red Gala Apples", "Sliced White Bread", 
        "Potato Chips (Classic)", "Classic Cola 2L", "Baby Carrots Bag", 
        "Whole Wheat Bread", "Peanut Butter (Creamy)", "Strawberry Jam", 
        "Russet Potatoes (5lb)", "White Rice (2lb)", "Yellow Onions", 
        "Tortilla Chips", "Sparkling Water (Lime)", "Honey Nut Cereal", 
        "Marinara Pasta Sauce", "Spaghetti Noodles", "Granola Bars (Oats)", 
        "Canned Sweet Corn", "Black Beans (15oz)", "Iceberg Lettuce", 
        "Roma Tomatoes", "Chocolate Chip Cookies", "Roasted Peanuts", 
        "Instant Coffee", "Green Seedless Grapes", "Avocados (Hass)", 
        "Bagged Salad Mix", "Pretzels", "Fruit Snacks", "English Muffins"
    ]

    def get_name(item_id):
        try:
            # Extract number from end of ID (e.g., FOODS_3_090 -> 90)
            num = int(item_id.split('_')[-1])
        except:
            num = np.random.randint(1, 100)
            
        brand = brands[num % len(brands)]
        
        if "FOODS_1" in item_id:
            return f"{brand} {f1_list[num % len(f1_list)]}"
        elif "FOODS_2" in item_id:
            return f"{brand} {f2_list[num % len(f2_list)]}"
        else: # FOODS_3
            return f"{brand} {f3_list[num % len(f3_list)]}"

    # 2. APPLY THE MAPPING TO THE DATAFRAME
    print("Generating names...")
    df['product_name'] = df['item_id'].apply(get_name)

    # 3. SAVE THE FINAL MAPPING
    # Ensure we include dept_id so the FastAPI filter doesn't crash
    output_cols = ['item_id', 'dept_id', 'product_name']
    df[output_cols].to_csv("data/product_mapping.csv", index=False)
    
    print(f"✅ Mapping complete. Generated {len(df)} names.")
    print(f"✅ Saved to data/product_mapping.csv with columns: {output_cols}")
    
    return df

if __name__ == "__main__":
    mapping_df = generate_professional_mapping()