import pickle

package = pickle.load(
    open(
        r"models\finlora_random_forest_deployment_package.pkl",
        "rb"
    )
)

print("Model features:")
print(package["model_features"])

print("\nThreshold:")
print(package["threshold"])

print("\nPreprocessor:")
print(package["preprocessor"])

print("\nPreprocessor transformers:")
for name, transformer, columns in package["preprocessor"].transformers_:
    print("\nNAME:", name)
    print("COLUMNS:", columns)

    if hasattr(transformer, "categories_"):
        print("CATEGORIES:")
        for category in transformer.categories_:
            print(category)