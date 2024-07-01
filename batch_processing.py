import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

class RecipeProcessor:
    def __init__(self, batch, ingr_mapping_df, ingredient_dic):
        self.batch = batch
        self.ingr_mapping_df = ingr_mapping_df
        self.ingredient_dic = ingredient_dic
        self.replaced_list = []

    def clean_ingredient(self, ingredient):
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', ingredient).lower().replace(' ', '_')
        return cleaned.strip() if len(cleaned.strip()) >= 2 else None

    def process_and_clean_ner_column(self, ner_list):
        try:
            if isinstance(ner_list, str):
                ner_list = eval(ner_list)
            if isinstance(ner_list, list):
                cleaned_ingredients = list(set(self.clean_ingredient(ingredient) for ingredient in ner_list if self.clean_ingredient(ingredient) is not None))
                return cleaned_ingredients
            else:
                return ner_list
        except Exception as e:
            print(f"Error processing list: {e}")
            return ner_list

    def replace_with_keys(self, ner_list):
        replaced_list = []
        for ingredient in ner_list:
            replaced = False
            for key, values in self.ingredient_dic.items():
                if ingredient in values:
                    replaced_list.append(key)
                    replaced = True
                    break
            if not replaced:
                replaced_list.append(ingredient)
        return replaced_list

    def preprocess_data(self):
        self.batch['NER'] = self.batch['NER'].apply(self.process_and_clean_ner_column)
        self.batch.columns = self.batch.columns.astype(str)
        self.batch['NER'] = self.batch['NER'].apply(lambda x: self.replace_with_keys(x))

    def calculate_total_ingredients(self):
        self.batch['total_ingredients'] = self.batch['NER'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    def update_ingredient_columns(self):
        ingredient_list = self.ingr_mapping_df['ingredient_name'].unique()
        new_columns_ingr_df = pd.DataFrame(0, index=self.batch.index, columns=ingredient_list)
        self.batch = pd.concat([self.batch, new_columns_ingr_df], axis=1)

        for index, row in self.batch.iterrows():
            ingredients_list = row['NER']
            for ingredient in ingredients_list:
                if ingredient in self.batch.columns:
                    self.batch.at[index, ingredient] = 1

    def calculate_matched_ingredients(self):
        ingredient_cols = [col for col in self.batch.columns if col not in ['Unnamed: 0', 'title', 'ingredients', 'directions', 'link', 'source', 'NER', 'total_ingredients', 'matched_ingredients', 'matched_percentage']]
        self.batch['matched_ingredients'] = self.batch[ingredient_cols].sum(axis=1)

    def filter_recipes(self):
        self.batch['matched_percentage'] = self.batch['matched_ingredients'] / self.batch['total_ingredients'] * 100
        self.batch = self.batch[self.batch['matched_percentage'] >= 60]

    def sort_columns(self):
        existing_columns = self.batch.columns.tolist()
        desired_columns = ['Unnamed: 0', 'title', 'ingredients', 'directions', 'link', 'source', 'NER', 'total_ingredients', 'matched_ingredients', 'matched_percentage']
        remaining_columns = [col for col in existing_columns if col not in desired_columns]
        new_columns_order = desired_columns + remaining_columns
        self.batch = self.batch[new_columns_order]

    def run_pipeline(self):
        try:
            self.preprocess_data()
            self.calculate_total_ingredients()
            self.update_ingredient_columns()
            self.calculate_matched_ingredients()
            self.filter_recipes()
            self.sort_columns()
        except Exception as e:
            print(f"Error in pipeline: {e}")

def process_batch(batch, ingr_mapping_df, ingredient_dic):
    processor = RecipeProcessor(batch, ingr_mapping_df, ingredient_dic)
    processor.run_pipeline()
    return processor.batch

#%%
