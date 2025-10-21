# D:\Portfolio_Project\main.py

from scripts.data_cleaner import clean_and_resample_single, process_all_files

if __name__ == "__main__":
   raw_dir = r"D:\Portfolio Optimzation project\raw_data"
   processed_dir = r"D:\Portfolio Optimzation project\processed_data"

    #  Option 1: To test on one file
   clean_and_resample_single(rf"{raw_dir}\HINDUNILVR.csv", processed_dir)


    #  Option 2: To process all files
   process_all_files(raw_dir, processed_dir)
