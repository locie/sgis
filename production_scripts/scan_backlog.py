import download_from_RNB_api as rnb

def loop_deparments():
    for dep_code in range(1, 100):
        print(f"Department: {dep_code}")
        rnb.get_csv_metadata(dep_code)