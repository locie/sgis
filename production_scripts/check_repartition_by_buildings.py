from pathlib import Path
import pandas as pd
# import geopandas as gpd
# from shapely import wkt

CSV_DIR = Path("/home/thebaulm/split/31/2025/rasters/repartition_by_rasters")
IMAGES_DIR = Path("/home/thebaulm/split/31/2025/rasters/images/")
RNB_CSV = "/media/thebaulm/LaCie_14TB/thebaulm/gis/vectors/cadastre/2026-04-04/unzipped/cadastre-31-batiments-csv/RNB_31.csv"


frames = []
for csv in sorted(CSV_DIR.glob("*.csv")):
    df = pd.read_csv(csv, header=None, names=["saved", "original"], dtype=str)
    # drop a header row if the file has one
    df = df[~df["saved"].str.lower().isin(["suffixed file name", "saved"])]
    df["csv"] = csv.name
    frames.append(df)

expected = pd.concat(frames, ignore_index=True)
on_disk = {p.name for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"}

missing = expected[~expected["saved"].isin(on_disk)]
extra = sorted(on_disk - set(expected["saved"]))
dups = expected[expected.duplicated("saved", keep=False)].sort_values("saved")

print(f"CSV files read:       {len(frames)}")
print(f"Expected images:      {len(expected)} ({expected['saved'].nunique()} unique)")
print(f"Images on disk:       {len(on_disk)}")
print(f"Missing on disk:      {len(missing)}")
print(f"Extra on disk:        {len(extra)}")
print(f"Duplicated in CSVs:   {dups['saved'].nunique()}")

if not missing.empty:
    print("\nMissing per CSV:")
    print(missing.groupby("csv").size().sort_values(ascending=False).to_string())
    missing.to_csv("missing_images.csv", index=False)
    
    missing = missing.copy()
    missing["ID"] = missing["original"].str.replace(r"\.jpg$", "", regex=True, case=False)

    ids = sorted(missing["ID"].unique())
    print(f"\n{len(ids)} building IDs missing:")
    for building_id in ids:
        print(building_id)

    # optional: also show which raster (CSV) each ID belongs to
    print("\nID -> raster:")
    print(missing[["ID", "saved", "csv"]].sort_values("csv").to_string(index=False))
    
    
    ###############################################################################
    # check geometries
    ###############################################################################
    # 1) Clean IDs from the missing list (`missing` comes from the previous script)
    missing = missing.copy()
    missing["rnb_id"] = (missing["original"]
                        .str.replace(r"\.jpg$", "", regex=True, case=False)
                        .str.strip())
    ids = set(missing["rnb_id"])

    # 2) Read only the two needed columns, keep only the missing buildings
    rnb = pd.read_csv(RNB_CSV, usecols=["rnb_id", "shape"], dtype=str)
    rnb = rnb[rnb["rnb_id"].isin(ids)].drop_duplicates("rnb_id")

    # 3) Merge (left join keeps every missing row, even without a match)
    merged = missing.merge(rnb, on="rnb_id", how="left")

    not_found = merged[merged["shape"].isna()]
    print(f"Missing rows: {len(merged)} | unique IDs: {len(ids)}")
    print(f"Geometry found: {merged['shape'].notna().sum()} | not found in RNB_31.csv: {not_found['rnb_id'].nunique()}")
    if not not_found.empty:
        print("Not found:", sorted(not_found["rnb_id"].unique())[:20])
        
    # merged comes from step 3 (missing left-joined with RNB_31.csv)
    table = (merged[["rnb_id", "saved", "csv", "shape"]]
            .rename(columns={"rnb_id": "ID", "csv": "raster"})
            .sort_values(["raster", "ID"])
            .reset_index(drop=True))

    # compact display: WKT truncated to 60 characters
    show = table.copy()
    show["shape"] = show["shape"].str.slice(0, 60) + "..."
    print("ID -> raster:")
    print(show.to_string(index=False))


    # # 4) WKT -> geometry, then reproject to Lambert-93
    # found = merged.dropna(subset=["shape"]).copy()
    # found["geometry"] = found["shape"].apply(wkt.loads)
    # gdf = gpd.GeoDataFrame(found.drop(columns="shape"), geometry="geometry", crs="EPSG:4326")
    # gdf = gdf.to_crs(2154)

    # print(gdf[["rnb_id", "saved", "csv"]].head())
    # print("Invalid geometries:", (~gdf.is_valid).sum(), "| empty:", gdf.is_empty.sum())

    
    
    
    
    
else:
    if missing.empty:
     print("No missing images.")

if extra:
    pd.Series(extra, name="image").to_csv("extra_images.csv", index=False)
if not dups.empty:
    dups.to_csv("duplicated_expected_names.csv", index=False)