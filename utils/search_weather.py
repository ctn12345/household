import requests

dataset_url = "https://www.data.gouv.fr/api/1/datasets/donnees-climatologiques-de-base-mensuelles/"
meta = requests.get(dataset_url, timeout=60).json()

resources = meta["resources"]

target_deps = ["75", "92"]

for dep in target_deps:
    print(f"\n========== 搜索部门 {dep} ==========")

    found = 0

    for r in resources:
        title = r.get("title", "") or ""
        url = r.get("url", "") or ""
        rid = r.get("id", "") or ""
        text = title + " " + url

        if f"MENSQ_{dep}_" in text:
            found += 1
            print("resource_id:", rid)
            print("title:", title)
            print("url:", url)
            print("-" * 80)

    print(f"部门 {dep} 匹配数量:", found)