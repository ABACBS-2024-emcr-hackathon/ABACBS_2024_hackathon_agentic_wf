import GEOparse
from collections import defaultdict

# Load the dataset
# GSE12345
# GSE275471 (has treatment)
# GSE265975 (no treatment)
geo_accession = "GSE265975"  #replace with your GEO accession ID
gse = GEOparse.get_GEO(geo=geo_accession)

# Explore metadata to identify key variables
for gsm_name, gsm in gse.gsms.items():
    print(f"Sample {gsm_name} metadata:")
    print(gsm.metadata)

# Group samples based on a specific metadata field, such as "disease status" or "treatment"
# Let's assume we find a variable in `characteristics_ch1` called "treatment"
groups = defaultdict(list)

for gsm_name, gsm in gse.gsms.items():
    for characteristic in gsm.metadata.get("characteristics_ch1", []):
        if "treatment" in characteristic:
            treatment = characteristic.split(": ")[1]  # e.g., "treated" or "untreated"
            groups[treatment].append(gsm_name)

# Output groups for comparisons
for treatment, samples in groups.items():
    print(f"Group '{treatment}' has samples: {samples}")
