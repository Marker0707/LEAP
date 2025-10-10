import leap
from leap.llm_client import Gemma3Client

content = """
Chief Complaint

Progressive weight gain and excessive eating behavior for the past 2 years.

History of Present Illness

The patient is a 7-year-old boy brought in by his parents with concerns of abnormal weight gain despite attempts at dietary restriction. Since around the age of 5, he has exhibited persistent food-seeking behavior, including stealing food and overeating. Parents also report decreased muscle tone and delayed motor milestones in infancy. Speech development was slower than peers, but he is now able to communicate in short sentences. Cognitive function appears mildly delayed.

Past Medical History

Birth history: Born at 39 weeks via normal vaginal delivery, birth weight 2.6 kg. Notable hypotonia during the neonatal period, requiring feeding assistance.

Development: Sat at 12 months, walked independently at 24 months. Speech onset delayed to 3 years of age.

No known drug allergies.

Family History

Non-consanguineous marriage. No family history of genetic syndromes, intellectual disability, or early-onset obesity.

Social History

Lives with parents and one older sibling (healthy). Attends special education school due to learning difficulties. Parents report behavioral problems, including stubbornness and temper outbursts.

Physical Examination

General appearance: Obese child with characteristic facial features (almond-shaped eyes, narrow bifrontal diameter, downturned mouth).

Anthropometry: Height: 112 cm (<10th percentile), Weight: 35 kg (>97th percentile), BMI: >99th percentile.

Neurological: Generalized hypotonia, mild developmental delay, normal reflexes.

Genital examination: Micropenis and small testes.

Other findings: Small hands and feet, fair skin compared to family members.

Investigations

Genetic testing: Confirmed deletion on chromosome 15q11-q13 (paternal origin).

Hormonal evaluation: Growth hormone deficiency, hypogonadism (low testosterone).

Other labs: Normal thyroid function, fasting glucose slightly elevated.
"""
content = "headache"

obj = leap.LEAP(
    model="../phenolink_eval/models/sentence-transformers-all-MiniLM-L12-v2-2025-03-26_21-10-23/final",
    llm_client=Gemma3Client(endpoint="http://10.200.232.11:11433/api/generate"),
    hp_obo_path="../phenolink_eval/data/hp.obo",
)


result = obj.convert_ehr(
    content="headache",
    rerank=False,
    rerank_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
    furthest=True,
    use_weighting=False,
    retrieve_cutoff=0.7,
    rerank_cutoff=0.1
)
print(result.final_leap_result)